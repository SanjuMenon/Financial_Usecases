from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


@dataclass(frozen=True)
class PipelineConfig:
    seed: int = 7
    test_size: float = 0.2
    artifacts_dir: Path = Path("artifacts")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


FEATURE_COLS = [
    "market_beta",
    "price_to_book",
    "sprd_rtn",
    "is_director",
    "prc_op_earnings_basic",
    "hml_beta",
    "acq_disp",
    "ret",
    "is_officer",
    "ten_percent_owner",
]


def _make_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    X = pd.get_dummies(X, columns=["acq_disp"], drop_first=True)
    return X


def _write_markdown_report(
    *,
    cfg: PipelineConfig,
    title: str,
    model_metrics: dict[str, object],
    shap_global: pd.DataFrame,
    causal_effects: pd.DataFrame,
    top_cases: pd.DataFrame,
    out_path: Path,
) -> Path:
    from tabulate import tabulate

    lines: list[str] = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append("### Model performance (holdout)")
    lines.append(f"- **ROC AUC**: {model_metrics['auc']:.4f}")
    lines.append("")

    # classification report (compact)
    rep = model_metrics["report"]
    for k in ["0", "1", "accuracy"]:
        if k not in rep:
            continue
        if k == "accuracy":
            lines.append(f"- **Accuracy**: {rep[k]:.4f}")
        else:
            lines.append(
                f"- **Class {k}**: precision={rep[k]['precision']:.3f}, recall={rep[k]['recall']:.3f}, f1={rep[k]['f1-score']:.3f}"
            )
    lines.append("")

    lines.append("### SHAP (global drivers)")
    lines.append(
        "Ranked by mean |SHAP| on the holdout set. Higher means the feature more strongly drives predictions."
    )
    lines.append("")
    lines.append(tabulate(shap_global.head(15), headers="keys", tablefmt="github", showindex=False))
    lines.append("")

    lines.append("### Causal forest (summary)")
    lines.append(
        "Mean marginal effects from the causal forest (mocked setup; interpret as direction/relative strength)."
    )
    lines.append("")
    lines.append(tabulate(causal_effects, headers="keys", tablefmt="github", showindex=False))
    lines.append("")

    lines.append("### Top flagged trades (examples)")
    lines.append("Highest `uit_risk` rows with their top driver features.")
    lines.append("")
    lines.append(tabulate(top_cases, headers="keys", tablefmt="github", showindex=False))
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_html_report(
    *,
    title: str,
    model_metrics: dict[str, object],
    shap_global: pd.DataFrame,
    causal_effects: pd.DataFrame,
    top_cases: pd.DataFrame,
    shap_plot_path: Path | None,
    out_path: Path,
) -> Path:
    def _df_html(df: pd.DataFrame) -> str:
        return df.to_html(index=False, escape=True, classes="tbl", border=0)

    img_html = ""
    if shap_plot_path is not None and shap_plot_path.exists():
        # Keep it simple: link + inline <img> from file path relative to report
        rel = shap_plot_path.name
        img_html = f"""
        <div class="card">
          <h3>SHAP beeswarm</h3>
          <div class="muted">Saved as <code>{rel}</code></div>
          <img src="{rel}" alt="SHAP beeswarm" />
        </div>
        """

    rep = model_metrics["report"]
    auc = float(model_metrics["auc"])
    acc = rep.get("accuracy", None)
    c0 = rep.get("0", {})
    c1 = rep.get("1", {})

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      --bg: #0b0f14;
      --card: #111826;
      --text: #e6edf3;
      --muted: #9fb0c0;
      --border: #223043;
      --accent: #7aa2f7;
    }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Noto Sans", "Helvetica Neue";
      background: var(--bg);
      color: var(--text);
      line-height: 1.4;
    }}
    .wrap {{ max-width: 1100px; margin: 28px auto; padding: 0 16px; }}
    h1 {{ margin: 0 0 10px; font-size: 22px; }}
    h2 {{ margin: 22px 0 10px; font-size: 18px; }}
    h3 {{ margin: 0 0 10px; font-size: 15px; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 14px;
    }}
    .kpi {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; }}
    .k {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      background: rgba(0,0,0,0.12);
    }}
    .k .v {{ font-size: 18px; font-weight: 700; }}
    .k .l {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
    .tbl {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      overflow: hidden;
    }}
    .tbl th, .tbl td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 10px;
      vertical-align: top;
      text-align: left;
      white-space: nowrap;
    }}
    .tbl th {{ color: var(--muted); font-weight: 600; }}
    code {{
      background: rgba(122,162,247,0.12);
      border: 1px solid rgba(122,162,247,0.22);
      padding: 1px 6px;
      border-radius: 999px;
      color: var(--text);
    }}
    img {{
      max-width: 100%;
      border-radius: 10px;
      border: 1px solid var(--border);
      margin-top: 10px;
      background: #0a0f17;
    }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{title}</h1>
    <div class="muted">Generated by the UIT pipeline.</div>

    <h2>Model performance (holdout)</h2>
    <div class="card">
      <div class="kpi">
        <div class="k"><div class="v">{auc:.4f}</div><div class="l">ROC AUC</div></div>
        <div class="k"><div class="v">{(acc if acc is not None else float('nan')):.4f}</div><div class="l">Accuracy</div></div>
        <div class="k"><div class="v">{c1.get('recall', float('nan')):.3f}</div><div class="l">Recall (class 1)</div></div>
      </div>
      <div class="muted" style="margin-top:10px">
        Class 0: precision={c0.get('precision', float('nan')):.3f}, recall={c0.get('recall', float('nan')):.3f}, f1={c0.get('f1-score', float('nan')):.3f}<br/>
        Class 1: precision={c1.get('precision', float('nan')):.3f}, recall={c1.get('recall', float('nan')):.3f}, f1={c1.get('f1-score', float('nan')):.3f}
      </div>
    </div>

    <h2>SHAP</h2>
    <div class="grid">
      <div class="card">
        <h3>Global drivers</h3>
        <div class="muted">Ranked by mean |SHAP| on the holdout set.</div>
        {_df_html(shap_global.head(20))}
      </div>
      {img_html}
    </div>

    <h2>Causal forest</h2>
    <div class="card">
      <div class="muted">Mean marginal effects (mocked setup; interpret direction/relative strength).</div>
      {_df_html(causal_effects)}
    </div>

    <h2>Top flagged trades (examples)</h2>
    <div class="card">
      <div class="muted">Highest <code>uit_risk</code> rows with their top driver features.</div>
      {_df_html(top_cases)}
    </div>
  </div>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    return out_path

def run_xgb_shap_causal(form4_trades: pd.DataFrame, cfg: PipelineConfig) -> dict[str, object]:
    """
    Train XGBoost -> SHAP global importance -> econml CausalForestDML ATE (demo).
    """
    _ensure_dir(cfg.artifacts_dir)

    df = form4_trades.dropna(subset=FEATURE_COLS + ["label_uit"]).copy()
    y = df["label_uit"].astype(int).values

    X = _make_X(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=y
    )

    # --- model
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=cfg.seed,
        n_jobs=0,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    report = classification_report(y_test, pred, output_dict=True)
    auc = float(roc_auc_score(y_test, proba))

    # --- SHAP (with robust fallback for some shap/xgboost combos on Windows)
    import shap

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    except Exception:
        # Fallback: use XGBoost native pred_contribs (Shapley-style contributions)
        import xgboost as xgb

        dtest = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
        contrib = model.get_booster().predict(dtest, pred_contribs=True)
        # last column is the bias term
        shap_values = contrib[:, :-1]
    # global importance
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_rank = (
        pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_rank.to_csv(cfg.artifacts_dir / "shap_global_ranking.csv", index=False)

    # plot (optional; works headless with Agg)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(cfg.artifacts_dir / "shap_beeswarm.png", dpi=160)
    plt.close()

    # --- Causal Forest (demo-style)
    # Treat a few "treatments" as the paper does; use remaining as controls.
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestRegressor

    # outcome: UIT label
    Y = y.astype(float)
    # treatments: is_director, price_to_book, market_beta, ret
    T = df[["is_director", "price_to_book", "market_beta", "ret"]].astype(float).values
    # controls: rest (incl one-hot acq_disp)
    W = pd.get_dummies(
        df[
            [
                "sprd_rtn",
                "prc_op_earnings_basic",
                "hml_beta",
                "acq_disp",
                "is_officer",
                "ten_percent_owner",
            ]
        ],
        columns=["acq_disp"],
        drop_first=True,
    ).astype(float)

    est = CausalForestDML(
        model_t=RandomForestRegressor(
            n_estimators=200, min_samples_leaf=20, random_state=cfg.seed, n_jobs=-1
        ),
        model_y=RandomForestRegressor(
            n_estimators=200, min_samples_leaf=20, random_state=cfg.seed, n_jobs=-1
        ),
        n_estimators=600,
        min_samples_leaf=50,
        max_depth=12,
        random_state=cfg.seed,
    )
    # econml's CausalForestDML requires X (features that define heterogeneity); use controls for X in this MVP.
    est.fit(Y, T, X=W.values, W=None)

    # With multi-dimensional, continuous treatments, a robust summary is the mean marginal effect.
    # This is not identical to a 0->1 ATE for each treatment, but is stable for this mocked setup.
    mte = est.const_marginal_effect(X=W.values)  # shape: (n_samples, n_treatments)
    ate = np.asarray(mte).mean(axis=0).reshape(-1)
    ate_df = pd.DataFrame(
        {
            "treatment": ["is_director", "price_to_book", "market_beta", "ret"],
            "mean_marginal_effect": ate,
        }
    )
    ate_df.to_csv(cfg.artifacts_dir / "causal_forest_ate.csv", index=False)

    # Quick "top cases" table from the holdout split using SHAP contributions
    sv = np.asarray(shap_values)
    sv_abs = np.abs(sv)
    idx = np.argsort(-sv_abs, axis=1)[:, :5]
    feat_names = np.array(X_test.columns)
    drivers_per_row = [",".join(feat_names[i].tolist()) for i in idx]

    base_cases = df.loc[X_test.index, ["trade_id", "cik", "permno", "personid", "transaction_date", "acq_disp"]].copy()
    base_cases["uit_risk"] = proba
    base_cases["top_drivers"] = drivers_per_row

    top_n = min(25, len(base_cases))
    top_cases = base_cases.sort_values("uit_risk", ascending=False).head(top_n)
    top_cases_out = cfg.artifacts_dir / "top_flagged_trades.csv"
    top_cases.to_csv(top_cases_out, index=False)

    # Markdown report
    report_out = cfg.artifacts_dir / "uit_report.md"
    _write_markdown_report(
        cfg=cfg,
        title="UIT report",
        model_metrics={"auc": auc, "report": report},
        shap_global=shap_rank,
        causal_effects=ate_df,
        top_cases=top_cases[
            ["trade_id", "cik", "personid", "transaction_date", "uit_risk", "top_drivers"]
        ],
        out_path=report_out,
    )
    html_out = cfg.artifacts_dir / "uit_report.html"
    _write_html_report(
        title="UIT report",
        model_metrics={"auc": auc, "report": report},
        shap_global=shap_rank,
        causal_effects=ate_df,
        top_cases=top_cases[["trade_id", "cik", "personid", "transaction_date", "uit_risk", "top_drivers"]],
        shap_plot_path=(cfg.artifacts_dir / "shap_beeswarm.png"),
        out_path=html_out,
    )

    return {
        "auc": auc,
        "report": report,
        "shap_rank_path": str(cfg.artifacts_dir / "shap_global_ranking.csv"),
        "shap_plot_path": str(cfg.artifacts_dir / "shap_beeswarm.png"),
        "ate_path": str(cfg.artifacts_dir / "causal_forest_ate.csv"),
        "report_path": str(report_out),
        "html_report_path": str(html_out),
        "top_cases_path": str(top_cases_out),
    }


def score_new_trades(
    train_trades: pd.DataFrame,
    new_trades: pd.DataFrame,
    cfg: PipelineConfig,
    out_path: Path,
    top_k_drivers: int = 5,
) -> Path:
    """
    Train on labeled trades (train_trades with label_uit), then score new_trades (no label needed).
    Writes a parquet with risk scores and top contributing drivers.
    """
    _ensure_dir(cfg.artifacts_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_df = train_trades.dropna(subset=FEATURE_COLS + ["label_uit"]).copy()
    y = train_df["label_uit"].astype(int).values
    X_train = _make_X(train_df)

    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=cfg.seed,
        n_jobs=0,
        eval_metric="logloss",
    )
    model.fit(X_train, y)

    score_df = new_trades.dropna(subset=FEATURE_COLS).copy()
    X_new = _make_X(score_df)
    # align columns (in case one-hot differs)
    X_new = X_new.reindex(columns=X_train.columns, fill_value=0.0)

    proba = model.predict_proba(X_new)[:, 1]
    score_df["uit_risk"] = proba

    # contributions per row (SHAP-like)
    import xgboost as xgb

    dnew = xgb.DMatrix(X_new, feature_names=list(X_new.columns))
    contrib = model.get_booster().predict(dnew, pred_contribs=True)
    contrib = contrib[:, :-1]  # drop bias
    contrib_abs = np.abs(contrib)
    top_idx = np.argsort(-contrib_abs, axis=1)[:, :top_k_drivers]
    feat_names = np.array(X_new.columns)

    score_df["top_drivers"] = [",".join(feat_names[idx].tolist()) for idx in top_idx]

    # also store top driver contributions for interpretability (same order as top_drivers)
    top_vals = np.take_along_axis(contrib, top_idx, axis=1)
    score_df["top_driver_contribs"] = [
        ",".join([f"{v:.5f}" for v in row]) for row in top_vals
    ]

    score_df.to_parquet(out_path, index=False)
    return out_path

