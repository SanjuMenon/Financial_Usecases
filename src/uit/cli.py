from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .mock_data import MockConfig, generate_mock_datasets, write_mock_datasets
from .pipeline import PipelineConfig, run_xgb_shap_causal, score_new_trades


def _cmd_mock(args: argparse.Namespace) -> int:
    cfg = MockConfig(
        seed=args.seed,
        n_firms=args.n_firms,
        n_insiders=args.n_insiders,
        start=args.start,
        end=args.end,
        n_trades=args.n_trades,
    )
    out_dir = Path(args.out_dir)
    datasets = generate_mock_datasets(cfg)
    write_mock_datasets(datasets, out_dir)
    print(f"Wrote mock datasets to: {out_dir.resolve()}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    in_dir = Path(args.in_dir)
    form4_path = in_dir / "form4_trades.parquet"
    if not form4_path.exists():
        raise FileNotFoundError(f"Missing {form4_path}. Run `uit mock ...` first.")

    form4 = pd.read_parquet(form4_path)
    cfg = PipelineConfig(seed=args.seed, artifacts_dir=Path(args.artifacts_dir))
    results = run_xgb_shap_causal(
        form4,
        cfg,
        llm_explain=args.llm_explain,
        openai_model=args.openai_model,
    )
    print(json.dumps(results, indent=2))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    train_path = Path(args.train)
    new_path = Path(args.new)
    out_path = Path(args.out)

    train_df = pd.read_parquet(train_path) if train_path.suffix == ".parquet" else pd.read_csv(train_path)
    new_df = pd.read_parquet(new_path) if new_path.suffix == ".parquet" else pd.read_csv(new_path)

    cfg = PipelineConfig(seed=args.seed, artifacts_dir=Path(args.artifacts_dir))
    written = score_new_trades(train_df, new_df, cfg=cfg, out_path=out_path, top_k_drivers=args.top_k)
    print(f"Wrote scored trades to: {written.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uit", description="UIT mock data + modeling pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser("mock", help="Generate realistic mock datasets (parquet)")
    pm.add_argument("--out-dir", default="mock_data", help="Output directory")
    pm.add_argument("--seed", type=int, default=7)
    pm.add_argument("--n-firms", type=int, default=250)
    pm.add_argument("--n-insiders", type=int, default=1200)
    pm.add_argument("--n-trades", type=int, default=30000)
    pm.add_argument("--start", default="2020-01-01")
    pm.add_argument("--end", default="2023-12-31")
    pm.set_defaults(func=_cmd_mock)

    pr = sub.add_parser("run", help="Train XGBoost + SHAP + causal forest on mock Form 4 trades")
    pr.add_argument("--in-dir", default="mock_data", help="Directory containing parquet files")
    pr.add_argument("--artifacts-dir", default="artifacts", help="Where to write reports/plots")
    pr.add_argument("--seed", type=int, default=7)
    pr.add_argument("--llm-explain", action="store_true", help="Add natural-language SHAP summary to HTML report (uses OPENAI_API_KEY)")
    pr.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model for the narrative summary")
    pr.set_defaults(func=_cmd_run)

    ps = sub.add_parser("score", help="Train on labeled data, score a new file (csv/parquet)")
    ps.add_argument("--train", default="mock_data/form4_trades.parquet", help="Labeled training trades (needs label_uit)")
    ps.add_argument("--new", required=True, help="New trades to score (no label required)")
    ps.add_argument("--out", default="artifacts/scored_trades.parquet", help="Output parquet path")
    ps.add_argument("--artifacts-dir", default="artifacts", help="Where to write artifacts")
    ps.add_argument("--top-k", type=int, default=5, help="Top K driver features per row")
    ps.add_argument("--seed", type=int, default=7)
    ps.set_defaults(func=_cmd_score)

    return p


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))

