## Insider trading (UIT) mock pipeline (uv)

This repo scaffolds a **realistic mock** of the datasets referenced in
“Detecting and Explaining Unlawful Insider Trading…” (XGBoost + SHAP + Causal Forest),
then runs an end-to-end pipeline locally.

### Setup (uv)

Install `uv`, then from the repo root:

```bash
uv python install 3.12
uv sync
```

### Generate realistic mock datasets

```bash
uv run uit mock --out-dir mock_data
```

This writes:
- `mock_data/form4_trades.parquet` (Form 4-like trades + injected UIT label)
- `mock_data/new_trades.parquet` (Unlabeled “new” trades for inference demos)
- `mock_data/crsp_daily.parquet` (CRSP-like daily panel)
- `mock_data/compustat_quarterly.parquet` (Compustat-like quarterly fundamentals)
- `mock_data/link.parquet` (cik/gvkey/permno mapping)
- `mock_data/enforcement_labels.parquet` (enforcement-like subset of positives)

### Run the model + explanations

```bash
uv run uit run --in-dir mock_data --artifacts-dir artifacts
```

To also generate a **plain-English narrative** of the top SHAP drivers in the HTML report (uses `OPENAI_API_KEY` from `.env`):

```bash
uv run uit run --in-dir mock_data --artifacts-dir artifacts --llm-explain
```

Optional model override:

```bash
uv run uit run --in-dir mock_data --artifacts-dir artifacts --llm-explain --openai-model gpt-4.1-mini
```

Outputs:
- `artifacts/shap_global_ranking.csv`
- `artifacts/shap_beeswarm.png`
- `artifacts/causal_forest_ate.csv`
- `artifacts/uit_report.md` (human-readable summary)
- `artifacts/uit_report.html` (human-readable HTML report)
- `artifacts/top_flagged_trades.csv` (examples)

**Preview the HTML report in the browser** (via [htmlpreview.github.io](https://htmlpreview.github.io/)):

[Open `uit_report.html` preview](https://htmlpreview.github.io/?https://github.com/SanjuMenon/Financial_Usecases/blob/insider_trading/artifacts/uit_report.html)

### Score new data (inference)

If you have a new trades file (CSV or Parquet) that includes the required feature columns, you can score it:

```bash
uv run uit score --new path/to/new_trades.parquet --out artifacts/scored_trades.parquet
```

To use the repo’s generated unlabeled mock:

```bash
uv run uit score --new mock_data/new_trades.parquet --out artifacts/scored_new_trades.parquet
```

The output includes:
- `uit_risk`: predicted probability
- `top_drivers`: top contributing features (comma-separated)