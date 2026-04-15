from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MockConfig:
    seed: int = 7
    n_firms: int = 250
    n_insiders: int = 1200
    start: str = "2020-01-01"
    end: str = "2023-12-31"
    n_trades: int = 30_000


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_mock_datasets(cfg: MockConfig) -> dict[str, pd.DataFrame]:
    """
    Generate "realistic modeling" mocks:
    - CRSP-like daily panel with returns, prices, volume, bid/ask + true beta/vol
    - Compustat-like quarterly panel with fundamentals and derived ratios
    - Link table (cik <-> gvkey <-> permno)
    - Form 4-like trades with insider roles
    - Labels with injected ground-truth UIT mechanism
    """
    rng = np.random.default_rng(cfg.seed)

    # --- entities / linking
    permnos = np.arange(10_000, 10_000 + cfg.n_firms, dtype=int)
    gvkeys = np.array([f"{i:06d}" for i in range(1, cfg.n_firms + 1)])
    ciks = np.array([f"{1000000 + i:010d}" for i in range(cfg.n_firms)])
    tickers = np.array([f"F{i:04d}" for i in range(cfg.n_firms)])

    link = pd.DataFrame(
        {
            "permno": permnos,
            "gvkey": gvkeys,
            "cik": ciks,
            "ticker": tickers,
        }
    )

    # --- time axes
    dates = pd.bdate_range(cfg.start, cfg.end, freq="B")
    quarters = pd.period_range(pd.Period(cfg.start, "Q"), pd.Period(cfg.end, "Q"), freq="Q")

    # --- market model for daily returns
    mkt = pd.Series(rng.normal(loc=0.0002, scale=0.01, size=len(dates)), index=dates, name="mkt_ret")
    smb = pd.Series(rng.normal(loc=0.0, scale=0.008, size=len(dates)), index=dates, name="smb")
    hml = pd.Series(rng.normal(loc=0.0, scale=0.008, size=len(dates)), index=dates, name="hml")

    # firm-specific parameters (stable, but heterogeneous)
    beta_mkt = rng.normal(1.0, 0.35, size=cfg.n_firms).clip(0.05, 2.5)
    beta_smb = rng.normal(0.2, 0.25, size=cfg.n_firms).clip(-1.0, 1.0)
    beta_hml = rng.normal(0.1, 0.25, size=cfg.n_firms).clip(-1.0, 1.0)
    idio_sigma = rng.lognormal(mean=np.log(0.012), sigma=0.35, size=cfg.n_firms).clip(0.004, 0.05)

    # generate daily panel
    daily_frames: list[pd.DataFrame] = []
    for i, permno in enumerate(permnos):
        eps = rng.normal(0.0, idio_sigma[i], size=len(dates))
        ret = 0.0001 + beta_mkt[i] * mkt.values + beta_smb[i] * smb.values + beta_hml[i] * hml.values + eps

        # price path (start around 20-200)
        p0 = float(rng.uniform(20, 200))
        prc = p0 * np.exp(np.cumsum(ret))

        # volume correlated with volatility and price level
        vol = rng.lognormal(mean=12.0, sigma=0.6, size=len(dates))
        vol = vol * (1.0 + 8.0 * np.abs(ret))

        # crude spreads (higher when vol higher, lower in liquid names)
        base_spread = rng.uniform(0.01, 0.15)  # dollars
        quoted_spread = base_spread * (1.0 + 10.0 * np.abs(ret))
        bid = prc - quoted_spread / 2.0
        ask = prc + quoted_spread / 2.0

        df = pd.DataFrame(
            {
                "date": dates,
                "permno": permno,
                "ret": ret,
                "retx": ret,  # mock "without dividends"
                "prc": prc,
                "vol": vol,
                "bid": bid,
                "ask": ask,
                "quoted_spread": quoted_spread,
                "mkt_ret": mkt.values,
                "smb": smb.values,
                "hml": hml.values,
            }
        )
        daily_frames.append(df)

    crsp_daily = pd.concat(daily_frames, ignore_index=True)

    # --- quarterly fundamentals (Compustat-like)
    firm_size = rng.lognormal(mean=np.log(5e9), sigma=1.0, size=cfg.n_firms)  # proxy size
    leverage = rng.beta(2.0, 4.0, size=cfg.n_firms)
    profitability = rng.normal(0.08, 0.05, size=cfg.n_firms).clip(-0.15, 0.35)

    q_frames: list[pd.DataFrame] = []
    for i, gvkey in enumerate(gvkeys):
        for q in quarters:
            # slow-moving fundamentals
            size_q = firm_size[i] * np.exp(rng.normal(0.0, 0.03))
            equity = size_q * (1.0 - leverage[i]) * rng.uniform(0.35, 0.65)
            debt = size_q * leverage[i] * rng.uniform(0.35, 0.75)

            assets = equity + debt + size_q * rng.uniform(0.05, 0.25)
            current_ratio = rng.lognormal(mean=np.log(1.7), sigma=0.35)
            quick_ratio = max(0.2, current_ratio - rng.uniform(0.2, 0.8))

            roe = profitability[i] + rng.normal(0.0, 0.02)
            roa = (roe * equity / assets) + rng.normal(0.0, 0.01)

            # link to a "valuation": simulate price-to-book / p/e-ish
            pb = rng.lognormal(mean=np.log(2.0), sigma=0.5)
            if profitability[i] < 0:
                pb *= rng.uniform(0.4, 0.9)

            debt_equity = debt / max(equity, 1.0)

            q_frames.append(
                {
                    "gvkey": gvkey,
                    "quarter": str(q),
                    "datadate": q.end_time,
                    "assets": assets,
                    "equity": equity,
                    "debt": debt,
                    "debt_equity": debt_equity,
                    "current_ratio": current_ratio,
                    "quick_ratio": quick_ratio,
                    "roe": roe,
                    "roa": roa,
                    "price_to_book": pb,
                }
            )

    comp_q = pd.DataFrame(q_frames)

    # --- insiders and Form 4-like trades
    insider_ids = np.arange(1, cfg.n_insiders + 1, dtype=int)
    # assign each insider to a firm (many-to-one)
    insider_firm_idx = rng.integers(0, cfg.n_firms, size=cfg.n_insiders)

    # roles: directors/officers/10% owners (allow overlap but keep simple)
    is_director = rng.binomial(1, 0.18, size=cfg.n_insiders)
    is_officer = rng.binomial(1, 0.35, size=cfg.n_insiders)
    ten_percent_owner = rng.binomial(1, 0.06, size=cfg.n_insiders)

    # generate trades sampling (firm, insider, date)
    trade_insider = rng.choice(insider_ids, size=cfg.n_trades, replace=True)
    trade_firm_idx = insider_firm_idx[trade_insider - 1]
    trade_permno = permnos[trade_firm_idx]
    trade_cik = ciks[trade_firm_idx]

    trade_dates = rng.choice(dates.values, size=cfg.n_trades, replace=True)
    trade_dates = pd.to_datetime(trade_dates)

    # join price at trade date
    crsp_key = crsp_daily[["permno", "date", "prc", "ret", "quoted_spread"]].copy()
    crsp_key["date"] = pd.to_datetime(crsp_key["date"])
    trade_df = pd.DataFrame(
        {
            "trade_id": np.arange(1, cfg.n_trades + 1, dtype=int),
            "cik": trade_cik,
            "permno": trade_permno,
            "personid": trade_insider,
            "transaction_date": trade_dates,
        }
    )
    trade_df = trade_df.merge(crsp_key, left_on=["permno", "transaction_date"], right_on=["permno", "date"], how="left")
    trade_df = trade_df.drop(columns=["date"])

    # acquisition/disposition
    acq_disp = rng.choice(["A", "D"], size=cfg.n_trades, p=[0.7, 0.3])
    shares = rng.lognormal(mean=7.0, sigma=0.9, size=cfg.n_trades).astype(int).clip(1, 2_000_000)

    # attach roles
    trade_df["is_director"] = is_director[trade_insider - 1]
    trade_df["is_officer"] = is_officer[trade_insider - 1]
    trade_df["ten_percent_owner"] = ten_percent_owner[trade_insider - 1]
    trade_df["acq_disp"] = acq_disp
    trade_df["shares"] = shares
    trade_df["price"] = trade_df["prc"]

    # attach most recent quarter fundamentals (as-of merge)
    # map permno -> gvkey via link, then merge quarterly by datadate <= trade date
    permno_to_gvkey = link.set_index("permno")["gvkey"]
    trade_df["gvkey"] = trade_df["permno"].map(permno_to_gvkey)

    # merge_asof requires both frames be sorted by [by, on] and have consistent dtypes
    comp_q_sorted = comp_q.copy()
    comp_q_sorted["gvkey"] = comp_q_sorted["gvkey"].astype("string")
    comp_q_sorted["datadate"] = pd.to_datetime(comp_q_sorted["datadate"])
    comp_q_sorted = comp_q_sorted.sort_values(["gvkey", "datadate"], kind="mergesort").reset_index(drop=True)

    trade_sorted = trade_df.copy()
    trade_sorted["gvkey"] = trade_sorted["gvkey"].astype("string")
    trade_sorted["transaction_date"] = pd.to_datetime(trade_sorted["transaction_date"])
    trade_sorted = trade_sorted.sort_values(["gvkey", "transaction_date"], kind="mergesort").reset_index(drop=True)

    # Pandas' merge_asof can be finicky about sort order when using `by`.
    # Do an as-of merge per firm key and concatenate (n_firms is modest).
    merged_parts: list[pd.DataFrame] = []
    for gvkey, left_g in trade_sorted.groupby("gvkey", sort=False):
        right_g = comp_q_sorted[comp_q_sorted["gvkey"] == gvkey]
        if right_g.empty:
            merged_parts.append(left_g.assign(**{c: np.nan for c in comp_q.columns if c not in ("gvkey",)}))
            continue
        left_g = left_g.sort_values("transaction_date", kind="mergesort")
        right_g = right_g.sort_values("datadate", kind="mergesort")
        mg = pd.merge_asof(
            left_g,
            right_g,
            left_on="transaction_date",
            right_on="datadate",
            direction="backward",
            allow_exact_matches=True,
        )
        merged_parts.append(mg)

    merged = pd.concat(merged_parts, ignore_index=True)
    # merge_asof keeps both sides' columns; if names overlap pandas may suffix.
    if "gvkey" not in merged.columns:
        if "gvkey_x" in merged.columns:
            merged["gvkey"] = merged["gvkey_x"]
        elif "gvkey_y" in merged.columns:
            merged["gvkey"] = merged["gvkey_y"]

    # --- features used in the paper's "post-correlation" set
    # market beta proxy from true beta_mkt; join by permno
    beta_map = pd.Series(beta_mkt, index=permnos, name="market_beta_true")
    merged["market_beta"] = merged["permno"].map(beta_map)

    # spread-of-return proxy: use quoted_spread normalized by price (roughly)
    merged["sprd_rtn"] = (merged["quoted_spread"] / merged["price"]).clip(0, 0.5)

    # mock price operating earnings basic (P/E-ish)
    merged["prc_op_earnings_basic"] = rng.lognormal(mean=np.log(18.0), sigma=0.55, size=len(merged)).clip(2, 120)

    # hml beta proxy from true beta_hml
    hml_map = pd.Series(beta_hml, index=permnos, name="hml_beta_true")
    merged["hml_beta"] = merged["permno"].map(hml_map)

    # --- ground-truth UIT probability (injects signals consistent with paper narrative)
    # higher risk for: director, low PB, low beta, low trailing returns, low spreads occasionally
    pb = merged["price_to_book"].astype(float)
    beta = merged["market_beta"].astype(float)
    ret = merged["ret"].astype(float)
    sprd = merged["sprd_rtn"].astype(float)

    # Some rows can be missing upstream joins (e.g., if a trade falls on a non-trading day after sampling).
    # Fill with conservative defaults so probabilities remain well-defined.
    pb = pb.fillna(pb.median()).clip(lower=0.05)
    beta = beta.fillna(beta.median() if np.isfinite(beta.median()) else 1.0).clip(lower=0.05)
    ret = ret.fillna(0.0)
    sprd = sprd.fillna(sprd.median()).clip(lower=0.0)

    score = (
        -3.2
        + 1.2 * merged["is_director"].astype(float)
        + 0.35 * (merged["acq_disp"] == "A").astype(float)
        + 0.75 * (1.0 / (1.0 + pb))  # lower PB -> higher score
        + 0.65 * (1.0 / (0.35 + beta))  # lower beta -> higher score
        + 0.6 * (-ret)  # low returns increase risk
        + 0.15 * (-sprd)  # smaller spread mildly increases (harder to detect)
        + rng.normal(0.0, 0.35, size=len(merged))
    )
    score = np.asarray(score, dtype=float)
    score = np.nan_to_num(score, nan=0.0, posinf=10.0, neginf=-10.0)

    p_uit = _sigmoid(score)
    p_uit = np.asarray(p_uit, dtype=float)
    p_uit = np.nan_to_num(p_uit, nan=0.5, posinf=1.0, neginf=0.0)
    p_uit = np.clip(p_uit, 1e-6, 1 - 1e-6)

    merged["label_uit"] = rng.binomial(1, p_uit, size=len(p_uit)).astype(int)

    # enforcement-like label table (subset of positives)
    positives = merged[merged["label_uit"] == 1].sample(frac=0.35, random_state=cfg.seed)
    labels = positives[["personid", "cik", "transaction_date", "label_uit"]].copy()
    labels = labels.rename(columns={"transaction_date": "event_date"})
    labels["case_id"] = rng.integers(10_000, 99_999, size=len(labels))

    # finalize
    form4 = merged[
        [
            "trade_id",
            "cik",
            "permno",
            "gvkey",
            "personid",
            "transaction_date",
            "acq_disp",
            "shares",
            "price",
            "is_director",
            "is_officer",
            "ten_percent_owner",
            # selected predictors
            "market_beta",
            "price_to_book",
            "sprd_rtn",
            "prc_op_earnings_basic",
            "hml_beta",
            "ret",
            # label
            "label_uit",
        ]
    ].copy()

    # "New" unlabeled trades for inference demos (same schema minus label)
    new_trades = (
        form4.sample(n=min(5000, len(form4)), random_state=cfg.seed + 1)
        .drop(columns=["label_uit"])
        .reset_index(drop=True)
    )

    return {
        "link": link,
        "crsp_daily": crsp_daily,
        "compustat_quarterly": comp_q,
        "form4_trades": form4,
        "new_trades": new_trades,
        "enforcement_labels": labels,
    }


def write_mock_datasets(datasets: dict[str, pd.DataFrame], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in datasets.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)

