# ============================================================
# BUILD AAVE MODEL READY ANALYSIS V2 - RAW TO FINAL PIPELINE
# ============================================================
# Mục tiêu:
# - Đi trực tiếp từ raw data:
#     aave_supply.csv / aave_borrow.csv / aave_repay.csv
#     aave_liquidationcall.csv
#     threshold_final.csv
#     token_prices_daily.csv
#     Fear & Greed Index.csv
# - Tái dựng trạng thái user-day của từng ví
# - Tính Health Factor, Liquidation Threshold, Distance to Liquidation
# - Gắn nhãn liquidated_next_3d từ liquidation raw
# - Xuất file cuối cùng:
#     data/processed/aave_model_ready_analysis_v2.csv
#
# Lưu ý quan trọng:
# - liquidator_address, liquidation_block, has_liquidation_metadata là biến metadata
#   phục vụ kiểm tra/gắn nhãn, KHÔNG dùng làm input feature khi train model.
# - Nếu có raw withdraw event, đặt file vào data/raw/aave_withdraw.csv.
#   Nếu không có, code vẫn chạy nhưng collateral có thể bị ước lượng cao hơn thực tế.
# ============================================================

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REPORT_DIR = PROJECT_DIR / "reports"

# Nếu chưa tạo thư mục data/raw, code sẽ tìm file ngay trong PROJECT_DIR.
if not RAW_DIR.exists():
    RAW_DIR = PROJECT_DIR

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "aave_model_ready_analysis_v2.csv"
REPORT_FILE = REPORT_DIR / "aave_model_ready_analysis_v2_report.txt"
RESERVE_POSITION_FILE = PROCESSED_DIR / "aave_user_day_reserve_positions_for_hf_v2.csv"

# Khoảng thời gian nghiên cứu. Có thể chỉnh lại nếu cần.
STUDY_START = pd.Timestamp("2021-01-01")
STUDY_END = pd.Timestamp("2026-05-29")

HF_CAP = 100.0
MAX_PRICE_STALENESS_DAYS = 30
MAX_THRESHOLD_STALENESS_DAYS = 180
MIN_POSITION_USD = 1e-6
LABEL_HORIZON_DAYS = 3

# Tên file có thể dùng. Code sẽ tự lấy file đầu tiên tồn tại.
SUPPLY_FILES = [
    "aave_supply.csv",
    "3. aave_v3_supply.csv",
]
BORROW_FILES = [
    "aave_borrow.csv",
    "4. aave_v3_borrow.csv",
]
REPAY_FILES = [
    "aave_repay.csv",
    "1. aave_v3_repay_merged.csv",
]
WITHDRAW_FILES = [
    "aave_withdraw.csv",
    "aave_v3_withdraw.csv",
    "5. aave_v3_withdraw.csv",
]
LIQUIDATION_FILES = [
    "aave_liquidationcall.csv",
    "aave_liquidation_call.csv",
]
PRICE_FILES = [
    "token_prices_daily.csv",
    "2. token_prices_daily.csv",
    "2. token_prices_daily_merged.csv",
]
THRESHOLD_FILES = [
    "threshold_final.csv",
    "aave_liquidation_threshold_daily.csv",
]
FEAR_GREED_FILES = [
    "Fear & Greed Index.csv",
    "fear_greed_index.csv",
    "fear_and_greed_index.csv",
]

FINAL_COLUMNS = [
    "snapshot",
    "observation_date",
    "protocol",
    "user_address",

    "health_factor",
    "liquidation_threshold",
    "distance_to_liquidation_pct",

    "collateral_asset_primary",
    "debt_asset_primary",

    "collateral_usd",
    "debt_usd",
    "log_collateral_usd",
    "log_debt_usd",

    "n_collateral_types",

    "tx_count_7d",
    "tx_count_30d",
    "inactive_days",
    "inactivity_flag_30d",
    "position_age_days",

    "fear_greed_index",
    "extreme_fear_flag",
    "fear_flag",
    "neutral_flag",
    "greed_flag",
    "extreme_greed_flag",

    "liquidator_address",
    "liquidation_block",
    "has_liquidation_metadata",

    "liquidated_next_3d",
]

# Token decimals dự phòng nếu threshold file thiếu decimals.
TOKEN_DECIMALS_FALLBACK = {
    "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2": 18,  # WETH
    "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599": 8,   # WBTC
    "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0": 18,  # wstETH
    "0xae78736cd615f374d3085123a210448e74fc6393": 18,  # rETH
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": 6,   # USDC
    "0xdac17f958d2ee523a220612309bda2f40e6db16": 6,   # USDT
    "0x6b175474e89094c44da98b954eedeac495271d0f": 18,  # DAI
    "0x514910771af9ca656af840dff83e8264ecf986ca": 18,  # LINK
    "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9": 18,  # AAVE
    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984": 18,  # UNI
    "0xd533a949740bb3306d119cc777fa900ba034cd52": 18,  # CRV
    "0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": 18,  # MKR
}

# ============================================================
# 2. HÀM PHỤ TRỢ
# ============================================================

def section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def clean_str_series(s: pd.Series, none_value: str = "none") -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "": none_value,
            "nan": none_value,
            "none": none_value,
            "null": none_value,
            "na": none_value,
            "nat": none_value,
        })
    )


def clean_address_series(s: pd.Series) -> pd.Series:
    return clean_str_series(s, none_value="none")


def clean_protocol_series(s: pd.Series) -> pd.Series:
    x = clean_str_series(s, none_value="none")
    x = (
        x.str.replace("-", "_", regex=False)
         .str.replace(" ", "_", regex=False)
         .str.replace("__", "_", regex=False)
    )
    x = x.str.replace("aave_", "", regex=False)
    x = x.str.replace("version_", "", regex=False)
    x = x.str.replace("protocol_", "", regex=False)
    x = x.replace({
        "2": "v2",
        "3": "v3",
        "v2": "v2",
        "v3": "v3",
        "aavev2": "v2",
        "aavev3": "v3",
    })
    return x.map(lambda v: f"aave_{v}" if v in ["v2", "v3"] else v)


def parse_date_series(s: pd.Series) -> pd.Series:
    # Cắt 10 ký tự đầu để xử lý các chuỗi dạng "2023-01-27 00:00:00.000 UTC".
    return pd.to_datetime(s.astype(str).str.slice(0, 10), errors="coerce")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def file_path_candidates(names):
    return [RAW_DIR / n for n in names] + [PROJECT_DIR / n for n in names]


def find_existing_files(names):
    seen = set()
    files = []
    for p in file_path_candidates(names):
        if p.exists() and p.resolve() not in seen:
            files.append(p)
            seen.add(p.resolve())
    return files


def find_first_existing(names):
    files = find_existing_files(names)
    return files[0] if files else None


def pick_col(columns, candidates, required=True, label=""):
    lower_map = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    if required:
        raise ValueError(
            f"Không tìm thấy cột {label}. Candidates={candidates}\n"
            f"Các cột hiện có: {list(columns)}"
        )
    return None


def snapshot_from_date(date_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date_series, errors="coerce")
    out = pd.Series("normal", index=date_series.index)
    out.loc[(dt >= pd.Timestamp("2023-03-08")) & (dt <= pd.Timestamp("2023-03-15"))] = "crisis_1"
    out.loc[(dt >= pd.Timestamp("2024-08-04")) & (dt <= pd.Timestamp("2024-08-10"))] = "crisis_2"
    return out


def recompute_fear_greed_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    fg = pd.to_numeric(df["fear_greed_index"], errors="coerce")
    df["extreme_fear_flag"] = ((fg >= 0) & (fg <= 25)).astype(int)
    df["fear_flag"] = ((fg >= 26) & (fg <= 45)).astype(int)
    df["neutral_flag"] = ((fg >= 46) & (fg <= 54)).astype(int)
    df["greed_flag"] = ((fg >= 55) & (fg <= 74)).astype(int)
    df["extreme_greed_flag"] = ((fg >= 75) & (fg <= 100)).astype(int)
    return df


def standardize_threshold_value(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    med = x.dropna().median()
    if pd.isna(med):
        return x
    # Nếu dữ liệu là bps như 8000, đổi sang 0.8.
    if med > 100:
        return x / 10000.0
    # Nếu dữ liệu là phần trăm như 80, đổi sang 0.8.
    if med > 1:
        return x / 100.0
    return x


def check_eth_address_or_none(s: pd.Series) -> int:
    x = s.astype(str).str.lower()
    valid_address = x.str.match(r"^0x[a-f0-9]{40}$", na=False)
    valid_none = x.eq("none")
    return int((~(valid_address | valid_none)).sum())

# ============================================================
# 3. ĐỌC RAW EVENTS
# ============================================================

def read_event_file(path: Path, event_type: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0, low_memory=False)
    header = normalize_columns(header)
    cols = list(header.columns)

    amount_col = pick_col(
        cols,
        ["amount_raw", "raw_amount", "amount", "value"],
        required=True,
        label=f"amount trong {path.name}",
    )
    date_col = pick_col(
        cols,
        ["date", "event_date", "block_date", "evt_block_time", "block_time", "timestamp", "time"],
        required=True,
        label=f"date trong {path.name}",
    )
    tx_col = pick_col(
        cols,
        ["evt_tx_hash", "tx_hash", "transaction_hash", "hash"],
        required=False,
        label=f"tx hash trong {path.name}",
    )
    protocol_col = pick_col(
        cols,
        ["protocol", "protocol_version", "market", "aave_version", "version", "pool_version"],
        required=True,
        label=f"protocol trong {path.name}",
    )
    reserve_col = pick_col(
        cols,
        ["reserve", "asset", "token", "token_address", "asset_address", "underlying_asset"],
        required=True,
        label=f"reserve trong {path.name}",
    )
    user_col = pick_col(
        cols,
        ["user_address", "user", "onbehalfof", "on_behalf_of", "borrower", "supplier", "depositor"],
        required=True,
        label=f"user trong {path.name}",
    )

    usecols = [amount_col, date_col, protocol_col, reserve_col, user_col]
    if tx_col is not None:
        usecols.append(tx_col)
    usecols = list(dict.fromkeys(usecols))

    temp = pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)
    temp = normalize_columns(temp)

    out = pd.DataFrame()
    out["protocol"] = clean_protocol_series(temp[protocol_col])
    out["user_address"] = clean_address_series(temp[user_col])
    out["reserve"] = clean_address_series(temp[reserve_col])
    out["event_date_dt"] = parse_date_series(temp[date_col])
    out["event_date"] = out["event_date_dt"].dt.strftime("%Y-%m-%d")
    out["amount_raw"] = safe_numeric(temp[amount_col])
    out["event_type"] = event_type
    out["source_file"] = path.name
    out["evt_tx_hash"] = clean_str_series(temp[tx_col], none_value="none") if tx_col else "none"

    out = out.dropna(subset=["protocol", "user_address", "reserve", "event_date_dt", "amount_raw"]).copy()
    out = out[(out["amount_raw"] >= 0) & (out["event_date_dt"] >= STUDY_START) & (out["event_date_dt"] <= STUDY_END)].copy()
    print(f"{event_type:12s} | {path.name:35s} | {len(out):,} rows")
    return out


def read_event_files(names, event_type: str, required=True) -> pd.DataFrame:
    files = find_existing_files(names)
    if not files:
        if required:
            raise FileNotFoundError(f"Không tìm thấy file cho {event_type}: {names}")
        print(f"[WARN] Không tìm thấy file {event_type}. Bỏ qua nhóm sự kiện này.")
        return pd.DataFrame(columns=[
            "protocol", "user_address", "reserve", "event_date_dt", "event_date",
            "amount_raw", "event_type", "source_file", "evt_tx_hash"
        ])

    parts = [read_event_file(p, event_type) for p in files]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    before = len(df)
    dedup_cols = ["protocol", "user_address", "reserve", "event_date", "amount_raw", "event_type", "evt_tx_hash"]
    df = df.drop_duplicates(dedup_cols, keep="first").reset_index(drop=True)
    print(f"Tổng {event_type}: {len(df):,} rows | removed dup: {before - len(df):,}")
    return df


def read_liquidation_files(names) -> pd.DataFrame:
    files = find_existing_files(names)
    if not files:
        raise FileNotFoundError(f"Không tìm thấy liquidation file: {names}")

    parts = []
    for path in files:
        header = pd.read_csv(path, nrows=0, low_memory=False)
        header = normalize_columns(header)
        cols = list(header.columns)

        protocol_col = pick_col(cols, ["protocol", "protocol_version", "market", "version"], label=f"protocol trong {path.name}")
        borrower_col = pick_col(cols, ["borrower_address", "borrower", "user_address", "user"], label=f"borrower trong {path.name}")
        collateral_col = pick_col(cols, ["collateral_asset", "collateral_reserve", "collateral", "collateral_token"], label=f"collateral asset trong {path.name}")
        debt_col = pick_col(cols, ["debt_asset", "debt_reserve", "reserve", "debt_token"], label=f"debt asset trong {path.name}")
        collateral_amt_col = pick_col(cols, ["collateral_seized_raw", "liquidated_collateral_amount", "collateral_amount", "amount_collateral_raw"], label=f"collateral amount trong {path.name}")
        debt_amt_col = pick_col(cols, ["debt_covered_raw", "debt_to_cover", "debt_amount", "amount_debt_raw"], label=f"debt amount trong {path.name}")
        date_col = pick_col(cols, ["date", "evt_block_time", "block_time", "timestamp", "time"], label=f"date trong {path.name}")
        block_col = pick_col(cols, ["evt_block_number", "block_number", "liquidation_block"], required=False, label=f"block trong {path.name}")
        tx_col = pick_col(cols, ["evt_tx_hash", "tx_hash", "transaction_hash", "hash"], required=False, label=f"tx hash trong {path.name}")
        liquidator_col = pick_col(cols, ["liquidator_address", "liquidator", "keeper"], required=False, label=f"liquidator trong {path.name}")

        usecols = [protocol_col, borrower_col, collateral_col, debt_col, collateral_amt_col, debt_amt_col, date_col]
        for c in [block_col, tx_col, liquidator_col]:
            if c is not None:
                usecols.append(c)
        usecols = list(dict.fromkeys(usecols))

        temp = pd.read_csv(path, usecols=usecols, dtype=str, low_memory=False)
        temp = normalize_columns(temp)

        out = pd.DataFrame()
        out["protocol"] = clean_protocol_series(temp[protocol_col])
        out["user_address"] = clean_address_series(temp[borrower_col])
        out["collateral_asset"] = clean_address_series(temp[collateral_col])
        out["debt_asset"] = clean_address_series(temp[debt_col])
        out["collateral_seized_raw"] = safe_numeric(temp[collateral_amt_col])
        out["debt_covered_raw"] = safe_numeric(temp[debt_amt_col])
        out["liquidation_date_dt"] = parse_date_series(temp[date_col])
        out["liquidation_date"] = out["liquidation_date_dt"].dt.strftime("%Y-%m-%d")
        out["liquidation_block"] = safe_numeric(temp[block_col]) if block_col else np.nan
        out["evt_tx_hash"] = clean_str_series(temp[tx_col], none_value="none") if tx_col else "none"
        out["liquidator_address"] = clean_address_series(temp[liquidator_col]) if liquidator_col else "none"
        out["source_file"] = path.name

        out = out.dropna(subset=[
            "protocol", "user_address", "collateral_asset", "debt_asset",
            "collateral_seized_raw", "debt_covered_raw", "liquidation_date_dt"
        ]).copy()
        out = out[(out["liquidation_date_dt"] >= STUDY_START) & (out["liquidation_date_dt"] <= STUDY_END)].copy()
        parts.append(out)
        print(f"liquidation  | {path.name:35s} | {len(out):,} rows")

    liq = pd.concat(parts, ignore_index=True)
    before = len(liq)
    liq = liq.drop_duplicates(["protocol", "user_address", "collateral_asset", "debt_asset", "liquidation_date", "evt_tx_hash"], keep="first")
    print(f"Tổng liquidation: {len(liq):,} rows | removed dup: {before - len(liq):,}")
    return liq.reset_index(drop=True)

# ============================================================
# 4. ĐỌC THRESHOLD, PRICE, FEAR & GREED
# ============================================================

def read_threshold() -> pd.DataFrame:
    path = find_first_existing(THRESHOLD_FILES)
    if path is None:
        raise FileNotFoundError(f"Không tìm thấy threshold file: {THRESHOLD_FILES}")
    print(f"Threshold file dùng: {path}")

    raw = pd.read_csv(path, low_memory=False)
    raw = normalize_columns(raw)
    cols = list(raw.columns)

    protocol_col = pick_col(cols, ["protocol", "protocol_version", "market", "version"], label="protocol trong threshold")
    reserve_col = pick_col(cols, ["reserve", "asset", "token_address", "asset_address", "underlying_asset"], label="reserve trong threshold")
    date_col = pick_col(cols, ["date", "day", "timestamp", "time"], label="date trong threshold")
    status_col = pick_col(cols, ["status", "config_status"], required=False, label="status trong threshold")
    decimals_col = pick_col(cols, ["decimals_fixed", "decimals", "token_decimals"], required=False, label="decimals trong threshold")

    lt_col = pick_col(
        cols,
        ["liquidation_threshold_fixed", "liquidation_threshold", "liquidation_threshold_bps", "threshold"],
        label="liquidation threshold trong threshold",
    )

    threshold = pd.DataFrame()
    threshold["protocol"] = clean_protocol_series(raw[protocol_col])
    threshold["reserve"] = clean_address_series(raw[reserve_col])
    threshold["date_dt"] = parse_date_series(raw[date_col])
    threshold["date"] = threshold["date_dt"].dt.strftime("%Y-%m-%d")
    threshold["liquidation_threshold_fixed"] = standardize_threshold_value(raw[lt_col])
    threshold["decimals_fixed"] = safe_numeric(raw[decimals_col]) if decimals_col else np.nan
    threshold["status"] = clean_str_series(raw[status_col], none_value="ok") if status_col else "ok"

    threshold["decimals_fixed"] = threshold["decimals_fixed"].fillna(threshold["reserve"].map(TOKEN_DECIMALS_FALLBACK))
    threshold["decimals_fixed"] = threshold["decimals_fixed"].fillna(18)

    threshold = threshold.dropna(subset=["protocol", "reserve", "date_dt", "liquidation_threshold_fixed", "decimals_fixed"]).copy()
    threshold = threshold[(threshold["date_dt"] >= STUDY_START - pd.Timedelta(days=MAX_THRESHOLD_STALENESS_DAYS)) & (threshold["date_dt"] <= STUDY_END)].copy()
    threshold = threshold[threshold["liquidation_threshold_fixed"].between(0, 1)].copy()
    threshold = threshold[threshold["decimals_fixed"].between(0, 40)].copy()
    threshold = threshold.sort_values(["protocol", "reserve", "date_dt"]).drop_duplicates(["protocol", "reserve", "date"], keep="last")

    print(f"Threshold rows: {len(threshold):,} | tokens={threshold['reserve'].nunique():,}")
    return threshold.reset_index(drop=True)


def read_price() -> pd.DataFrame:
    path = find_first_existing(PRICE_FILES)
    if path is None:
        raise FileNotFoundError(f"Không tìm thấy price file: {PRICE_FILES}")
    print(f"Price file dùng: {path}")

    raw = pd.read_csv(path, low_memory=False)
    raw = normalize_columns(raw)
    cols = list(raw.columns)

    date_col = pick_col(cols, ["date", "price_date", "day", "timestamp", "time"], label="date trong price")
    token_col = pick_col(cols, ["token_address", "reserve", "asset", "asset_address", "address", "contract_address"], label="token trong price")
    price_col = pick_col(cols, ["price_usd", "usd_price", "price", "close", "close_usd"], label="price_usd trong price")

    price = pd.DataFrame()
    price["reserve"] = clean_address_series(raw[token_col])
    price["date_dt"] = parse_date_series(raw[date_col])
    price["date"] = price["date_dt"].dt.strftime("%Y-%m-%d")
    price["price_usd"] = safe_numeric(raw[price_col])

    price = price.dropna(subset=["reserve", "date_dt", "price_usd"]).copy()
    price = price[(price["price_usd"] > 0) & (price["date_dt"] >= STUDY_START - pd.Timedelta(days=MAX_PRICE_STALENESS_DAYS)) & (price["date_dt"] <= STUDY_END)].copy()
    price = price.sort_values(["reserve", "date_dt"]).drop_duplicates(["reserve", "date"], keep="last")

    print(f"Price rows: {len(price):,} | tokens={price['reserve'].nunique():,}")
    return price.reset_index(drop=True)


def read_fear_greed() -> pd.DataFrame:
    path = find_first_existing(FEAR_GREED_FILES)
    if path is None:
        raise FileNotFoundError(f"Không tìm thấy Fear & Greed file: {FEAR_GREED_FILES}")
    print(f"Fear & Greed file dùng: {path}")

    raw = pd.read_csv(path, low_memory=False)
    raw = normalize_columns(raw)
    cols = list(raw.columns)

    date_col = pick_col(cols, ["date", "timestamp", "time", "day"], label="date trong Fear & Greed")
    value_col = pick_col(cols, ["fear_greed_index", "value", "score", "index"], label="value trong Fear & Greed")

    fg = pd.DataFrame()
    fg["observation_date_dt"] = parse_date_series(raw[date_col])
    fg["observation_date"] = fg["observation_date_dt"].dt.strftime("%Y-%m-%d")
    fg["fear_greed_index"] = safe_numeric(raw[value_col])
    fg = fg.dropna(subset=["observation_date_dt", "fear_greed_index"]).copy()
    fg = fg[(fg["observation_date_dt"] >= STUDY_START) & (fg["observation_date_dt"] <= STUDY_END)].copy()
    fg["fear_greed_index"] = fg["fear_greed_index"].clip(0, 100).round().astype(int)
    fg = fg.sort_values("observation_date_dt").drop_duplicates("observation_date", keep="last")
    fg = recompute_fear_greed_flags(fg)

    print(f"Fear & Greed rows: {len(fg):,}")
    return fg.drop(columns=["observation_date_dt"]).reset_index(drop=True)

# ============================================================
# 5. TẠO POSITION DAILY TỪ RAW EVENTS
# ============================================================

def build_delta_positions(supply, borrow, repay, withdraw, liquidation):
    parts = []

    if len(supply):
        x = supply[["protocol", "user_address", "reserve", "event_date_dt", "evt_tx_hash", "amount_raw", "event_type"]].copy()
        x["collateral_delta_raw"] = x["amount_raw"]
        x["debt_delta_raw"] = 0.0
        parts.append(x)

    if len(withdraw):
        x = withdraw[["protocol", "user_address", "reserve", "event_date_dt", "evt_tx_hash", "amount_raw", "event_type"]].copy()
        x["collateral_delta_raw"] = -x["amount_raw"]
        x["debt_delta_raw"] = 0.0
        parts.append(x)

    if len(borrow):
        x = borrow[["protocol", "user_address", "reserve", "event_date_dt", "evt_tx_hash", "amount_raw", "event_type"]].copy()
        x["collateral_delta_raw"] = 0.0
        x["debt_delta_raw"] = x["amount_raw"]
        parts.append(x)

    if len(repay):
        x = repay[["protocol", "user_address", "reserve", "event_date_dt", "evt_tx_hash", "amount_raw", "event_type"]].copy()
        x["collateral_delta_raw"] = 0.0
        x["debt_delta_raw"] = -x["amount_raw"]
        parts.append(x)

    if len(liquidation):
        # Liquidation làm giảm debt của debt_asset.
        debt_liq = liquidation[["protocol", "user_address", "debt_asset", "liquidation_date_dt", "evt_tx_hash", "debt_covered_raw"]].copy()
        debt_liq = debt_liq.rename(columns={"debt_asset": "reserve", "liquidation_date_dt": "event_date_dt", "debt_covered_raw": "amount_raw"})
        debt_liq["event_type"] = "liquidation_debt_reduction"
        debt_liq["collateral_delta_raw"] = 0.0
        debt_liq["debt_delta_raw"] = -debt_liq["amount_raw"]
        parts.append(debt_liq)

        # Liquidation làm giảm collateral của collateral_asset.
        col_liq = liquidation[["protocol", "user_address", "collateral_asset", "liquidation_date_dt", "evt_tx_hash", "collateral_seized_raw"]].copy()
        col_liq = col_liq.rename(columns={"collateral_asset": "reserve", "liquidation_date_dt": "event_date_dt", "collateral_seized_raw": "amount_raw"})
        col_liq["event_type"] = "liquidation_collateral_seized"
        col_liq["collateral_delta_raw"] = -col_liq["amount_raw"]
        col_liq["debt_delta_raw"] = 0.0
        parts.append(col_liq)

    deltas = pd.concat(parts, ignore_index=True)
    deltas = deltas.dropna(subset=["protocol", "user_address", "reserve", "event_date_dt"]).copy()
    deltas["event_date"] = deltas["event_date_dt"].dt.strftime("%Y-%m-%d")

    # Gom về cấp ngày để trạng thái cuối ngày là trạng thái quan sát.
    daily = (
        deltas
        .groupby(["protocol", "user_address", "reserve", "event_date_dt", "event_date"], as_index=False)
        .agg(
            collateral_delta_raw=("collateral_delta_raw", "sum"),
            debt_delta_raw=("debt_delta_raw", "sum"),
            tx_count=("evt_tx_hash", "nunique"),
        )
        .sort_values(["protocol", "user_address", "reserve", "event_date_dt"])
        .reset_index(drop=True)
    )

    daily["collateral_raw_cum"] = daily.groupby(["protocol", "user_address", "reserve"])["collateral_delta_raw"].cumsum()
    daily["debt_raw_cum"] = daily.groupby(["protocol", "user_address", "reserve"])["debt_delta_raw"].cumsum()

    neg_collateral = int((daily["collateral_raw_cum"] < 0).sum())
    neg_debt = int((daily["debt_raw_cum"] < 0).sum())
    daily["collateral_raw_cum"] = daily["collateral_raw_cum"].clip(lower=0)
    daily["debt_raw_cum"] = daily["debt_raw_cum"].clip(lower=0)

    print(f"Delta daily rows: {len(daily):,}")
    print(f"Negative collateral before clip: {neg_collateral:,}")
    print(f"Negative debt before clip: {neg_debt:,}")
    return daily, deltas


def expand_active_debt_user_days(position_events: pd.DataFrame) -> pd.DataFrame:
    # Chỉ tạo target user-day cho những ngày ví còn dư nợ > 0.
    debt_events = position_events[["protocol", "user_address", "reserve", "event_date_dt", "debt_raw_cum"]].copy()
    debt_events = debt_events.sort_values(["protocol", "user_address", "reserve", "event_date_dt"])
    debt_events["next_event_date_dt"] = debt_events.groupby(["protocol", "user_address", "reserve"])["event_date_dt"].shift(-1)
    debt_events["interval_end_dt"] = debt_events["next_event_date_dt"].fillna(STUDY_END + pd.Timedelta(days=1)) - pd.Timedelta(days=1)
    debt_events["interval_end_dt"] = debt_events["interval_end_dt"].clip(upper=STUDY_END)
    debt_events["event_date_dt"] = debt_events["event_date_dt"].clip(lower=STUDY_START)

    active = debt_events[(debt_events["debt_raw_cum"] > 0) & (debt_events["event_date_dt"] <= debt_events["interval_end_dt"])].copy()
    if len(active) == 0:
        raise ValueError("Không tạo được active debt intervals. Kiểm tra raw borrow/repay/liquidation.")

    active["observation_date_dt"] = active.apply(
        lambda r: pd.date_range(r["event_date_dt"], r["interval_end_dt"], freq="D"),
        axis=1,
    )
    active = active.explode("observation_date_dt")
    active = active[["protocol", "user_address", "observation_date_dt"]].drop_duplicates()
    active["observation_date"] = active["observation_date_dt"].dt.strftime("%Y-%m-%d")
    active = active.sort_values(["observation_date_dt", "protocol", "user_address"]).reset_index(drop=True)
    print(f"Active debt user-day rows: {len(active):,}")
    return active


def align_positions_to_user_days(active_user_days: pd.DataFrame, position_events: pd.DataFrame) -> pd.DataFrame:
    # Với mỗi user-day có nợ, lấy trạng thái gần nhất của toàn bộ reserves mà ví từng dùng.
    position_keys = position_events[["protocol", "user_address", "reserve"]].drop_duplicates()
    target = active_user_days[["protocol", "user_address", "observation_date_dt"]].merge(
        position_keys,
        on=["protocol", "user_address"],
        how="inner",
    )
    target = target.rename(columns={"observation_date_dt": "date_dt"})
    print(f"Target protocol-user-reserve-date rows: {len(target):,}")

    right = position_events[[
        "protocol", "user_address", "reserve", "event_date_dt", "collateral_raw_cum", "debt_raw_cum"
    ]].rename(columns={"event_date_dt": "date_dt"}).copy()

    parts = []
    for protocol_value in sorted(target["protocol"].dropna().unique()):
        target_p = target[target["protocol"] == protocol_value].copy()
        right_p = right[right["protocol"] == protocol_value].copy()
        reserves = sorted(target_p["reserve"].dropna().unique())
        print(f"  Align {protocol_value}: {len(reserves):,} reserves")

        for reserve_value in reserves:
            left_r = target_p[target_p["reserve"] == reserve_value].copy()
            right_r = right_p[right_p["reserve"] == reserve_value].copy()
            if len(left_r) == 0:
                continue
            if len(right_r) == 0:
                left_r["collateral_raw_cum"] = 0.0
                left_r["debt_raw_cum"] = 0.0
                parts.append(left_r)
                continue

            left_r = left_r.sort_values(["date_dt", "user_address"]).reset_index(drop=True)
            right_r = right_r.drop(columns=["protocol", "reserve"]).sort_values(["date_dt", "user_address"]).reset_index(drop=True)
            aligned = pd.merge_asof(
                left_r,
                right_r,
                on="date_dt",
                by="user_address",
                direction="backward",
                allow_exact_matches=True,
            )
            parts.append(aligned)

    pos = pd.concat(parts, ignore_index=True)
    pos["collateral_raw_cum"] = pos["collateral_raw_cum"].fillna(0.0)
    pos["debt_raw_cum"] = pos["debt_raw_cum"].fillna(0.0)
    pos = pos[(pos["collateral_raw_cum"] > 0) | (pos["debt_raw_cum"] > 0)].copy()
    pos = pos.rename(columns={"date_dt": "observation_date_dt"})
    pos["observation_date"] = pos["observation_date_dt"].dt.strftime("%Y-%m-%d")
    print(f"Aligned position rows after non-zero filter: {len(pos):,}")
    return pos.reset_index(drop=True)

# ============================================================
# 6. GHÉP PRICE/THRESHOLD VÀ TÍNH HF
# ============================================================

def merge_threshold_asof(pos: pd.DataFrame, threshold: pd.DataFrame) -> pd.DataFrame:
    exact = pos.merge(
        threshold[["protocol", "reserve", "date", "liquidation_threshold_fixed", "decimals_fixed", "status"]].rename(columns={"date": "observation_date", "status": "threshold_status"}),
        on=["protocol", "reserve", "observation_date"],
        how="left",
    )
    missing = exact["liquidation_threshold_fixed"].isna()
    print(f"Missing threshold exact: {int(missing.sum()):,}")

    if missing.sum() == 0:
        return exact

    left = exact.loc[missing, ["protocol", "reserve", "observation_date_dt"]].reset_index().copy()
    right = threshold[["protocol", "reserve", "date_dt", "liquidation_threshold_fixed", "decimals_fixed", "status"]].copy()
    right = right.rename(columns={"date_dt": "threshold_date_dt", "status": "threshold_status_asof"})

    parts = []
    for protocol_value in sorted(left["protocol"].dropna().unique()):
        left_p = left[left["protocol"] == protocol_value].copy()
        right_p = right[right["protocol"] == protocol_value].copy()
        for reserve_value in sorted(left_p["reserve"].dropna().unique()):
            l = left_p[left_p["reserve"] == reserve_value].sort_values("observation_date_dt")
            r = right_p[right_p["reserve"] == reserve_value].drop(columns=["protocol", "reserve"]).sort_values("threshold_date_dt")
            if len(l) == 0 or len(r) == 0:
                continue
            f = pd.merge_asof(
                l,
                r,
                left_on="observation_date_dt",
                right_on="threshold_date_dt",
                direction="backward",
                allow_exact_matches=True,
            )
            parts.append(f)

    if parts:
        filled = pd.concat(parts, ignore_index=True)
        filled["age_days"] = (filled["observation_date_dt"] - filled["threshold_date_dt"]).dt.days
        valid = filled["liquidation_threshold_fixed"].notna() & filled["age_days"].between(0, MAX_THRESHOLD_STALENESS_DAYS)
        idx = filled.loc[valid, "index"]
        exact.loc[idx, "liquidation_threshold_fixed"] = filled.loc[valid, "liquidation_threshold_fixed"].values
        exact.loc[idx, "decimals_fixed"] = filled.loc[valid, "decimals_fixed"].values
        exact.loc[idx, "threshold_status"] = filled.loc[valid, "threshold_status_asof"].values

    # Decimals fallback vẫn được fill bằng map nếu threshold thiếu.
    exact["decimals_fixed"] = exact["decimals_fixed"].fillna(exact["reserve"].map(TOKEN_DECIMALS_FALLBACK)).fillna(18)
    print(f"Missing threshold after asof: {int(exact['liquidation_threshold_fixed'].isna().sum()):,}")
    return exact


def merge_price_asof(pos: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    out = pos.merge(
        price[["reserve", "date", "price_usd"]].rename(columns={"date": "observation_date"}),
        on=["reserve", "observation_date"],
        how="left",
    )
    out["price_fill_method"] = np.where(out["price_usd"].notna(), "exact", "missing")
    out["price_date_used"] = out["observation_date"]

    missing = out["price_usd"].isna()
    print(f"Missing price exact: {int(missing.sum()):,}")

    if missing.sum() == 0:
        return out

    left = out.loc[missing, ["reserve", "observation_date_dt"]].reset_index().copy()
    right = price[["reserve", "date_dt", "date", "price_usd"]].rename(columns={
        "date_dt": "price_date_dt",
        "date": "price_date",
        "price_usd": "price_usd_asof",
    }).copy()

    parts = []
    for reserve_value in sorted(left["reserve"].dropna().unique()):
        l = left[left["reserve"] == reserve_value].sort_values("observation_date_dt")
        r = right[right["reserve"] == reserve_value].drop(columns=["reserve"]).sort_values("price_date_dt")
        if len(l) == 0 or len(r) == 0:
            continue
        f = pd.merge_asof(
            l,
            r,
            left_on="observation_date_dt",
            right_on="price_date_dt",
            direction="backward",
            allow_exact_matches=True,
        )
        parts.append(f)

    if parts:
        filled = pd.concat(parts, ignore_index=True)
        filled["price_age_days"] = (filled["observation_date_dt"] - filled["price_date_dt"]).dt.days
        valid = filled["price_usd_asof"].notna() & filled["price_age_days"].between(0, MAX_PRICE_STALENESS_DAYS)
        idx = filled.loc[valid, "index"]
        out.loc[idx, "price_usd"] = filled.loc[valid, "price_usd_asof"].values
        out.loc[idx, "price_fill_method"] = "asof_backward"
        out.loc[idx, "price_date_used"] = filled.loc[valid, "price_date"].values

    print(f"Missing price after asof: {int(out['price_usd'].isna().sum()):,}")
    return out


def compute_hf(pos: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = pos.copy()
    pos["missing_config_flag"] = pos["liquidation_threshold_fixed"].isna().astype(int)
    pos["missing_price_flag"] = pos["price_usd"].isna().astype(int)

    for c in ["decimals_fixed", "liquidation_threshold_fixed", "price_usd"]:
        pos[c] = safe_numeric(pos[c])

    pos["decimals_power"] = np.power(10.0, pos["decimals_fixed"])
    pos["collateral_token_amount"] = pos["collateral_raw_cum"] / pos["decimals_power"]
    pos["debt_token_amount"] = pos["debt_raw_cum"] / pos["decimals_power"]
    pos["collateral_usd_hf"] = pos["collateral_token_amount"] * pos["price_usd"]
    pos["debt_usd_hf"] = pos["debt_token_amount"] * pos["price_usd"]
    pos["weighted_collateral_usd"] = pos["collateral_usd_hf"] * pos["liquidation_threshold_fixed"]

    for c in ["collateral_usd_hf", "debt_usd_hf", "weighted_collateral_usd"]:
        pos[c] = pos[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    pos["collateral_type_flag"] = np.where(pos["collateral_usd_hf"] > MIN_POSITION_USD, 1, 0)
    pos["debt_type_flag"] = np.where(pos["debt_usd_hf"] > MIN_POSITION_USD, 1, 0)

    hf = (
        pos
        .groupby(["protocol", "user_address", "observation_date"], as_index=False)
        .agg(
            collateral_usd=("collateral_usd_hf", "sum"),
            debt_usd=("debt_usd_hf", "sum"),
            weighted_collateral_usd=("weighted_collateral_usd", "sum"),
            n_collateral_types=("collateral_type_flag", "sum"),
            n_debt_types=("debt_type_flag", "sum"),
            n_missing_config_positions=("missing_config_flag", "sum"),
            n_missing_price_positions=("missing_price_flag", "sum"),
        )
    )

    hf["liquidation_threshold"] = safe_divide(hf["weighted_collateral_usd"], hf["collateral_usd"])
    hf["health_factor_raw"] = safe_divide(hf["weighted_collateral_usd"], hf["debt_usd"])
    hf["health_factor"] = hf["health_factor_raw"].clip(lower=0, upper=HF_CAP)
    hf["distance_to_liquidation_pct"] = (hf["health_factor"] - 1.0) * 100.0

    hf = hf[(hf["collateral_usd"] > MIN_POSITION_USD) & (hf["debt_usd"] > MIN_POSITION_USD)].copy()

    # Tài sản chính: reserve có USD lớn nhất trong từng user-day.
    col_pos = pos[pos["collateral_usd_hf"] > MIN_POSITION_USD].copy()
    debt_pos = pos[pos["debt_usd_hf"] > MIN_POSITION_USD].copy()

    col_idx = col_pos.groupby(["protocol", "user_address", "observation_date"])["collateral_usd_hf"].idxmax()
    debt_idx = debt_pos.groupby(["protocol", "user_address", "observation_date"])["debt_usd_hf"].idxmax()

    primary_col = col_pos.loc[col_idx, ["protocol", "user_address", "observation_date", "reserve"]].rename(columns={"reserve": "collateral_asset_primary"})
    primary_debt = debt_pos.loc[debt_idx, ["protocol", "user_address", "observation_date", "reserve"]].rename(columns={"reserve": "debt_asset_primary"})

    hf = hf.merge(primary_col, on=["protocol", "user_address", "observation_date"], how="left")
    hf = hf.merge(primary_debt, on=["protocol", "user_address", "observation_date"], how="left")
    hf["collateral_asset_primary"] = hf["collateral_asset_primary"].fillna("none")
    hf["debt_asset_primary"] = hf["debt_asset_primary"].fillna("none")

    print(f"HF user-day rows after collateral/debt filter: {len(hf):,}")
    print(hf[["health_factor", "liquidation_threshold", "distance_to_liquidation_pct"]].describe())
    return hf.reset_index(drop=True), pos

# ============================================================
# 7. HÀNH VI NGƯỜI DÙNG, LABEL, FEAR/GREED
# ============================================================

def compute_activity_features(base: pd.DataFrame, all_events: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["observation_date_dt"] = pd.to_datetime(base["observation_date"])

    ev = all_events[["protocol", "user_address", "event_date_dt", "evt_tx_hash"]].dropna().copy()
    ev = ev[(ev["event_date_dt"] >= STUDY_START) & (ev["event_date_dt"] <= STUDY_END)]
    ev["event_day"] = ev["event_date_dt"].dt.floor("D")

    # Lưu duplicate event_day để searchsorted trả về số lượng event, không chỉ số ngày có event.
    ev = ev.sort_values(["protocol", "user_address", "event_day"])

    base["tx_count_7d"] = 0
    base["tx_count_30d"] = 0
    base["inactive_days"] = 0
    base["position_age_days"] = 0

    pieces = []
    grouped_events = {
        k: g["event_day"].values.astype("datetime64[D]")
        for k, g in ev.groupby(["protocol", "user_address"], sort=False)
    }

    for (protocol, user), g in base.groupby(["protocol", "user_address"], sort=False):
        obs = g["observation_date_dt"].values.astype("datetime64[D]")
        ed = grouped_events.get((protocol, user))
        tmp = g.copy()
        if ed is None or len(ed) == 0:
            pieces.append(tmp)
            continue

        ed = np.sort(ed)
        obs_plus_1 = obs + np.timedelta64(1, "D")
        left7 = obs - np.timedelta64(6, "D")
        left30 = obs - np.timedelta64(29, "D")

        tmp["tx_count_7d"] = np.searchsorted(ed, obs_plus_1, side="left") - np.searchsorted(ed, left7, side="left")
        tmp["tx_count_30d"] = np.searchsorted(ed, obs_plus_1, side="left") - np.searchsorted(ed, left30, side="left")

        last_idx = np.searchsorted(ed, obs_plus_1, side="left") - 1
        valid_last = last_idx >= 0
        inactive = np.zeros(len(tmp), dtype=int)
        inactive[valid_last] = (obs[valid_last] - ed[last_idx[valid_last]]).astype("timedelta64[D]").astype(int)
        tmp["inactive_days"] = inactive

        first_day = ed[0]
        tmp["position_age_days"] = (obs - first_day).astype("timedelta64[D]").astype(int)
        tmp["position_age_days"] = tmp["position_age_days"].clip(lower=0)
        pieces.append(tmp)

    out = pd.concat(pieces, ignore_index=True)
    out["inactivity_flag_30d"] = (out["inactive_days"] >= 30).astype(int)
    return out


def add_liquidation_label(base: pd.DataFrame, liquidation: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    base["observation_date_dt"] = pd.to_datetime(base["observation_date"])

    liq = liquidation.copy()
    liq = liq.sort_values(["protocol", "user_address", "liquidation_date_dt", "liquidation_block"])
    liq_daily = (
        liq.groupby(["protocol", "user_address", "liquidation_date"], as_index=False)
        .agg(
            liquidator_address=("liquidator_address", "first"),
            liquidation_block=("liquidation_block", "first"),
            liquidation_tx_hash=("evt_tx_hash", "first"),
        )
    )
    liq_daily["liquidation_date_dt"] = pd.to_datetime(liq_daily["liquidation_date"])

    candidates = []
    for k in range(1, LABEL_HORIZON_DAYS + 1):
        temp = liq_daily.copy()
        temp["observation_date_dt"] = temp["liquidation_date_dt"] - pd.Timedelta(days=k)
        temp["observation_date"] = temp["observation_date_dt"].dt.strftime("%Y-%m-%d")
        temp["days_ahead"] = k
        candidates.append(temp)

    label_map = pd.concat(candidates, ignore_index=True)
    label_map = label_map[
        (label_map["observation_date_dt"] >= STUDY_START)
        & (label_map["observation_date_dt"] <= STUDY_END)
    ].copy()
    label_map = label_map.sort_values(["protocol", "user_address", "observation_date_dt", "days_ahead", "liquidation_block"])
    label_map = label_map.drop_duplicates(["protocol", "user_address", "observation_date"], keep="first")

    base = base.merge(
        label_map[["protocol", "user_address", "observation_date", "liquidator_address", "liquidation_block", "days_ahead"]],
        on=["protocol", "user_address", "observation_date"],
        how="left",
    )
    base["liquidated_next_3d"] = base["days_ahead"].notna().astype(int)
    base["liquidator_address"] = base["liquidator_address"].fillna("none")
    base.loc[base["liquidated_next_3d"] == 0, "liquidation_block"] = np.nan
    base["has_liquidation_metadata"] = (
        (base["liquidated_next_3d"] == 1)
        & base["liquidator_address"].ne("none")
        & base["liquidation_block"].notna()
    ).astype(int)
    base = base.drop(columns=["days_ahead"], errors="ignore")
    return base

# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def main():
    report = []
    section("1. ĐỌC RAW EVENTS")
    supply = read_event_files(SUPPLY_FILES, "supply", required=True)
    borrow = read_event_files(BORROW_FILES, "borrow", required=True)
    repay = read_event_files(REPAY_FILES, "repay", required=True)
    withdraw = read_event_files(WITHDRAW_FILES, "withdraw", required=False)
    liquidation = read_liquidation_files(LIQUIDATION_FILES)

    if len(withdraw) == 0:
        print("[WARN] Không có raw withdraw event. Nếu người dùng từng rút collateral, collateral_usd/HF có thể bị ước lượng cao hơn thực tế.")

    section("2. ĐỌC MARKET DATA")
    threshold = read_threshold()
    price = read_price()
    fear_greed = read_fear_greed()

    section("3. TẠO DELTA POSITION VÀ ACTIVE USER-DAY")
    position_events, all_event_rows = build_delta_positions(supply, borrow, repay, withdraw, liquidation)
    active_user_days = expand_active_debt_user_days(position_events)
    pos = align_positions_to_user_days(active_user_days, position_events)

    section("4. GHÉP THRESHOLD / PRICE VÀ TÍNH HEALTH FACTOR")
    pos = merge_threshold_asof(pos, threshold)
    pos = merge_price_asof(pos, price)
    hf, reserve_positions = compute_hf(pos)

    reserve_positions_out_cols = [
        "protocol", "user_address", "observation_date", "reserve",
        "collateral_raw_cum", "debt_raw_cum", "decimals_fixed", "price_usd",
        "price_fill_method", "price_date_used", "liquidation_threshold_fixed", "threshold_status",
        "collateral_usd_hf", "debt_usd_hf", "weighted_collateral_usd",
        "missing_config_flag", "missing_price_flag",
    ]
    reserve_positions[reserve_positions_out_cols].to_csv(RESERVE_POSITION_FILE, index=False, encoding="utf-8-sig")
    print(f"Saved reserve positions: {RESERVE_POSITION_FILE}")

    section("5. THÊM HÀNH VI NGƯỜI DÙNG")
    # Chuẩn hóa all_events phục vụ tính activity.
    basic_events = []
    for df_event in [supply, borrow, repay, withdraw]:
        if len(df_event):
            basic_events.append(df_event[["protocol", "user_address", "event_date_dt", "evt_tx_hash", "event_type"]])
    if len(liquidation):
        liq_ev = liquidation[["protocol", "user_address", "liquidation_date_dt", "evt_tx_hash"]].rename(columns={"liquidation_date_dt": "event_date_dt"}).copy()
        liq_ev["event_type"] = "liquidation"
        basic_events.append(liq_ev)
    all_events_for_activity = pd.concat(basic_events, ignore_index=True)

    df = compute_activity_features(hf, all_events_for_activity)

    section("6. GẮN LABEL THANH LÝ NEXT 3 DAYS")
    df = add_liquidation_label(df, liquidation)
    print(df["liquidated_next_3d"].value_counts(dropna=False).sort_index())
    print(f"Positive rate: {df['liquidated_next_3d'].mean():.6f}")

    section("7. GHÉP FEAR & GREED VÀ CLEAN FINAL")
    df = df.merge(fear_greed, on="observation_date", how="left")
    # Nếu thiếu ngày, fill theo thời gian.
    df["observation_date_dt"] = pd.to_datetime(df["observation_date"])
    df = df.sort_values(["observation_date_dt", "protocol", "user_address"]).reset_index(drop=True)
    fg_cols = ["fear_greed_index", "extreme_fear_flag", "fear_flag", "neutral_flag", "greed_flag", "extreme_greed_flag"]
    df["fear_greed_index"] = df["fear_greed_index"].ffill().bfill()
    if df["fear_greed_index"].isna().any():
        df["fear_greed_index"] = df["fear_greed_index"].fillna(50)
    df["fear_greed_index"] = df["fear_greed_index"].clip(0, 100).round().astype(int)
    df = recompute_fear_greed_flags(df)

    df["snapshot"] = snapshot_from_date(df["observation_date_dt"])
    df["log_collateral_usd"] = np.log1p(df["collateral_usd"])
    df["log_debt_usd"] = np.log1p(df["debt_usd"])

    numeric_int_cols = ["n_collateral_types", "tx_count_7d", "tx_count_30d", "inactive_days", "inactivity_flag_30d", "position_age_days"]
    for c in numeric_int_cols:
        df[c] = safe_numeric(df[c]).fillna(0).clip(lower=0).round().astype(int)

    # Đảm bảo tx_count_30d không nhỏ hơn tx_count_7d.
    bad_tx = df["tx_count_7d"] > df["tx_count_30d"]
    df.loc[bad_tx, "tx_count_30d"] = df.loc[bad_tx, "tx_count_7d"]

    # Đảm bảo tuổi vị thế không nhỏ hơn số ngày inactive.
    bad_age = df["position_age_days"] < df["inactive_days"]
    df.loc[bad_age, "position_age_days"] = df.loc[bad_age, "inactive_days"]

    # Clean address fields.
    for c in ["user_address", "collateral_asset_primary", "debt_asset_primary", "liquidator_address"]:
        df[c] = clean_address_series(df[c])

    df["liquidation_block"] = safe_numeric(df["liquidation_block"])
    df.loc[df["liquidated_next_3d"] == 0, "liquidator_address"] = "none"
    df.loc[df["liquidated_next_3d"] == 0, "liquidation_block"] = np.nan
    df["has_liquidation_metadata"] = (
        (df["liquidated_next_3d"] == 1)
        & df["liquidator_address"].ne("none")
        & df["liquidation_block"].notna()
    ).astype(int)

    # Chọn đúng bộ cột cuối.
    missing_cols = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu final columns: {missing_cols}")

    analysis = df[FINAL_COLUMNS].copy()
    analysis = analysis.sort_values(["observation_date", "protocol", "user_address"]).reset_index(drop=True)

    section("8. VALIDATION CUỐI")
    key_cols = ["protocol", "user_address", "observation_date"]
    checks = {
        "duplicate_key": int(analysis.duplicated(key_cols).sum()),
        "missing_key": int(analysis[key_cols].isna().sum().sum()),
        "bad_health_factor": int((analysis["health_factor"].isna() | (analysis["health_factor"] < 0) | (analysis["health_factor"] > HF_CAP)).sum()),
        "bad_liquidation_threshold": int((analysis["liquidation_threshold"].isna() | (analysis["liquidation_threshold"] < 0) | (analysis["liquidation_threshold"] > 1)).sum()),
        "bad_collateral_usd": int((analysis["collateral_usd"].isna() | (analysis["collateral_usd"] <= 0)).sum()),
        "bad_debt_usd": int((analysis["debt_usd"].isna() | (analysis["debt_usd"] <= 0)).sum()),
        "bad_label": int((~analysis["liquidated_next_3d"].isin([0, 1])).sum()),
        "bad_user_address": check_eth_address_or_none(analysis["user_address"]),
        "bad_collateral_address": check_eth_address_or_none(analysis["collateral_asset_primary"]),
        "bad_debt_address": check_eth_address_or_none(analysis["debt_asset_primary"]),
        "bad_liquidator_address": check_eth_address_or_none(analysis["liquidator_address"]),
        "fear_greed_flag_sum_bad": int((analysis[["extreme_fear_flag", "fear_flag", "neutral_flag", "greed_flag", "extreme_greed_flag"]].sum(axis=1) != 1).sum()),
        "label0_has_liquidator": int(((analysis["liquidated_next_3d"] == 0) & analysis["liquidator_address"].ne("none")).sum()),
        "label0_has_liquidation_block": int(((analysis["liquidated_next_3d"] == 0) & analysis["liquidation_block"].notna()).sum()),
    }

    distance_mismatch = ~np.isclose(
        analysis["distance_to_liquidation_pct"],
        (analysis["health_factor"] - 1.0) * 100.0,
        rtol=1e-9,
        atol=1e-6,
    )
    checks["distance_mismatch"] = int(distance_mismatch.sum())

    # Không bắt liquidation_block phải có ở label=0.
    must_not_missing = [c for c in FINAL_COLUMNS if c not in ["liquidation_block"]]
    checks["missing_required_total_exclude_liquidation_block"] = int(analysis[must_not_missing].isna().sum().sum())

    for k, v in checks.items():
        print(f"{k}: {v:,}")

    bad_total = sum(checks.values())
    if bad_total > 0:
        print("\nMissing by column:")
        print(analysis[must_not_missing].isna().sum()[analysis[must_not_missing].isna().sum() > 0])
        raise ValueError("Validation cuối còn lỗi. Xem các check phía trên.")

    section("9. XUẤT FILE")
    analysis.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"Đã xuất file: {OUTPUT_FILE}")
    print(f"Rows: {len(analysis):,}")
    print(f"Columns: {len(analysis.columns):,}")

    section("10. GHI REPORT")
    label_counts = analysis["liquidated_next_3d"].value_counts(dropna=False).sort_index()
    positive_rate = analysis["liquidated_next_3d"].mean()

    report.extend([
        "AAVE MODEL READY ANALYSIS V2 REPORT - RAW TO FINAL PIPELINE",
        "=" * 100,
        f"Run time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Raw dir: {RAW_DIR}",
        f"Output file: {OUTPUT_FILE}",
        "",
        "INPUT SUMMARY",
        f"- Supply rows: {len(supply):,}",
        f"- Borrow rows: {len(borrow):,}",
        f"- Repay rows: {len(repay):,}",
        f"- Withdraw rows: {len(withdraw):,}",
        f"- Liquidation rows: {len(liquidation):,}",
        f"- Threshold rows: {len(threshold):,}",
        f"- Price rows: {len(price):,}",
        f"- Fear & Greed rows: {len(fear_greed):,}",
        "",
        "OUTPUT SUMMARY",
        f"- Output rows: {len(analysis):,}",
        f"- Output columns: {len(analysis.columns):,}",
        f"- Positive rate: {positive_rate:.6f}",
        "",
        "LABEL COUNTS",
        label_counts.to_string(),
        "",
        "SNAPSHOT COUNTS",
        analysis["snapshot"].value_counts().to_string(),
        "",
        "PROTOCOL COUNTS",
        analysis["protocol"].value_counts().to_string(),
        "",
        "HEALTH_FACTOR DESCRIBE",
        analysis["health_factor"].describe().to_string(),
        "",
        "LIQUIDATION_THRESHOLD DESCRIBE",
        analysis["liquidation_threshold"].describe().to_string(),
        "",
        "DISTANCE_TO_LIQUIDATION_PCT DESCRIBE",
        analysis["distance_to_liquidation_pct"].describe().to_string(),
        "",
        "VALIDATION CHECKS",
    ])
    for k, v in checks.items():
        report.append(f"- {k}: {v:,}")
    report.extend([
        "",
        "IMPORTANT NOTES",
        "- liquidator_address, liquidation_block and has_liquidation_metadata are metadata/leakage columns.",
        "- Do NOT use these metadata columns as model input features.",
        "- health_factor is capped at 100 for model stability.",
        "- distance_to_liquidation_pct = (health_factor - 1) * 100.",
        "- If withdraw raw is absent, collateral/HF may be over-estimated for users who withdrew collateral.",
        "",
        "FINAL COLUMNS",
    ])
    for c in analysis.columns:
        report.append(f"- {c}")

    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
    print(f"Report saved: {REPORT_FILE}")

    section("11. HOÀN TẤT")
    print("File cuối cùng:")
    print(OUTPUT_FILE)
    print("\nReport:")
    print(REPORT_FILE)


if __name__ == "__main__":
    main()
