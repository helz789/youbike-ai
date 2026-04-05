# 這是把抓到的 YouBike 資料做成訓練資料的程式
# 讀取 collect_snapshots.py 定期抓到的快照資料
# 產生未來 15 分鐘的欄位（future_available_rent_bikes, future_available_return_bikes, future_act）
# 做標籤（label_borrow_risk_15m, label_return_risk_15m）
# 存成 CSV 到 data/collector/youbike_training_data.csv

from pathlib import Path
import pandas as pd

INPUT_FILE = Path("data/collector/youbike_snapshots.csv")
OUTPUT_FILE = Path("data/collector/youbike_training_data.csv")

FUTURE_STEPS = 3  # 每 5 分鐘一筆，3 步 = 15 分鐘後
RISK_THRESHOLD = 3


def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    # 時間欄位轉型
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], errors="coerce")
    df = df.dropna(subset=["snapshot_time"])

    # 數值欄位轉型
    numeric_cols = [
        "quantity",
        "available_rent_bikes",
        "available_return_bikes",
        "latitude",
        "longitude",
        "snapshot_hour",
        "snapshot_minute",
        "snapshot_weekday",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # act 統一成字串
    if "act" in df.columns:
        df["act"] = df["act"].astype(str)

    # 依站點與時間排序
    df = df.sort_values(["city", "sno", "snapshot_time"]).reset_index(drop=True)

    # 產生 15 分鐘後的欄位
    group_cols = ["city", "sno"]

    df["future_snapshot_time"] = df.groupby(group_cols)["snapshot_time"].shift(-FUTURE_STEPS)
    df["future_available_rent_bikes"] = df.groupby(group_cols)["available_rent_bikes"].shift(-FUTURE_STEPS)
    df["future_available_return_bikes"] = df.groupby(group_cols)["available_return_bikes"].shift(-FUTURE_STEPS)
    df["future_act"] = df.groupby(group_cols)["act"].shift(-FUTURE_STEPS)

    # 計算實際時間差（避免中間有漏抓）
    df["minutes_ahead"] = (
        (df["future_snapshot_time"] - df["snapshot_time"]).dt.total_seconds() / 60
    )

    # 只保留真的接近 15 分鐘後的資料
    df = df[df["minutes_ahead"].between(10, 20, inclusive="both")].copy()

    # 做標籤
    # 借車模式：15 分鐘後可借車數 <= 3 視為高風險
    df["label_borrow_risk_15m"] = (
        (df["future_act"] == "1") &
        (df["future_available_rent_bikes"] <= RISK_THRESHOLD)
    ).astype(int)

    # 還車模式：15 分鐘後可還車位 <= 3 視為高風險
    df["label_return_risk_15m"] = (
        (df["future_act"] == "1") &
        (df["future_available_return_bikes"] <= RISK_THRESHOLD)
    ).astype(int)

    # 可額外保留「未來完全沒車 / 沒位」版本
    df["label_borrow_empty_15m"] = (
        (df["future_act"] == "1") &
        (df["future_available_rent_bikes"] == 0)
    ).astype(int)

    df["label_return_full_15m"] = (
        (df["future_act"] == "1") &
        (df["future_available_return_bikes"] == 0)
    ).astype(int)

    # 你之後模型會用到的主要欄位
    keep_cols = [
        "city",
        "sno",
        "sna",
        "sarea",
        "ar",
        "latitude",
        "longitude",
        "act",
        "quantity",
        "available_rent_bikes",
        "available_return_bikes",
        "snapshot_time",
        "snapshot_hour",
        "snapshot_minute",
        "snapshot_weekday",
        "future_snapshot_time",
        "future_available_rent_bikes",
        "future_available_return_bikes",
        "minutes_ahead",
        "label_borrow_risk_15m",
        "label_return_risk_15m",
        "label_borrow_empty_15m",
        "label_return_full_15m",
    ]

    existing_cols = [c for c in keep_cols if c in df.columns]
    training_df = df[existing_cols].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("訓練資料已建立")
    print(f"輸出檔案：{OUTPUT_FILE}")
    print(f"資料筆數：{len(training_df)}")
    print("\n欄位：")
    print(training_df.columns.tolist())

    print("\n前 5 筆：")
    print(training_df.head())

    print("\n借車 15 分鐘高風險比例：")
    print(training_df["label_borrow_risk_15m"].value_counts(dropna=False))

    print("\n還車 15 分鐘高風險比例：")
    print(training_df["label_return_risk_15m"].value_counts(dropna=False))


if __name__ == "__main__":
    main()