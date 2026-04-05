# 這是定期抓 YouBike 資料的程式
# 每 5 分鐘抓一次雙北市的 YouBike 資料
# 用 requests.get() 抓官方 JSON
# 用 pandas.DataFrame() 把 JSON 轉成 DataFrame
# 加入時間特徵（snapshot_time, snapshot_date, snapshot_hour, snapshot_minute, snapshot_weekday）
# 把資料存成 CSV，附加在同一個檔案（data/collector/youbike_snapshots.csv）

from pathlib import Path
from datetime import datetime
import time
import requests
import pandas as pd

TAIPEI_URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
NEW_TAIPEI_URL = "https://data.ntpc.gov.tw/api/datasets/010e5b15-3823-4b20-b401-b1cf000550c5/json?size=2000"

OUTFILE = Path("data/collector/youbike_snapshots.csv")
OUTFILE.parent.mkdir(parents=True, exist_ok=True)

INTERVAL_SECONDS = 300  # 每 5 分鐘抓一次


def fetch_taipei_data() -> pd.DataFrame:
    response = requests.get(TAIPEI_URL, timeout=30)
    response.raise_for_status()
    df = pd.DataFrame(response.json())

    df["city"] = "臺北市"

    keep_cols = [
        "city",
        "sno",
        "sna",
        "sarea",
        "ar",
        "quantity",
        "available_rent_bikes",
        "available_return_bikes",
        "latitude",
        "longitude",
        "act",
        "infoTime",
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    if "infoTime" in df.columns:
        df = df.rename(columns={"infoTime": "info_time"})

    return df


def fetch_new_taipei_data() -> pd.DataFrame:
    response = requests.get(NEW_TAIPEI_URL, timeout=30)
    response.raise_for_status()
    df = pd.DataFrame(response.json())

    df = df.rename(
        columns={
            "scity": "city",
            "tot_quantity": "quantity",
            "sbi_quantity": "available_rent_bikes",
            "bemp": "available_return_bikes",
            "lat": "latitude",
            "lng": "longitude",
            "mday": "info_time",
        }
    )

    keep_cols = [
        "city",
        "sno",
        "sna",
        "sarea",
        "ar",
        "quantity",
        "available_rent_bikes",
        "available_return_bikes",
        "latitude",
        "longitude",
        "act",
        "info_time",
    ]
    existing_cols = [c for c in keep_cols if c in df.columns]
    return df[existing_cols].copy()


def load_all_data() -> pd.DataFrame:
    taipei_df = fetch_taipei_data()
    new_taipei_df = fetch_new_taipei_data()

    df = pd.concat([taipei_df, new_taipei_df], ignore_index=True)

    numeric_cols = [
        "quantity",
        "available_rent_bikes",
        "available_return_bikes",
        "latitude",
        "longitude",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "act" in df.columns:
        df["act"] = df["act"].astype(str)

    df = df.dropna(subset=["latitude", "longitude"])
    return df


def add_time_features(df: pd.DataFrame, snapshot_time: datetime) -> pd.DataFrame:
    df = df.copy()
    df["snapshot_time"] = snapshot_time.strftime("%Y-%m-%d %H:%M:%S")
    df["snapshot_date"] = snapshot_time.strftime("%Y-%m-%d")
    df["snapshot_hour"] = snapshot_time.hour
    df["snapshot_minute"] = snapshot_time.minute
    df["snapshot_weekday"] = snapshot_time.weekday()
    return df


def save_snapshot(df: pd.DataFrame) -> None:
    write_header = not OUTFILE.exists()
    df.to_csv(
        OUTFILE,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    print("開始蒐集雙北 YouBike 快照資料。")
    print(f"輸出檔案：{OUTFILE}")
    print("按 Ctrl + C 可停止。")

    while True:
        try:
            now = datetime.now()
            df = load_all_data()
            df = add_time_features(df, now)

            save_snapshot(df)

            print(
                f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"已寫入 {len(df)} 筆資料到 {OUTFILE}"
            )

        except Exception as e:
            print(f"抓取失敗：{e}")

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()