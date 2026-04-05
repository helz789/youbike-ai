# 這是抓 YouBike 資料的程式
# 用 requests.get()抓官方 JSON
# 用 pandas.DataFrame() 把 JSON 轉成 DataFrame
# 用目前時間命名檔案，存成 CSV
# 存成 CSV到 data/raw/

from pathlib import Path
from datetime import datetime
import requests
import pandas as pd

URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    response = requests.get(URL, timeout=30)
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"youbike_{timestamp}.csv"

    keep_cols = [
        "sno",
        "sna",
        "sarea",
        "quantity",
        "available_rent_bikes",
        "available_return_bikes",
        "latitude",
        "longitude",
        "act",
        "infoTime",
    ]

    existing_cols = [c for c in keep_cols if c in df.columns]

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("前 5 筆資料：")
    print(df[existing_cols].head())

    print(f"\n總筆數：{len(df)}")
    print(f"已存檔：{csv_path}")


if __name__ == "__main__":
    main()