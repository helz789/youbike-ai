# 這是 Streamlit App 的程式
# 會抓 YouBike 即時資料，顯示在地圖上

import requests
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium

URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"


def fetch_youbike_data() -> pd.DataFrame:
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data)

def get_risk_color(row) -> str:
    act = str(row.get("act", ""))
    rent = pd.to_numeric(row.get("available_rent_bikes", 0), errors="coerce")
    ret = pd.to_numeric(row.get("available_return_bikes", 0), errors="coerce")

    if pd.isna(rent):
        rent = 0
    if pd.isna(ret):
        ret = 0

    if act != "1":
        return "gray"
    if rent == 0 or ret == 0:
        return "red"
    if rent <= 3 or ret <= 3:
        return "orange"
    return "green"


def main():
    st.set_page_config(page_title="YouBike 即時風險地圖", layout="wide")
    st.title("YouBike 即時風險地圖")
    st.write("顯示臺北市 YouBike 站點的即時借車／還車風險。")

    if st.button("重新整理資料"):
        st.rerun()

    try:
        df = fetch_youbike_data()
    except Exception as e:
        st.error(f"資料抓取失敗：{e}")
        return

    numeric_cols = [
        "latitude",
        "longitude",
        "available_rent_bikes",
        "available_return_bikes",
        "quantity",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["latitude", "longitude"])

    st.subheader("資料概況")
    col1, col2, col3 = st.columns(3)

    active_count = (df["act"].astype(str) == "1").sum() if "act" in df.columns else 0
    red_count = sum(get_risk_color(row) == "red" for _, row in df.iterrows())

    col1.metric("站點總數", len(df))
    col2.metric("啟用站點數", int(active_count))
    col3.metric("高風險站點數", int(red_count))

    m = folium.Map(location=[25.0418, 121.5500], zoom_start=12)

    for _, row in df.iterrows():
        color = get_risk_color(row)

        station_name = row.get("sna", "未知站點")
        area = row.get("sarea", "未知行政區")
        rent = row.get("available_rent_bikes", "N/A")
        ret = row.get("available_return_bikes", "N/A")
        quantity = row.get("quantity", "N/A")
        info_time = row.get("infoTime", "N/A")

        popup_html = f"""
        <b>{station_name}</b><br>
        行政區：{area}<br>
        可借車數：{rent}<br>
        可還車位：{ret}<br>
        總車格：{quantity}<br>
        更新時間：{info_time}
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
        ).add_to(m)

    st.subheader("站點地圖")
    st_folium(m, width=1200, height=650)

    st.subheader("顏色規則")
    st.markdown(
        """
- 灰色：站點未啟用
- 紅色：完全沒車可借，或完全沒位可還
- 橘色：可借車數 ≤ 3，或可還車位 ≤ 3
- 綠色：目前狀態正常
        """
    )

    st.subheader("前 10 筆資料")
    show_cols = [
        "sno",
        "sna",
        "sarea",
        "available_rent_bikes",
        "available_return_bikes",
        "quantity",
        "act",
        "infoTime",
    ]
    existing_cols = [c for c in show_cols if c in df.columns]
    st.dataframe(df[existing_cols].head(10), width="stretch")


if __name__ == "__main__":
    main()