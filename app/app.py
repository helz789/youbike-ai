# 這是 Streamlit App 的主程式
# 主要功能：  

import requests
import pandas as pd
import folium
import streamlit as st
from streamlit_folium import st_folium

TAIPEI_URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
NEW_TAIPEI_URL = "https://data.ntpc.gov.tw/api/datasets/010e5b15-3823-4b20-b401-b1cf000550c5/json?size=2000"


@st.cache_data(ttl=60)
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


@st.cache_data(ttl=300)
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


def get_risk_label(row) -> str:
    act = str(row.get("act", ""))
    rent = row.get("available_rent_bikes", 0)
    ret = row.get("available_return_bikes", 0)

    if pd.isna(rent):
        rent = 0
    if pd.isna(ret):
        ret = 0

    if act != "1":
        return "停用站"
    if rent == 0 or ret == 0:
        return "高風險"
    if rent <= 3 or ret <= 3:
        return "中風險"
    return "正常"


def get_risk_color(risk_label: str) -> str:
    color_map = {
        "停用站": "gray",
        "高風險": "red",
        "中風險": "orange",
        "正常": "green",
    }
    return color_map.get(risk_label, "blue")

def render_sidebar_legend():
    st.sidebar.markdown("---")
    st.sidebar.subheader("地圖圖例")

    st.sidebar.markdown(
        """
<div style="line-height: 1.9;">
<span style="color:red; font-size:18px;">●</span> 高風險：完全沒車可借，或完全沒位可還<br>
<span style="color:orange; font-size:18px;">●</span> 中風險：可借車數 ≤ 3，或可還車位 ≤ 3<br>
<span style="color:green; font-size:18px;">●</span> 正常：其餘正常站點<br>
<span style="color:gray; font-size:18px;">●</span> 停用站：站點未啟用
</div>
        """,
        unsafe_allow_html=True,
    )

def filter_data(
    df: pd.DataFrame,
    selected_cities,
    selected_districts,
    selected_risks,
    keyword,
) -> pd.DataFrame:
    result = df.copy()

    if selected_cities:
        result = result[result["city"].isin(selected_cities)]

    if selected_districts:
        result = result[result["sarea"].isin(selected_districts)]

    if selected_risks:
        result = result[result["risk_label"].isin(selected_risks)]

    keyword = keyword.strip()
    if keyword:
        result = result[
            result["sna"].astype(str).str.contains(keyword, case=False, na=False)
        ]

    return result


def build_map(df: pd.DataFrame) -> folium.Map:
    if df.empty:
        m = folium.Map(location=[25.03, 121.52], zoom_start=11, tiles=None)
        folium.TileLayer(
            tiles="CartoDB positron",
            name="淡色底圖",
            control=False,
            opacity=0.72,
        ).add_to(m)
        return m

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)
    folium.TileLayer(
        tiles="CartoDB positron",
        name="淡色底圖",
        control=False,
        opacity=0.72,
    ).add_to(m)

    for _, row in df.iterrows():
        risk_label = row["risk_label"]
        color = get_risk_color(risk_label)

        popup_html = f"""
        <b>{row.get('sna', '未知站點')}</b><br>
        城市：{row.get('city', '未知')}<br>
        行政區：{row.get('sarea', '未知')}<br>
        地址：{row.get('ar', '未知')}<br>
        可借車數：{row.get('available_rent_bikes', 'N/A')}<br>
        可還車位：{row.get('available_return_bikes', 'N/A')}<br>
        總車格：{row.get('quantity', 'N/A')}<br>
        風險等級：{risk_label}<br>
        更新時間：{row.get('info_time', 'N/A')}
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            popup=folium.Popup(popup_html, max_width=320),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            weight=1,
        ).add_to(m)

    bounds = [
        [df["latitude"].min(), df["longitude"].min()],
        [df["latitude"].max(), df["longitude"].max()],
    ]
    m.fit_bounds(bounds)

    return m


def main():
    st.set_page_config(page_title="雙北 YouBike 即時風險地圖", layout="wide")

    st.title("雙北 YouBike 即時風險地圖")
    st.caption("臺北市 + 新北市即時站點資料")

    if st.button("重新整理資料"):
        st.cache_data.clear()
        st.rerun()

    try:
        df = load_all_data()
    except Exception as e:
        st.error(f"資料抓取失敗：{e}")
        return

    df["risk_label"] = df.apply(get_risk_label, axis=1)

    st.sidebar.header("篩選條件")

    city_options = sorted(df["city"].dropna().unique().tolist())
    selected_cities = st.sidebar.multiselect(
        "選擇城市",
        options=city_options,
        default=city_options,
    )

    district_source = df[df["city"].isin(selected_cities)] if selected_cities else df
    district_options = sorted(district_source["sarea"].dropna().unique().tolist())
    selected_districts = st.sidebar.multiselect(
        "選擇行政區",
        options=district_options,
        default=[],
    )

    risk_options = ["高風險", "中風險", "正常", "停用站"]
    selected_risks = st.sidebar.multiselect(
        "選擇風險等級",
        options=risk_options,
        default=risk_options,
    )

    keyword = st.sidebar.text_input("搜尋站名", value="")
    render_sidebar_legend()

    filtered_df = filter_data(
        df=df,
        selected_cities=selected_cities,
        selected_districts=selected_districts,
        selected_risks=selected_risks,
        keyword=keyword,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("目前顯示站點數", len(filtered_df))
    col2.metric("高風險", int((filtered_df["risk_label"] == "高風險").sum()))
    col3.metric("中風險", int((filtered_df["risk_label"] == "中風險").sum()))
    col4.metric("停用站", int((filtered_df["risk_label"] == "停用站").sum()))

    st.subheader("站點地圖")
    map_obj = build_map(filtered_df)
    st_folium(map_obj, width=1200, height=680)

    st.subheader("資料表")
    show_cols = [
        "city",
        "sno",
        "sna",
        "sarea",
        "available_rent_bikes",
        "available_return_bikes",
        "quantity",
        "risk_label",
        "info_time",
    ]
    existing_cols = [c for c in show_cols if c in filtered_df.columns]
    st.dataframe(filtered_df[existing_cols], width="stretch", height=350)


if __name__ == "__main__":
    main()