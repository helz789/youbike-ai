# 這是 Streamlit App 的主程式
# 主要功能：  

import requests
import pandas as pd
import folium
import streamlit as st
import joblib
from datetime import datetime
from streamlit_folium import st_folium

TAIPEI_URL = "https://tcgbusfs.blob.core.windows.net/dotapp/youbike/v2/youbike_immediate.json"
NEW_TAIPEI_URL = "https://data.ntpc.gov.tw/api/datasets/010e5b15-3823-4b20-b401-b1cf000550c5/json?size=2000"

BORROW_MODEL_FILE = "models/borrow_risk_model.joblib"
RETURN_MODEL_FILE = "models/return_risk_model.joblib"

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

@st.cache_resource
def load_models():
    borrow_model = joblib.load(BORROW_MODEL_FILE)
    return_model = joblib.load(RETURN_MODEL_FILE)
    return borrow_model, return_model

def predict_risk_for_current_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    now = datetime.now()
    df["snapshot_hour"] = now.hour
    df["snapshot_minute"] = now.minute
    df["snapshot_weekday"] = now.weekday()

    borrow_model, return_model = load_models()

    feature_cols = [
        "city",
        "sarea",
        "latitude",
        "longitude",
        "quantity",
        "available_rent_bikes",
        "available_return_bikes",
        "snapshot_hour",
        "snapshot_minute",
        "snapshot_weekday",
    ]

    df["pred_borrow_risk_prob_15m"] = 0.0
    df["pred_return_risk_prob_15m"] = 0.0

    active_mask = df["act"].astype(str) == "1"

    if active_mask.any():
        X_pred = df.loc[active_mask, feature_cols].copy()

        df.loc[active_mask, "pred_borrow_risk_prob_15m"] = borrow_model.predict_proba(X_pred)[:, 1]
        df.loc[active_mask, "pred_return_risk_prob_15m"] = return_model.predict_proba(X_pred)[:, 1]

    return df

def get_risk_label(row, mode: str) -> str:
    act = str(row.get("act", ""))
    rent = row.get("available_rent_bikes", 0)
    ret = row.get("available_return_bikes", 0)

    if pd.isna(rent):
        rent = 0
    if pd.isna(ret):
        ret = 0

    if act != "1":
        return "停用站"

    if mode == "我要借車":
        if rent == 0:
            return "高風險"
        if rent <= 3:
            return "中風險"
        return "正常"

    if mode == "我要還車":
        if ret == 0:
            return "高風險"
        if ret <= 3:
            return "中風險"
        return "正常"

    # 預設：一起看
    if rent == 0 or ret == 0:
        return "高風險"
    if rent <= 3 or ret <= 3:
        return "中風險"
    return "正常"

def get_predicted_risk_label(row, risk_mode: str) -> str:
    act = str(row.get("act", ""))

    if act != "1":
        return "停用站"

    borrow_prob = row.get("pred_borrow_risk_prob_15m", 0)
    return_prob = row.get("pred_return_risk_prob_15m", 0)

    if pd.isna(borrow_prob):
        borrow_prob = 0
    if pd.isna(return_prob):
        return_prob = 0

    if risk_mode == "我要借車":
        score = borrow_prob
    elif risk_mode == "我要還車":
        score = return_prob
    else:
        score = max(borrow_prob, return_prob)

    if score >= 0.60:
        return "高風險"
    if score >= 0.30:
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

def render_sidebar_legend(risk_mode: str, display_mode: str):
    st.sidebar.markdown("---")
    st.sidebar.subheader("地圖圖例")

    if display_mode == "15分鐘後預測風險":
        if risk_mode == "我要借車":
            st.sidebar.markdown(
                """
<div style="line-height: 1.9;">
<span style="color:red; font-size:18px;">●</span> 高風險：15 分鐘後借車高風險機率高<br>
<span style="color:orange; font-size:18px;">●</span> 中風險：15 分鐘後借車風險中等<br>
<span style="color:green; font-size:18px;">●</span> 正常：15 分鐘後借車風險低<br>
<span style="color:gray; font-size:18px;">●</span> 停用站：站點未啟用
</div>
                """,
                unsafe_allow_html=True,
            )
            return

        if risk_mode == "我要還車":
            st.sidebar.markdown(
                """
<div style="line-height: 1.9;">
<span style="color:red; font-size:18px;">●</span> 高風險：15 分鐘後還車高風險機率高<br>
<span style="color:orange; font-size:18px;">●</span> 中風險：15 分鐘後還車風險中等<br>
<span style="color:green; font-size:18px;">●</span> 正常：15 分鐘後還車風險低<br>
<span style="color:gray; font-size:18px;">●</span> 停用站：站點未啟用
</div>
                """,
                unsafe_allow_html=True,
            )
            return

        st.sidebar.markdown(
            """
<div style="line-height: 1.9;">
<span style="color:red; font-size:18px;">●</span> 高風險：15 分鐘後借/還任一風險高<br>
<span style="color:orange; font-size:18px;">●</span> 中風險：15 分鐘後借/還風險中等<br>
<span style="color:green; font-size:18px;">●</span> 正常：15 分鐘後整體風險低<br>
<span style="color:gray; font-size:18px;">●</span> 停用站：站點未啟用
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    # 目前風險模式
    if risk_mode == "我要借車":
        st.sidebar.markdown(
            """
<div style="line-height: 1.9;">
<span style="color:red; font-size:18px;">●</span> 高風險：完全沒車可借<br>
<span style="color:orange; font-size:18px;">●</span> 中風險：可借車數 ≤ 3<br>
<span style="color:green; font-size:18px;">●</span> 正常：可借車數 > 3<br>
<span style="color:gray; font-size:18px;">●</span> 停用站：站點未啟用
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    if risk_mode == "我要還車":
        st.sidebar.markdown(
            """
<div style="line-height: 1.9;">
<span style="color:red; font-size:18px;">●</span> 高風險：完全沒位可還<br>
<span style="color:orange; font-size:18px;">●</span> 中風險：可還車位 ≤ 3<br>
<span style="color:green; font-size:18px;">●</span> 正常：可還車位 > 3<br>
<span style="color:gray; font-size:18px;">●</span> 停用站：站點未啟用
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.sidebar.markdown(
        """
<div style="line-height: 1.9;">
<span style="color:red; font-size:18px;">●</span> 高風險：沒車可借，或沒位可還<br>
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


def build_map(df: pd.DataFrame, risk_mode: str) -> folium.Map:
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
        lat = row["latitude"]
        lon = row["longitude"]
        station_name = str(row.get("sna", "未知站點"))

        google_maps_url = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        google_maps_directions_url = (
            f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}&travelmode=walking"
        )

        borrow_prob_text = ""
        return_prob_text = ""

        if "pred_borrow_risk_prob_15m" in row.index:
            borrow_prob = row.get("pred_borrow_risk_prob_15m", 0)
            if pd.notna(borrow_prob):
                borrow_prob_text = f"借車高風險機率(15分後)：{borrow_prob:.1%}<br>"

        if "pred_return_risk_prob_15m" in row.index:
            return_prob = row.get("pred_return_risk_prob_15m", 0)
            if pd.notna(return_prob):
                return_prob_text = f"還車高風險機率(15分後)：{return_prob:.1%}<br>"

        popup_html = f"""
        <b>{row.get('sna', '未知站點')}</b><br>
        城市：{row.get('city', '未知')}<br>
        行政區：{row.get('sarea', '未知')}<br>
        地址：{row.get('ar', '未知')}<br>
        可借車數：{row.get('available_rent_bikes', 'N/A')}<br>
        可還車位：{row.get('available_return_bikes', 'N/A')}<br>
        總車格：{row.get('quantity', 'N/A')}<br>
        評估模式：{risk_mode}<br>
        風險等級：{risk_label}<br>
        更新時間：{row.get('info_time', row.get('snapshot_time', 'N/A'))}<br>
        {borrow_prob_text}
        {return_prob_text}<br>
        <a href="{google_maps_url}" target="_blank"> 在 Google Maps 開啟位置</a><br>
        <a href="{google_maps_directions_url}" target="_blank"> 導航到這裡</a>
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

    st.sidebar.header("篩選條件")

    display_mode = st.sidebar.radio(
        "顯示模式",
        ["目前風險", "15分鐘後預測風險"],
        index=0,
    )

    risk_mode = st.sidebar.radio(
        "使用情境",
        ["我要借車", "我要還車", "一起看"],
        index=0,
    )

    try:
        df = load_all_data()

        if display_mode == "目前風險":
            df["risk_label"] = df.apply(
                lambda row: get_risk_label(row, risk_mode),
                axis=1
            )
        else:
            df = predict_risk_for_current_data(df)
            df["risk_label"] = df.apply(
                lambda row: get_predicted_risk_label(row, risk_mode),
                axis=1
            )
    except Exception as e:
        st.error(f"資料載入失敗：{e}")
        return

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

    render_sidebar_legend(risk_mode, display_mode)

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
    map_obj = build_map(filtered_df, risk_mode)
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