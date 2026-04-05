# 這是訓練基線模型的程式
# 讀取 make_training_data.py 產生的訓練資料
# 用隨機森林訓練兩個模型：借車風險模型、還車風險模型
# 評估模型表現（classification report, confusion matrix）
# 存下訓練好的模型（可選）
# 模型訓練的特徵可以先用比較簡單的，像是：city, sarea, latitude, longitude, quantity, available_rent_bikes, available_return_bikes, snapshot_hour, snapshot_minute, snapshot_weekday
# 之後可以再慢慢加入更多特徵，或是換更複雜的模型，來看看能不能提升預測表現

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

INPUT_FILE = Path("data/collector/youbike_training_data.csv")
MODEL_DIR = Path("models")
BORROW_MODEL_FILE = MODEL_DIR / "borrow_risk_model.joblib"
RETURN_MODEL_FILE = MODEL_DIR / "return_risk_model.joblib"

FEATURE_COLS = [
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


def build_pipeline(categorical_cols, numeric_cols):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return clf


def prepare_training_df() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    if "act" in df.columns:
        df["act"] = df["act"].astype(str)
        df = df[df["act"] == "1"].copy()

    numeric_cols = [
        "latitude",
        "longitude",
        "quantity",
        "available_rent_bikes",
        "available_return_bikes",
        "snapshot_hour",
        "snapshot_minute",
        "snapshot_weekday",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def train_and_save_model(df: pd.DataFrame, target_col: str, output_file: Path):
    X = df[FEATURE_COLS].copy()
    y = df[target_col].copy()

    categorical_cols = [c for c in ["city", "sarea"] if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    clf = build_pipeline(categorical_cols, numeric_cols)
    clf.fit(X, y)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_file)

    print(f"已儲存模型：{output_file}")


def main():
    df = prepare_training_df()

    train_and_save_model(
        df=df,
        target_col="label_borrow_risk_15m",
        output_file=BORROW_MODEL_FILE,
    )

    train_and_save_model(
        df=df,
        target_col="label_return_risk_15m",
        output_file=RETURN_MODEL_FILE,
    )

    print("模型存檔完成。")


if __name__ == "__main__":
    main()