# 這是訓練基線模型的程式
# 讀取 make_training_data.py 產生的訓練資料
# 用隨機森林訓練兩個模型：借車風險模型、還車風險模型
# 評估模型表現（classification report, confusion matrix）
# 存下訓練好的模型（可選）
# 模型訓練的特徵可以先用比較簡單的，像是：city, sarea, latitude, longitude, quantity, available_rent_bikes, available_return_bikes, snapshot_hour, snapshot_minute, snapshot_weekday
# 之後可以再慢慢加入更多特徵，或是換更複雜的模型，來看看能不能提升預測表現

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


INPUT_FILE = Path("data/collector/youbike_training_data.csv")


def train_and_evaluate(df: pd.DataFrame, target_col: str, model_name: str):
    print(f"\n===== {model_name} =====")

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

    existing_feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[existing_feature_cols].copy()
    y = df[target_col].copy()

    categorical_cols = [c for c in ["city", "sarea"] if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"訓練資料筆數：{len(X_train)}")
    print(f"測試資料筆數：{len(X_test)}")

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    return clf


def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    # 只保留啟用站點
    if "act" in df.columns:
        df["act"] = df["act"].astype(str)
        df = df[df["act"] == "1"].copy()

    # 目標欄位存在才訓練
    targets = [
        ("label_borrow_risk_15m", "借車風險模型"),
        ("label_return_risk_15m", "還車風險模型"),
    ]

    for target_col, model_name in targets:
        if target_col not in df.columns:
            print(f"{target_col} 不存在，跳過。")
            continue

        train_and_evaluate(df, target_col, model_name)


if __name__ == "__main__":
    main()