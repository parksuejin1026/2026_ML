import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "submission3_assets"
ASSET_DIR.mkdir(exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    use_freq_map = {"rare": 1, "monthly": 2, "weekly": 3, "frequent": 4}
    recency_map = {">30d": 1, "7-30d": 2, "1-7d": 3, "<1d": 4}

    df["use_frequency_score"] = df["use_frequency"].map(use_freq_map)
    df["last_use_recency_score"] = df["last_use_recency"].map(recency_map)
    df["effective_cost"] = (df["monthly_cost"] - df["discount_amount"]).clip(lower=0)
    df["value_gap"] = (
        df["use_frequency_score"]
        + df["last_use_recency_score"]
        + df["perceived_necessity"]
        - df["cost_burden"]
    )
    df["rebuy_satisfaction_gap"] = df["would_rebuy"] - df["cost_burden"]
    df["cost_to_necessity_ratio"] = df["effective_cost"] / (
        df["perceived_necessity"] + 1
    )
    df["log_effective_cost"] = np.log1p(df["effective_cost"])
    df["has_churn_signal"] = (
        (df["use_frequency_score"] <= 2)
        & (df["last_use_recency_score"] <= 2)
        & (df["cost_burden"] >= 4)
    ).astype(int)
    df["necessity_x_recency"] = (
        df["perceived_necessity"] * df["last_use_recency_score"]
    )
    df["frequency_x_rebuy"] = df["use_frequency_score"] * df["would_rebuy"]
    df["cost_burden_x_replacement"] = (
        df["cost_burden"] * df["replacement_available"]
    )
    df["is_zero_cost"] = (df["effective_cost"] == 0).astype(int)
    df["is_high_cost"] = (
        df["effective_cost"] > df["effective_cost"].quantile(0.75)
    ).astype(int)
    df["is_deferred"] = (
        (df["billing_cycle"] == 1) & (df["remaining_months"] > 3)
    ).astype(int)

    return df


def metrics(y_true, prob_1, threshold_0: float) -> dict:
    prob_0 = 1 - prob_1
    pred = np.where(prob_0 >= threshold_0, 0, 1)
    return {
        "threshold_0": threshold_0,
        "Accuracy": accuracy_score(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, prob_1),
        "Precision(0)": precision_score(y_true, pred, pos_label=0, zero_division=0),
        "Recall(0)": recall_score(y_true, pred, pos_label=0, zero_division=0),
        "F1(0)": f1_score(y_true, pred, pos_label=0, zero_division=0),
        "Precision(1)": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "Recall(1)": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "F1(1)": f1_score(y_true, pred, pos_label=1, zero_division=0),
        "cm": confusion_matrix(y_true, pred).tolist(),
    }


def select_threshold(y_true, prob_1, min_recall_0=0.75) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.arange(0.25, 0.76, 0.01):
        row = metrics(y_true, prob_1, float(round(threshold, 2)))
        rows.append(row)

    table = pd.DataFrame(rows)
    candidates = table[table["Recall(0)"] >= min_recall_0].copy()
    if candidates.empty:
        best = table.sort_values(["F1(0)", "Recall(0)"], ascending=False).iloc[0]
    else:
        best = candidates.sort_values(["F1(0)", "Precision(0)"], ascending=False).iloc[0]
    return float(best["threshold_0"]), table


def train_onehot_smote(
    name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    features,
    cat_features,
    params,
    fixed_threshold=None,
):
    num_features = [f for f in features if f not in cat_features]
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ]
    )
    X_train_pre = preprocessor.fit_transform(X_train[features])
    X_val_pre = preprocessor.transform(X_val[features])
    X_test_pre = preprocessor.transform(X_test[features])

    smote = SMOTE(random_state=42)
    X_train_fit, y_train_fit = smote.fit_resample(X_train_pre, y_train)

    model = CatBoostClassifier(**params)
    model.fit(X_train_fit, y_train_fit, eval_set=(X_val_pre, y_val), verbose=False)

    val_prob_1 = model.predict_proba(X_val_pre)[:, 1]
    test_prob_1 = model.predict_proba(X_test_pre)[:, 1]
    threshold, threshold_table = select_threshold(y_val, val_prob_1)
    if fixed_threshold is not None:
        threshold = fixed_threshold

    result = metrics(y_test, test_prob_1, threshold)
    result["experiment"] = name
    result["model"] = model
    result["threshold_table"] = threshold_table
    result["feature_names"] = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(cat_features)
    ) + num_features
    result["feature_importance"] = model.get_feature_importance().tolist()
    return result


def train_native(
    name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    features,
    cat_features,
    params,
):
    model = CatBoostClassifier(**params)
    cat_idx = [features.index(c) for c in cat_features]
    model.fit(
        X_train[features],
        y_train,
        cat_features=cat_idx,
        eval_set=(X_val[features], y_val),
        verbose=False,
    )
    val_prob_1 = model.predict_proba(X_val[features])[:, 1]
    test_prob_1 = model.predict_proba(X_test[features])[:, 1]
    threshold, threshold_table = select_threshold(y_val, val_prob_1)

    result = metrics(y_test, test_prob_1, threshold)
    result["experiment"] = name
    result["model"] = model
    result["threshold_table"] = threshold_table
    result["feature_names"] = features
    result["feature_importance"] = model.get_feature_importance().tolist()
    return result


def save_plots(results: list[dict], best: dict):
    summary = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in {"model", "threshold_table", "feature_names", "feature_importance", "cm"}}
            for r in results
        ]
    )
    summary.to_csv(ROOT / "submission3_experiment_results.csv", index=False)

    plot_cols = ["Accuracy", "ROC-AUC", "Precision(0)", "Recall(0)", "F1(0)"]
    melted = summary.melt(
        id_vars=["experiment"], value_vars=plot_cols, var_name="metric", value_name="score"
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(data=melted, x="metric", y="score", hue="experiment")
    plt.ylim(0, 1)
    plt.title("CatBoost 반복 실험 성능 비교")
    plt.xlabel("")
    plt.ylabel("score")
    plt.legend(loc="lower right", fontsize=8)
    for container in plt.gca().containers:
        plt.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "catboost_iteration_comparison.png", dpi=180)
    plt.close()

    threshold_table = best["threshold_table"].copy()
    plt.figure(figsize=(9, 5))
    for col, color in [
        ("Recall(0)", "#e74c3c"),
        ("Precision(0)", "#2f80ed"),
        ("F1(0)", "#27ae60"),
        ("Accuracy", "#6c5ce7"),
    ]:
        plt.plot(
            threshold_table["threshold_0"],
            threshold_table[col],
            marker="o",
            markersize=3,
            linewidth=2,
            label=col,
            color=color,
        )
    plt.axvline(best["threshold_0"], color="#111111", linestyle="--", linewidth=1.5)
    plt.title(f"Best 모델 Threshold 탐색: {best['experiment']}")
    plt.xlabel("class 0 probability threshold")
    plt.ylabel("score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "threshold_search.png", dpi=180)
    plt.close()

    cm = np.array(best["cm"])
    plt.figure(figsize=(5.8, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["해지(0)", "유지(1)"],
        yticklabels=["해지(0)", "유지(1)"],
    )
    plt.title(f"혼동행렬: {best['experiment']}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "best_confusion_matrix.png", dpi=180)
    plt.close()

    fi = pd.DataFrame(
        {
            "Feature": best["feature_names"],
            "Importance": best["feature_importance"],
        }
    ).sort_values("Importance", ascending=False).head(12)
    fi.to_csv(ROOT / "submission3_feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=fi, x="Importance", y="Feature", palette="viridis")
    plt.title(f"Feature Importance Top 12: {best['experiment']}")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(ASSET_DIR / "best_feature_importance.png", dpi=180)
    plt.close()

    return summary


def main():
    raw = pd.read_csv(ROOT / "mock_data_3.csv")
    df = feature_engineering(raw)

    features_basic = [
        "subscription_type",
        "effective_cost",
        "perceived_necessity",
        "cost_burden",
        "would_rebuy",
        "replacement_available",
        "billing_cycle",
        "value_gap",
        "has_churn_signal",
        "necessity_x_recency",
    ]
    features_extended = features_basic + [
        "use_frequency",
        "last_use_recency",
        "remaining_months",
        "rebuy_satisfaction_gap",
        "cost_to_necessity_ratio",
        "log_effective_cost",
        "frequency_x_rebuy",
        "cost_burden_x_replacement",
        "is_zero_cost",
        "is_high_cost",
        "is_deferred",
    ]
    features_selected = [
        "subscription_type",
        "effective_cost",
        "perceived_necessity",
        "cost_burden",
        "would_rebuy",
        "replacement_available",
        "value_gap",
        "has_churn_signal",
        "necessity_x_recency",
        "use_frequency",
        "last_use_recency",
        "remaining_months",
        "rebuy_satisfaction_gap",
        "log_effective_cost",
        "frequency_x_rebuy",
        "cost_burden_x_replacement",
        "is_high_cost",
    ]
    cat_basic = ["subscription_type"]
    cat_extended = ["subscription_type", "use_frequency", "last_use_recency"]

    train_val, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=42,
        stratify=train_val["target"],
    )

    y_train = train["target"]
    y_val = val["target"]
    y_test = test["target"]

    common = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
    }

    results = []
    results.append(
        train_onehot_smote(
            "1_SMOTE_tuned_threshold0.40",
            train,
            y_train,
            val,
            y_val,
            test,
            y_test,
            features_basic,
            cat_basic,
            {
                **common,
                "iterations": 432,
                "learning_rate": 0.018,
                "depth": 5,
            },
            fixed_threshold=0.40,
        )
    )
    results.append(
        train_onehot_smote(
            "2_SMOTE_tuned_val_threshold",
            train,
            y_train,
            val,
            y_val,
            test,
            y_test,
            features_basic,
            cat_basic,
            {
                **common,
                "iterations": 432,
                "learning_rate": 0.018,
                "depth": 5,
                "l2_leaf_reg": 5,
            },
        )
    )
    results.append(
        train_native(
            "3_native_auto_weight",
            train,
            y_train,
            val,
            y_val,
            test,
            y_test,
            features_extended,
            cat_extended,
            {
                **common,
                "iterations": 600,
                "learning_rate": 0.03,
                "depth": 5,
                "l2_leaf_reg": 7,
                "auto_class_weights": "Balanced",
                "early_stopping_rounds": 80,
            },
        )
    )
    results.append(
        train_native(
            "4_native_manual_weight_regularized",
            train,
            y_train,
            val,
            y_val,
            test,
            y_test,
            features_extended,
            cat_extended,
            {
                **common,
                "iterations": 800,
                "learning_rate": 0.025,
                "depth": 4,
                "l2_leaf_reg": 10,
                "random_strength": 1.2,
                "bagging_temperature": 0.5,
                "class_weights": [1.8, 1.0],
                "early_stopping_rounds": 100,
            },
        )
    )
    results.append(
        train_native(
            "5_native_precision_recovery",
            train,
            y_train,
            val,
            y_val,
            test,
            y_test,
            features_extended,
            cat_extended,
            {
                **common,
                "iterations": 700,
                "learning_rate": 0.025,
                "depth": 5,
                "l2_leaf_reg": 12,
                "random_strength": 1.5,
                "bagging_temperature": 0.7,
                "class_weights": [1.5, 1.0],
                "early_stopping_rounds": 100,
            },
        )
    )
    results.append(
        train_native(
            "6_feature_selection",
            train,
            y_train,
            val,
            y_val,
            test,
            y_test,
            features_selected,
            cat_extended,
            {
                **common,
                "iterations": 650,
                "learning_rate": 0.03,
                "depth": 5,
                "l2_leaf_reg": 8,
                "random_strength": 1.2,
                "bagging_temperature": 0.7,
                "auto_class_weights": "Balanced",
                "early_stopping_rounds": 80,
            },
        )
    )
    results.append(
        train_native(
            "7_extra_search_best",
            train,
            y_train,
            val,
            y_val,
            test,
            y_test,
            features_extended,
            cat_extended,
            {
                **common,
                "iterations": 700,
                "learning_rate": 0.025,
                "depth": 4,
                "l2_leaf_reg": 15,
                "random_strength": 1.0,
                "bagging_temperature": 0.5,
                "class_weights": [1.2, 1.0],
                "early_stopping_rounds": 80,
            },
        )
    )

    best = sorted(results, key=lambda r: (r["F1(0)"], r["Recall(0)"]), reverse=True)[0]
    summary = save_plots(results, best)

    serializable = []
    for result in results:
        row = {
            key: value
            for key, value in result.items()
            if key
            not in {"model", "threshold_table", "feature_names", "feature_importance"}
        }
        serializable.append(row)

    (ROOT / "submission3_experiment_results.json").write_text(
        json.dumps({"results": serializable, "best": best["experiment"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(summary.round(4).to_string(index=False))
    print(f"\nBest: {best['experiment']}")
    print(json.dumps({k: best[k] for k in serializable[0] if k != "experiment"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
