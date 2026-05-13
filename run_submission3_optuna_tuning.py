import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

from run_submission3_experiments import feature_engineering, metrics, select_threshold


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent


FEATURES = [
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
CAT_FEATURES = ["subscription_type", "use_frequency", "last_use_recency"]
CAT_INDICES = [FEATURES.index(col) for col in CAT_FEATURES]


def build_params(trial: optuna.Trial) -> dict:
    return {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
        "iterations": trial.suggest_int("iterations", 450, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.055, log=True),
        "depth": trial.suggest_int("depth", 3, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 6.0, 30.0),
        "random_strength": trial.suggest_float("random_strength", 0.5, 2.5),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
        "class_weights": [trial.suggest_float("class0_weight", 1.0, 1.8), 1.0],
        "early_stopping_rounds": 70,
    }


def evaluate_cv(df: pd.DataFrame, params: dict, n_splits: int = 3) -> dict:
    X = df[FEATURES].reset_index(drop=True)
    y = df["target"].reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    for train_idx, valid_idx in splitter.split(X, y):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train,
            y_train,
            cat_features=CAT_INDICES,
            eval_set=(X_valid, y_valid),
            verbose=False,
        )
        prob_1 = model.predict_proba(X_valid)[:, 1]
        threshold, _ = select_threshold(y_valid, prob_1, min_recall_0=0.75)
        row = metrics(y_valid, prob_1, threshold)
        rows.append(row)

    metric_names = [
        "Accuracy",
        "ROC-AUC",
        "Precision(0)",
        "Recall(0)",
        "F1(0)",
        "threshold_0",
    ]
    return {name: float(np.mean([row[name] for row in rows])) for name in metric_names}


def main():
    raw = pd.read_csv(ROOT / "mock_data_3.csv")
    df = feature_engineering(raw)

    train_val, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["target"]
    )
    train, val = train_test_split(
        train_val,
        test_size=0.25,
        random_state=42,
        stratify=train_val["target"],
    )

    trial_rows = []

    def objective(trial: optuna.Trial) -> float:
        params = build_params(trial)
        cv_result = evaluate_cv(train_val, params, n_splits=3)
        row = {
            "trial": trial.number,
            **trial.params,
            **cv_result,
        }
        trial_rows.append(row)
        print(
            f"trial={trial.number:02d} "
            f"cv_f1_0={cv_result['F1(0)']:.4f} "
            f"cv_recall_0={cv_result['Recall(0)']:.4f} "
            f"cv_precision_0={cv_result['Precision(0)']:.4f}"
        )
        return cv_result["F1(0)"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=24, show_progress_bar=False)

    trials_df = pd.DataFrame(trial_rows).sort_values("F1(0)", ascending=False)
    trials_df.to_csv(ROOT / "submission3_optuna_trials.csv", index=False)

    best_params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "allow_writing_files": False,
        "verbose": False,
        "iterations": study.best_params["iterations"],
        "learning_rate": study.best_params["learning_rate"],
        "depth": study.best_params["depth"],
        "l2_leaf_reg": study.best_params["l2_leaf_reg"],
        "random_strength": study.best_params["random_strength"],
        "bagging_temperature": study.best_params["bagging_temperature"],
        "class_weights": [study.best_params["class0_weight"], 1.0],
        "early_stopping_rounds": 80,
    }

    final_model = CatBoostClassifier(**best_params)
    final_model.fit(
        train[FEATURES],
        train["target"],
        cat_features=CAT_INDICES,
        eval_set=(val[FEATURES], val["target"]),
        verbose=False,
    )

    val_prob_1 = final_model.predict_proba(val[FEATURES])[:, 1]
    threshold, threshold_table = select_threshold(val["target"], val_prob_1, min_recall_0=0.75)
    test_prob_1 = final_model.predict_proba(test[FEATURES])[:, 1]
    test_result = metrics(test["target"], test_prob_1, threshold)

    final_summary = {
        "best_cv": trials_df.iloc[0].to_dict(),
        "best_params": best_params,
        "test_result": test_result,
    }
    (ROOT / "submission3_optuna_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    threshold_table.to_csv(ROOT / "submission3_optuna_thresholds.csv", index=False)

    print("\nBEST CV TRIAL")
    print(trials_df.head(5).round(5).to_string(index=False))
    print("\nFINAL TEST")
    print(json.dumps(test_result, ensure_ascii=False, indent=2))
    print("\nBEST PARAMS")
    print(json.dumps(best_params, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
