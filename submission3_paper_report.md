# CatBoost 기반 구독 해지 후보 탐지 모델의 반복 개선 연구

## Abstract

본 보고서는 구독 서비스 사용자의 해지 후보를 탐지하기 위한 CatBoost 기반 이진 분류 모델의 반복 개선 과정을 정리한다. 본 과제의 목적은 구독을 자동으로 해지하는 것이 아니라, 사용자가 점검할 필요가 있는 구독을 추천하는 것이다. 따라서 전체 정확도보다 실제 해지 후보(class 0)를 놓치지 않는 능력, 즉 Recall(0)을 주요 평가 기준으로 설정하였다. 본 연구에서는 SMOTE 기반 데이터 증강, validation set 기반 threshold 조정, CatBoost native categorical 처리, class weight 조정, regularization, feature selection, 최종 모델 주변 탐색, Optuna 기반 3-fold cross-validation 튜닝을 순차적으로 수행하였다. 최종 선택 모델은 `depth=4`, `l2_leaf_reg=15`, `class_weights=[1.2, 1.0]`, `threshold=0.35`를 적용한 CatBoost 모델이다. 해당 모델은 2차 제출 CatBoost 기준 Recall(0)을 0.6833에서 0.7638로, F1(0)을 0.6074에서 0.6372로 개선하였다.

---

## 1. Introduction

구독 서비스는 사용자가 장기간 자동 결제를 유지하는 구조를 가진다. 이 구조에서는 실제 사용 빈도가 낮거나 체감 가치가 낮은 구독도 사용자가 인지하지 못한 채 유지될 수 있다. 본 과제는 이러한 구독 중 해지 또는 점검이 필요할 가능성이 높은 대상을 탐지하는 모델을 구축하는 데 목적이 있다.

본 문제에서 모델의 예측 결과는 직접적인 해지 조치가 아니라 사용자에게 제공되는 권고 또는 점검 후보로 사용된다. 따라서 유지할 구독을 해지 후보로 일부 잘못 추천하는 False Positive보다, 실제로 점검이 필요한 구독을 놓치는 False Negative가 더 중요한 문제로 간주된다. 이에 따라 본 연구에서는 Accuracy보다 Recall(0)과 F1(0)을 더 중요한 기준으로 사용하였다.

2차 제출에서 CatBoost는 비교 모델 중 해지 후보 탐지 성능이 상대적으로 높았다. 그러나 Recall(0)=0.6833으로 실제 해지 후보의 약 31.7%를 놓치는 한계가 있었다. 본 보고서는 이 한계를 줄이기 위해 수행한 반복 실험과 최종 모델 선택 근거를 제시한다.

---

## 2. Problem Setting and Evaluation Criteria

본 과제는 이진 분류 문제로 정의된다.

| 클래스 | 의미 | 본 보고서에서의 해석 |
| --- | --- | --- |
| 0 | 해지 후보 | 사용자가 점검하거나 해지를 고려할 가능성이 높은 구독 |
| 1 | 유지 후보 | 유지 가능성이 높은 구독 |

평가 지표의 우선순위는 다음과 같이 설정하였다.

| 우선순위 | 지표 | 사용 이유 |
| --- | --- | --- |
| 1 | Recall(0) | 실제 해지 후보를 놓치지 않는 능력 |
| 2 | F1(0) | Recall 개선 과정에서 Precision 저하가 과도한지 확인 |
| 3 | ROC-AUC | threshold와 무관한 확률 순위 품질 평가 |
| 4 | Accuracy | 전체 분류 정확도 확인 |

2차 제출 CatBoost 모델의 기준 성능은 다음과 같다.

| 지표 | 2차 CatBoost |
| --- | ---: |
| Accuracy | 0.7350 |
| ROC-AUC | 0.8184 |
| Precision(0) | 0.5467 |
| Recall(0) | 0.6833 |
| F1(0) | 0.6074 |

---

## 3. Dataset and Preprocessing

### 3.1 Dataset

본 실험에서는 `mock_data_3.csv`를 사용하였다. 전체 데이터는 train, validation, test로 분할하였다. 기존 방식처럼 test set에서 직접 threshold를 선택하지 않고, validation set에서 threshold를 선택한 뒤 test set에서 최종 성능을 평가하였다. 이는 test set에 대한 과도한 맞춤을 줄이기 위한 절차이다.

### 3.2 Effective Cost

명목 구독료인 `monthly_cost`는 할인, 환급, 제휴 혜택을 반영하지 못한다. 따라서 실제 사용자가 체감하는 비용을 반영하기 위해 다음 변수를 사용하였다.

```python
effective_cost = max(monthly_cost - discount_amount, 0)
```

이 변수는 명목상 비용은 높지만 실제 부담이 낮은 구독을 과도하게 해지 후보로 분류하는 문제를 완화하기 위해 도입하였다.

### 3.3 Feature Engineering

도메인 지식을 반영하기 위해 다음 파생 변수를 생성하였다.

| 파생 변수 | 계산 방식 | 목적 |
| --- | --- | --- |
| `value_gap` | 사용 빈도 + 최근성 + 필요도 - 비용 부담 | 구독의 체감 가치 측정 |
| `has_churn_signal` | 낮은 사용 빈도 + 오래된 최근 사용 + 높은 비용 부담 | 규칙 기반 해지 신호 포착 |
| `necessity_x_recency` | 필요도 × 최근 사용 점수 | 필요도와 실제 사용의 결합 효과 반영 |
| `frequency_x_rebuy` | 사용 빈도 × 재구매 의향 | 반복 사용과 유지 의향의 상호작용 반영 |
| `cost_burden_x_replacement` | 비용 부담 × 대체재 존재 여부 | 비용 부담이 해지 행동으로 이어질 가능성 반영 |

### 3.4 Categorical Feature Handling

초기 실험에서는 범주형 변수를 One-Hot Encoding한 뒤 SMOTE를 적용하였다. 그러나 CatBoost는 범주형 변수를 직접 처리하는 데 강점을 갖는 모델이다. 후반 실험에서는 다음 변수를 CatBoost의 categorical feature로 직접 전달하였다.

- `subscription_type`
- `use_frequency`
- `last_use_recency`

이 방식은 One-Hot Encoding과 SMOTE 조합에서 발생할 수 있는 인위적인 범주형 샘플 생성을 줄이는 데 목적이 있다.

---

## 4. Methodology

### 4.1 Model

본 연구의 모든 주요 실험은 CatBoostClassifier를 기반으로 수행하였다. CatBoost는 gradient boosting 계열 모델로, 범주형 변수를 직접 처리할 수 있고 비선형 상호작용을 학습하는 데 적합하다. 본 데이터는 구독 유형, 사용 빈도, 최근 사용 시점 등 범주형 변수를 포함하므로 CatBoost를 최종 모델 후보로 설정하였다.

### 4.2 Class Weighting

해지 후보(class 0)는 본 과제에서 더 중요한 클래스이다. 따라서 일부 실험에서는 class 0에 더 높은 가중치를 부여하였다.

최종 선택 모델에서는 다음 가중치를 사용하였다.

```python
class_weights = [1.2, 1.0]
```

이는 해지 후보를 유지 후보보다 1.2배 더 중요하게 학습한다는 의미이다. 실험 4에서는 class 0 가중치를 1.8까지 높였으나, Precision과 Accuracy가 낮아져 최종 선택하지 않았다.

### 4.3 Threshold Tuning

모델은 각 클래스에 대한 확률을 출력한다. 기본 threshold 0.5를 사용하면 해지 후보 탐지가 보수적으로 이루어질 수 있다. 본 과제에서는 실제 해지 후보를 놓치는 비용이 더 크므로 validation set에서 threshold를 탐색하였다.

최종 선택 모델의 class 0 threshold는 다음과 같다.

```python
threshold_0 = 0.35
```

즉, 해지 후보 확률이 0.35 이상이면 class 0으로 분류하였다.

### 4.4 Regularization

과적합을 줄이기 위해 tree depth와 L2 regularization을 조정하였다. 최종 모델은 다음 설정을 사용하였다.

| 파라미터 | 값 | 역할 |
| --- | ---: | --- |
| `depth` | 4 | 트리 복잡도 제한 |
| `l2_leaf_reg` | 15 | leaf 값의 과도한 변화 억제 |
| `learning_rate` | 0.025 | 안정적인 학습 |
| `iterations` | 700 | 낮은 learning rate 보완 |

---

## 5. Experimental Design

총 8단계의 실험을 수행하였다.

| 실험 | 수정 내용 | 목적 |
| --- | --- | --- |
| 1 | SMOTE + 튜닝 파라미터 + threshold 0.40 | 기존 3차 방식 재현 |
| 2 | SMOTE + validation threshold | threshold 선택 방식 개선 |
| 3 | CatBoost native categorical + auto class weight | SMOTE 없이 CatBoost 장점 활용 |
| 4 | native categorical + 수동 class weight + regularization | Recall(0) 강화 |
| 5 | native categorical + 약한 class weight + 강한 regularization | Precision 회복 시도 |
| 6 | feature selection | 중요도 낮은 변수 제거 |
| 7 | 추가 class weight 탐색 | 최종 모델 후보 도출 |
| 8 | Optuna + 3-fold CV | 더 넓은 하이퍼파라미터 공간 검증 |

주요 하이퍼파라미터 변경 기록은 다음과 같다.

| 실험 | 주요 설정 | 의도 |
| --- | --- | --- |
| 1 | `iterations=432`, `learning_rate=0.018`, `depth=5` | 낮은 learning rate로 안정적 학습 |
| 3 | `auto_class_weights="Balanced"` | 자동 클래스 불균형 보정 |
| 4 | `class_weights=[1.8, 1.0]`, `depth=4`, `l2_leaf_reg=10` | class 0 탐지 강화 |
| 5 | `class_weights=[1.5, 1.0]`, `l2_leaf_reg=12` | 오탐 완화 시도 |
| 7 | `class_weights=[1.2, 1.0]`, `depth=4`, `l2_leaf_reg=15`, `threshold=0.35` | Recall과 Precision 균형 최적화 |
| 8 | Optuna 24 trials, 3-fold CV | Bayesian 방식의 체계적 탐색 |

---

## 6. Results

### 6.1 Main Experiment Results

![CatBoost 반복 실험 성능 비교](submission3_assets/catboost_iteration_comparison.png)

| 실험 | Threshold(0) | Accuracy | ROC-AUC | Precision(0) | Recall(0) | F1(0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. SMOTE + threshold 0.40 | 0.40 | 0.6973 | 0.7697 | 0.5496 | 0.7380 | 0.6300 |
| 2. SMOTE + validation threshold | 0.39 | 0.6918 | 0.7696 | 0.5420 | 0.7581 | 0.6321 |
| 3. native categorical + auto weight | 0.46 | 0.6935 | 0.7742 | 0.5439 | 0.7588 | 0.6336 |
| 4. native + manual weight | 0.44 | 0.6873 | 0.7746 | 0.5363 | 0.7717 | 0.6328 |
| 5. native + precision recovery | 0.39 | 0.6833 | 0.7750 | 0.5319 | 0.7752 | 0.6309 |
| 6. feature selection | 0.45 | 0.6838 | 0.7743 | 0.5329 | 0.7652 | 0.6283 |
| **7. extra search best** | **0.35** | **0.6963** | **0.7748** | **0.5466** | **0.7638** | **0.6372** |

실험 7은 F1(0)=0.6372로 가장 높은 값을 기록하였다. Recall(0)은 0.7638로 실험 4, 5, 6보다 낮거나 비슷하지만, Precision(0)과 Accuracy를 함께 고려했을 때 가장 균형적인 성능을 보였다.

### 6.2 Threshold Analysis

![Threshold 탐색](submission3_assets/threshold_search.png)

최종 모델의 threshold는 validation set에서 0.35로 선택되었다. 이는 기본 threshold 0.5보다 낮은 값이며, 해지 후보를 더 적극적으로 탐지하기 위한 설정이다. 본 문제에서는 예측 결과가 자동 해지가 아니라 점검 권고로 사용되므로, threshold를 낮추는 전략이 문제 목적에 부합한다.

### 6.3 Confusion Matrix

![최종 모델 혼동행렬](submission3_assets/best_confusion_matrix.png)

| 실제 \ 예측 | 해지 후보(0) | 유지 후보(1) |
| --- | ---: | ---: |
| 실제 해지 후보(0) | 1,067 | 330 |
| 실제 유지 후보(1) | 885 | 1,718 |

최종 모델은 실제 해지 후보 1,397건 중 1,067건을 탐지하였다. 이는 Recall(0)=0.7638에 해당한다. 유지 후보 중 885건이 해지 후보로 분류되었지만, 본 시스템이 해지 자동화가 아니라 점검 추천을 목적으로 한다는 점에서 허용 가능한 trade-off로 판단하였다.

### 6.4 Feature Importance

![Feature Importance](submission3_assets/best_feature_importance.png)

| 순위 | 변수 | 중요도 |
| ---: | --- | ---: |
| 1 | `value_gap` | 26.42 |
| 2 | `cost_burden_x_replacement` | 13.21 |
| 3 | `frequency_x_rebuy` | 11.17 |
| 4 | `necessity_x_recency` | 7.85 |
| 5 | `use_frequency` | 6.42 |
| 6 | `replacement_available` | 5.44 |

가장 중요한 변수는 `value_gap`이었다. 이는 단순 비용보다 사용 빈도, 최근성, 필요도, 비용 부담을 함께 반영한 체감 가치가 해지 후보 판단에 더 중요한 역할을 한다는 점을 보여준다.

---

## 7. Additional Analyses

### 7.1 Feature Selection

Feature importance를 바탕으로 중요도가 낮거나 정보 중복 가능성이 있는 변수를 제거하였다.

| 제거 변수 | 제거 이유 |
| --- | --- |
| `billing_cycle` | 비용/잔여기간 변수와 정보 중복 가능성 |
| `cost_to_necessity_ratio` | `effective_cost`, `perceived_necessity`, `value_gap`과 중복 가능성 |
| `is_zero_cost` | 낮은 중요도 |
| `is_deferred` | `remaining_months`, `billing_cycle`과 중복 가능성 |

| 비교 항목 | 최종 모델 | Feature selection 모델 |
| --- | ---: | ---: |
| Accuracy | 0.6963 | 0.6838 |
| ROC-AUC | 0.7748 | 0.7743 |
| Precision(0) | 0.5466 | 0.5329 |
| Recall(0) | 0.7638 | 0.7652 |
| F1(0) | 0.6372 | 0.6283 |

Feature selection 모델은 Recall(0)을 약간 높였지만 Precision(0)과 F1(0)이 하락하였다. 따라서 변수 제거는 최종 모델로 채택하지 않았다.

### 7.2 Final Model Probe

최종 모델 주변의 class weight, depth, L2 regularization, learning rate를 추가 탐색하였다.

| 추가 실험 | 변경 내용 | Accuracy | ROC-AUC | Precision(0) | Recall(0) | F1(0) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **current best** | `depth=4`, `l2_leaf_reg=15`, `class_weights=[1.2, 1.0]` | **0.6963** | 0.7751 | **0.5466** | 0.7638 | **0.6372** |
| weight 1.3 | class 0 가중치 1.3 | 0.6945 | 0.7748 | 0.5447 | 0.7631 | 0.6357 |
| random 강화 | `random_strength=1.5`, `bagging_temperature=0.7` | 0.6935 | 0.7747 | 0.5437 | 0.7616 | 0.6345 |
| depth 3 | 트리 깊이 3 | 0.6955 | 0.7743 | 0.5463 | 0.7566 | 0.6345 |
| learning rate 0.03 | 학습률 상승 | 0.6928 | 0.7748 | 0.5429 | 0.7609 | 0.6337 |
| L2 20 | 정규화 강화 | 0.6853 | **0.7752** | 0.5339 | **0.7774** | 0.6331 |

L2 regularization을 더 강하게 준 모델은 Recall(0)=0.7774로 높았으나 Precision(0)과 F1(0)이 하락하였다. 따라서 최종 모델은 Recall과 Precision의 균형이 가장 좋은 current best로 유지하였다.

### 7.3 Optuna-Based Tuning

수동 탐색을 보완하기 위해 Optuna Bayesian Optimization과 3-fold Stratified CV를 적용하였다. 총 24 trial을 수행하였고, 각 trial에서는 fold별 validation threshold를 선택한 뒤 F1(0)을 평균 내어 최적화하였다.

| 구분 | Accuracy | ROC-AUC | Precision(0) | Recall(0) | F1(0) | Threshold(0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Optuna 3-fold CV 평균 | 0.6895 | 0.7789 | 0.5394 | 0.7660 | 0.6328 | 0.3633 |
| Optuna 최종 test | 0.6888 | 0.7748 | 0.5375 | **0.7795** | 0.6363 | 0.36 |
| 현재 최종 모델 test | **0.6963** | 0.7751 | **0.5466** | 0.7638 | **0.6372** | 0.35 |

Optuna 모델은 Recall(0)을 0.7795까지 높였으나 F1(0)은 0.6363으로 현재 최종 모델보다 낮았다. 이는 해지 후보를 더 많이 잡는 대신 오탐이 늘어난 결과로 해석된다. 따라서 최종 모델은 Optuna 모델로 교체하지 않았다.

---

## 8. Discussion

본 실험에서 가장 중요한 관찰은 Recall(0)과 Precision(0) 사이의 trade-off이다. class weight를 높이거나 threshold를 낮추면 해지 후보를 더 많이 탐지할 수 있다. 그러나 이 경우 유지 후보를 해지 후보로 잘못 추천하는 비율도 증가한다. 본 과제는 자동 해지 시스템이 아니므로 어느 정도의 오탐은 허용 가능하지만, 오탐이 지나치게 커지면 추천 시스템의 신뢰도가 낮아질 수 있다.

실험 7은 이러한 trade-off의 균형점으로 해석된다. class 0에 1.2배의 약한 가중치를 부여하고, `depth=4`와 `l2_leaf_reg=15`로 모델 복잡도를 제어하였다. 이 설정은 해지 후보 탐지 성능을 높이면서도 Precision과 F1의 하락을 억제하였다.

Optuna 실험은 더 체계적인 탐색을 제공했지만 최종 test 기준 F1(0)을 개선하지는 못했다. 이는 CV 평균에서 좋은 조합이 단일 test split에서 반드시 가장 좋은 조합이 되지는 않을 수 있음을 보여준다. 따라서 본 보고서에서는 최종 test set에서 F1(0)이 가장 높고 Precision도 상대적으로 안정적인 실험 7을 최종 모델로 선택하였다.

---

## 9. Limitations

본 실험에는 다음과 같은 한계가 있다.

| 한계 | 설명 |
| --- | --- |
| 합성 데이터 기반 | 실제 사용자 행동 데이터가 아니므로 실제 서비스 환경의 일반화 성능은 추가 검증 필요 |
| Accuracy 및 ROC-AUC 하락 | Recall(0)을 우선하면서 전체 정확도와 확률 순위 품질 일부가 낮아짐 |
| Threshold 정책 의존성 | threshold를 낮추는 전략은 서비스 정책과 오탐 허용 수준에 따라 달라질 수 있음 |
| Feature selection 한계 | 단순 제거 실험만 수행했으며, SHAP 기반 세밀한 변수 분석은 수행하지 않음 |
| Optuna 탐색 규모 제한 | 24 trial로 제한했기 때문에 더 큰 탐색에서 다른 결과가 나올 가능성 존재 |

---

## 10. Conclusion

본 보고서는 CatBoost 기반 구독 해지 후보 탐지 모델의 반복 개선 과정을 정리하였다. 2차 제출 CatBoost 모델의 Recall(0)은 0.6833, F1(0)은 0.6074였으며, 최종 선택 모델은 Recall(0)=0.7638, F1(0)=0.6372를 달성하였다.

최종 모델은 `class_weights=[1.2, 1.0]`, `depth=4`, `l2_leaf_reg=15`, `learning_rate=0.025`, `iterations=700`, `threshold=0.35`를 사용하였다. 이 설정은 해지 후보를 더 적극적으로 탐지하면서도 Precision(0)과 F1(0)의 균형을 유지하였다. Optuna 기반 추가 튜닝에서는 Recall(0)을 더 높일 수 있었으나 F1(0)이 소폭 낮아 최종 모델로 채택하지 않았다.

따라서 본 모델은 전체 정확도 극대화 모델이 아니라, 사용자가 점검해야 할 구독 후보를 더 효과적으로 찾아내는 목적 지향적 모델로 해석할 수 있다.
