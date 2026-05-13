# 제출 3 요구사항 충족 여부 점검표

## 1. 전체 점검 요약

| 요구사항 | 충족 여부 | 현재 반영 내용 | 확인 위치 |
| --- | --- | --- | --- |
| 반복 수정 기록 | 충족 | CatBoost 모델을 7개 실험으로 나누어 순차 개선 과정을 기록 | `submission3.md` 2장, 5장 |
| 무엇을 바꿨는지 | 충족 | SMOTE, validation threshold, native categorical, class weight, regularization 변경 기록 | `submission3.md` 2장, 4장 |
| 왜 바꿨는지 | 충족 | 각 실험별 목적과 하이퍼파라미터 변경 이유 설명 | `submission3.md` 2장, 4장, 8장 |
| 결과가 어떻게 달라졌는지 | 충족 | 실험별 Accuracy, ROC-AUC, Precision(0), Recall(0), F1(0) 비교표 작성 | `submission3.md` 5장 |
| 하이퍼파라미터 조정 | 충족 | `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `class_weights`, `early_stopping_rounds` 비교 | `submission3.md` 4장 |
| learning rate 언급 | 충족 | 실험별 `learning_rate=0.018`, `0.03`, `0.025` 기록 | `submission3.md` 4장 |
| max depth 언급 | 충족 | `depth=4`, `depth=5` 비교 및 변경 이유 기록 | `submission3.md` 4장 |
| regularization 언급 | 충족 | `l2_leaf_reg`, `random_strength`, `bagging_temperature`, `early_stopping_rounds` 사용 | `submission3.md` 4장, 8장 |
| 데이터 전처리 변경 | 충족 | `effective_cost`, 파생 변수, 범주형 처리 방식 변경 기록 | `submission3.md` 3장 |
| feature scaling | 충족 | SMOTE 실험에서 One-Hot Encoding + `StandardScaler` 사용 | `run_submission3_experiments.py`, `submission3.md` 3장 |
| feature selection | 충족 | 중요도 분석 후 낮은 중요도/중복 가능 변수 제거 실험을 추가했고, 성능 하락으로 미채택한 이유까지 기록 | `submission3.md` 7장 |
| 데이터 확장/증강 | 충족 | SMOTE를 통한 소수 클래스 증강 실험 포함 | `submission3.md` 2장, 5장 |
| 모델 비교표 | 충족 | 7개 CatBoost 실험 비교표 포함 | `submission3.md` 5장 |
| 성능 그래프 | 충족 | 반복 실험 성능 비교, threshold 탐색, 혼동행렬, feature importance 그래프 포함 | `submission3_assets/` |
| 변화 원인 분석 | 충족 | threshold, SMOTE, native categorical, class weight의 영향 분석 | `submission3.md` 8장 |
| CatBoost 모델 기준 | 충족 | 모든 실험이 CatBoost 기반으로 구성됨 | 전체 보고서 |
| PDF 변환 가능성 | 충족 | Markdown 보고서와 PNG 그래프를 사용해 PDF 변환에 적합 | `submission3.md` |

---

## 2. 세부 요구사항별 평가

| 제출 3 안내 문구 | 현재 보고서 대응 | 평가 |
| --- | --- | --- |
| "세 번째 제출의 핵심은 모델을 개선한 과정을 기록하는 것입니다." | 1회 수정이 아니라 7단계 반복 실험으로 구성 | 충족 |
| "무엇을 바꿨고" | 전처리, SMOTE, threshold, native categorical, class weight, regularization을 표로 정리 | 충족 |
| "왜 바꿨고" | 각 변경의 목적을 "해지 후보 Recall 개선", "오탐 완화", "CatBoost 장점 활용"으로 설명 | 충족 |
| "결과가 어떻게 달라졌는지 보여주세요." | 실험별 성능표와 그래프를 통해 변화 수치 제시 | 충족 |
| "하이퍼파라미터 조정 learning rate, max depth, regularization 등" | `learning_rate`, `depth`, `l2_leaf_reg`, `class_weights`, `early_stopping_rounds` 기록 | 충족 |
| "데이터 전처리 변경 feature scaling, feature selection, 데이터 확장" | scaling, SMOTE, 파생 변수 추가, 범주형 처리 변경, feature selection 실험 반영 | 충족 |
| "제출물 작성 모델 비교표" | 7개 CatBoost 실험 비교표 작성 | 충족 |
| "성능 그래프" | 4개 PNG 그래프 포함 | 충족 |
| "변화 원인 분석" | 성능 개선 원인과 Accuracy/ROC-AUC 하락 원인 분석 | 충족 |

---

## 3. 현재 보고서에서 특히 강한 부분

| 강점 | 이유 |
| --- | --- |
| 반복 수정 구조가 명확함 | 실험 1~7로 개선 흐름이 보이기 때문에 "반복 수정 기록" 요구에 잘 맞음 |
| CatBoost 기준이 일관됨 | 다른 모델로 분산되지 않고 CatBoost 내부 개선에 집중함 |
| 결과 해석이 솔직함 | Recall(0)은 개선됐지만 Accuracy/ROC-AUC는 낮아졌다는 trade-off를 명확히 설명함 |
| 시각화 자료가 충분함 | 성능 비교, threshold 탐색, 혼동행렬, feature importance가 모두 포함됨 |
| 최종 모델 선택 근거가 있음 | 단순히 Recall이 가장 높은 실험이 아니라 F1(0)과 오탐 부담을 함께 고려함 |

---

## 4. 보완하면 더 좋아지는 부분

| 보완 항목 | 현재 상태 | 권장 보완 |
| --- | --- | --- |
| Feature selection | 중요도 낮은 변수 제거 후 재학습 실험까지 완료 | 추가 보완 불필요 |
| 성능 기준 설명 | Recall(0) 중심 이유는 설명됨 | "왜 Accuracy보다 Recall(0)을 우선했는지"를 발표 때 강조 |
| PDF 변환 | Markdown과 이미지 준비 완료 | PDF 변환 전 이미지가 깨지지 않는지 미리 확인 |
| 실험 재현성 | 스크립트 존재 | `run_submission3_experiments.py`를 함께 제출하거나 부록으로 언급하면 좋음 |

---

## 5. 최종 판단

| 항목 | 판단 |
| --- | --- |
| 제출 3 요구사항 충족도 | 높음 |
| 추가 수정 필요성 | 현재 기준 필수 보완 없음 |
| 가장 보완 가치가 큰 항목 | PDF 변환 전 이미지 표시 확인 |
| 현재 상태로 제출 가능 여부 | 가능 |

현재 `submission3.md`는 제출 3에서 요구한 핵심 항목을 충족한다. 특히 반복 수정 기록, 하이퍼파라미터 조정, 데이터 전처리 변경, feature selection, 모델 비교표, 성능 그래프, 변화 원인 분석이 모두 포함되어 있다.

Feature selection은 실제로 중요도 낮은 변수 제거 후 재학습까지 수행했으며, 성능이 하락해 최종 모델로 채택하지 않은 이유도 보고서에 기록했다. 따라서 현재 상태로 제출 요구사항 충족도는 충분하다.
