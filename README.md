# 📊 구독 해지 예측 프로젝트 (Subscription Churn Prediction)

> 사용자의 이용 패턴을 기반으로, 다음 달 구독 유지 여부를 예측하는 머신러닝 프로젝트

---

## 🚀 프로젝트 개요

현대인은 다양한 구독 서비스를 이용하지만, 실제 사용량과 관계없이 자동 결제가 이루어지면서  
불필요한 지출(구독 피로감)이 발생합니다.

본 프로젝트는 사용자 데이터를 기반으로  
다음 달 구독을 유지할지(1) / 해지할지(0)를 예측합니다.

---

## 🎯 목표

- 사용자 행동 데이터를 기반으로 구독 유지 여부 예측
- 가성비 + 불만 요소 반영
- 합리적인 소비 의사결정 지원

---

## 📂 데이터 설명

- 데이터 크기: 5,000개
- 데이터 출처: AI 생성 데이터

### 주요 변수

- Monthly_Fee: 월 구독료
- Access_Days: 접속 일수
- Usage_Time: 이용 시간
- Content_Count: 콘텐츠 소비량
- Customer_Inquiry: 문의 횟수

---

## 🧠 파생 변수

Inquiry_Rate = Customer_Inquiry / (Access_Days + 1)  
Value_Score = Usage_Time / Monthly_Fee  

---

## 🏷️ 타겟

- 0: 해지
- 1: 유지

---

## 🤖 모델

Logistic Regression

---

## 📈 성능

- Accuracy: 75%
- Precision: 72%
- Recall: 68%

---

## ⚠️ 한계

- 선형 모델의 한계
- 복잡한 패턴 반영 어려움

---

## 🔧 개선 방향

- Decision Tree, Random Forest 적용
- Feature Engineering 확장

---

## ⭐ 한 줄 요약

사용자 데이터를 기반으로 구독 해지 여부를 예측하는 모델
