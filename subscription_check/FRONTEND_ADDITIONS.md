# SubCut Frontend Additions

## 추가된 화면

- 하단 탭 쉘: 홈, 해지 추천, 캘린더, 분석, 설정을 전환하는 떠 있는 타원형 탭바를 추가했습니다. 탭은 좌우 스와이프와 탭 버튼 모두로 전환됩니다.
- 해지 추천 탭: 기존 예측 결과 중 `isChurnCandidate`가 true인 구독을 모아 월 절감 가능액, 추천 이유, 신뢰도, 유지/해지 피드백 액션을 보여줍니다.
- 캘린더 탭: 현재 등록된 구독을 월간 캘린더와 결제 예정 목록으로 표시합니다. 날짜를 누르면 해당 날짜의 결제 예정 구독이 바텀시트로 표시됩니다.
- 월/연도 선택: 캘린더 우상단의 `yyyy년 M월` 버튼을 누르면 1900년부터 2900년까지 선택 가능한 iOS 스타일 피커가 열립니다.
- 설정 탭: 알림, 데이터, 보안, 정보 섹션을 추가했습니다. 무료체험 종료 알림은 제거했습니다.
- 분석 탭: 기존 `AnalyticsScreen`을 하단 탭 안에서도 사용할 수 있게 뒤로가기 버튼을 선택적으로 숨기도록 변경했습니다.
- 이용약관 화면: 첫 실행 시 약관 및 데이터 활용 동의를 받습니다. 설정 탭에서도 다시 열 수 있습니다.
- 앱 잠금 화면: PIN 잠금과 Face ID/Touch ID 인증을 로컬에서 동작하도록 추가했습니다.
- 고정 헤더: 모든 탭의 상단 헤더가 상태표시줄까지 이어지고, 본문 영역만 스크롤되도록 변경했습니다.

## 추가된 홈 기능

- 카테고리 필터: 전체, 영상, 음악, 생활, 교육, 게임, 운동, 뉴스 필터를 추가했습니다.
- 정렬: 추천순, 높은 금액순, 낮은 금액순, 이름순 정렬을 추가했습니다.
- 추천순은 해지 후보를 먼저 보여주고, 같은 그룹 안에서는 예측 신뢰도를 기준으로 정렬합니다.

## 현재 프론트 동작 방식

- 기본 실행은 프론트엔드 전용 모드입니다. `BACKEND_ENABLED`의 기본값이 `false`라서 구독 CRUD, 해지 예측, 절감액 조회는 앱 내부 mock 데이터로 동작하고 실제 서버 요청이나 기기 ID 생성은 발생하지 않습니다.
- 실제 백엔드를 사용할 때는 빌드/실행 시 `--dart-define=BACKEND_ENABLED=true`를 지정합니다. 웹은 기본 `http://localhost:5050`을 사용하고, 다른 플랫폼이나 별도 호스트는 `--dart-define=SERVER_BASE_URL=<url>`을 함께 지정합니다.
- 해지 추천 탭은 기존 `/predict_batch` 결과가 `SubscriptionProvider.results`에 들어온 뒤 작동합니다.
- 캘린더 탭은 실제 결제일 데이터가 없으므로 구독 목록 순서 기반으로 임시 결제일을 생성합니다.
- 결제일 선택, 날짜별 결제 목록, 월/연도 선택은 모두 프론트엔드 로컬 상태로 동작합니다.
- 앱 잠금 설정, PIN, 생체인증 사용 여부, 약관 동의 상태는 `SharedPreferences`에 저장됩니다.
- Face ID/Touch ID는 `local_auth` 플러그인을 통해 실제 기기 인증을 호출합니다.
- 데이터 내보내기와 보관함은 백엔드 연동 전 자리 표시 UI로 구성했습니다.

## 필요한 백엔드/API

### 1. 결제 일정 필드

구독 생성/수정/조회 API에 아래 필드가 필요합니다.

- `billing_day`: 매월 결제일
- `next_billing_at`: 다음 결제 예정일
- `trial_ends_at`: 무료체험 종료일
- `discount_ends_at`: 할인 종료일
- `renewal_notice_days`: 알림 기준일

### 2. 캘린더 API

월별 결제 이벤트를 직접 조회하는 API가 있으면 프론트 계산을 제거할 수 있습니다.

```http
GET /billing_events?year=2026&month=5
```

응답 예시:

```json
[
  {
    "id": "evt_1",
    "subscription_id": "12",
    "name": "Netflix",
    "emoji": "🎬",
    "amount": 17000,
    "event_type": "monthly_payment",
    "scheduled_at": "2026-05-17T00:00:00Z"
  }
]
```

### 3. 설정 저장 API

설정 탭의 토글을 영구 저장하려면 기기별 설정 API가 필요합니다.

```http
GET /settings
PATCH /settings
```

필드 예시:

- `billing_alert_enabled`
- `weekly_report_enabled`
- `app_lock_enabled`
- `biometric_lock_enabled`
- `terms_accepted_at`
- `default_notice_days`

### 4. 데이터 내보내기 API

CSV 또는 JSON 내보내기용 엔드포인트가 필요합니다.

```http
GET /export/subscriptions.csv
GET /export/subscriptions.json
```

### 5. 삭제/해지 보관함 API

해지한 구독과 삭제한 구독을 별도 관리하려면 soft delete 또는 archive 모델이 필요합니다.

```http
GET /archived_subscriptions
POST /subscriptions/:id/archive
POST /subscriptions/:id/restore
```

### 6. 해지 추천 개선 API

현재는 예측 결과만 표시합니다. 추천 화면을 더 안정적으로 만들려면 추천 전용 API가 좋습니다.

```http
GET /recommendations
```

응답에는 추천 순위, 설명, 예상 절감액, 권장 액션을 포함하는 것이 좋습니다.
