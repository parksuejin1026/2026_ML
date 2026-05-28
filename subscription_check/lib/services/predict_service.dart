import 'dart:convert';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;
import '../config/backend_mode.dart';
import '../models/subscription.dart';
import '../providers/subscription_provider.dart';
import 'device_id_service.dart';

String get _baseUrl {
  if (configuredServerBaseUrl.isNotEmpty) return configuredServerBaseUrl;
  if (kIsWeb) return 'http://localhost:5050';
  throw StateError(
    'SERVER_BASE_URL is required when BACKEND_ENABLED=true on this platform.',
  );
}

Future<Map<String, String>> _jsonHeaders() async {
  final deviceId = await getOrCreateDeviceId();
  return {
    'Content-Type': 'application/json',
    'X-Device-Id': deviceId,
  };
}

// ─── Subscription CRUD ─────────────────────────────────────────────────────

Future<List<Subscription>> fetchSubscriptions() async {
  if (!backendEnabled) {
    return _mockSubscriptions.map((s) => s.copyWith()).toList();
  }

  final response = await http.get(
    Uri.parse('$_baseUrl/subscriptions'),
    headers: await _jsonHeaders(),
  );
  if (response.statusCode != 200) {
    throw Exception('구독 목록 조회 실패: ${response.statusCode}');
  }
  final list = jsonDecode(utf8.decode(response.bodyBytes)) as List<dynamic>;
  return list
      .map((e) => Subscription.fromServerJson(e as Map<String, dynamic>))
      .toList();
}

Future<Subscription> createSubscription(Subscription s) async {
  if (!backendEnabled) {
    final created = s.copyWith(id: s.id.isEmpty ? _nextMockId() : s.id);
    _mockSubscriptions.add(created);
    return created;
  }

  final response = await http.post(
    Uri.parse('$_baseUrl/subscriptions'),
    headers: await _jsonHeaders(),
    body: jsonEncode(s.toPersistenceJson()),
  );
  if (response.statusCode != 201) {
    throw Exception('구독 생성 실패: ${response.statusCode}');
  }
  return Subscription.fromServerJson(
    jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>,
  );
}

Future<Subscription> patchSubscription(Subscription s) async {
  if (!backendEnabled) {
    final idx = _mockSubscriptions.indexWhere((item) => item.id == s.id);
    if (idx >= 0) {
      _mockSubscriptions[idx] = s;
    }
    return s;
  }

  final response = await http.patch(
    Uri.parse('$_baseUrl/subscriptions/${s.id}'),
    headers: await _jsonHeaders(),
    body: jsonEncode(s.toPersistenceJson()),
  );
  if (response.statusCode != 200) {
    throw Exception('구독 수정 실패: ${response.statusCode}');
  }
  return Subscription.fromServerJson(
    jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>,
  );
}

Future<void> deleteSubscription(String id) async {
  if (!backendEnabled) {
    _mockSubscriptions.removeWhere((item) => item.id == id);
    return;
  }

  final response = await http.delete(
    Uri.parse('$_baseUrl/subscriptions/$id'),
    headers: await _jsonHeaders(),
  );
  if (response.statusCode != 200) {
    throw Exception('구독 삭제 실패: ${response.statusCode}');
  }
}

// ─── Prediction ────────────────────────────────────────────────────────────

Future<Map<String, ChurnResult>> predictBatch(
  List<Subscription> subscriptions,
) async {
  if (!backendEnabled) {
    return _mockPredictBatch(subscriptions);
  }

  final body = subscriptions.map((s) => s.toApiJson()).toList();

  final response = await http.post(
    Uri.parse('$_baseUrl/predict_batch'),
    headers: await _jsonHeaders(),
    body: jsonEncode(body),
  );

  if (response.statusCode != 200) {
    throw Exception('서버 응답 오류: ${response.statusCode}');
  }

  final Map<String, dynamic> data = jsonDecode(utf8.decode(response.bodyBytes));
  final results = <String, ChurnResult>{};

  for (final entry in data.entries) {
    final r = entry.value as Map<String, dynamic>;
    results[entry.key] = ChurnResult(
      predictionId: (r['prediction_id'] as num).toInt(),
      isChurnCandidate: r['is_churn_candidate'] as bool,
      confidence: (r['confidence'] as num).toDouble(),
      reason: r['reason'] as String,
    );
  }
  return results;
}

Future<void> submitFeedback({
  required int predictionId,
  required bool actualKept,
  String? subscriptionId,
}) async {
  if (!backendEnabled) {
    final id = subscriptionId;
    if (id != null) {
      final idx = _mockSubscriptions.indexWhere((item) => item.id == id);
      if (idx >= 0) {
        _mockSubscriptions[idx] = _mockSubscriptions[idx].copyWith(
          lastFeedbackKept: actualKept,
          lastFeedbackAt: DateTime.now(),
        );
      }
    }
    return;
  }

  final body = <String, dynamic>{
    'prediction_id': predictionId,
    'actual_kept': actualKept,
  };
  if (subscriptionId != null) {
    body['subscription_id'] = int.tryParse(subscriptionId) ?? subscriptionId;
  }

  final response = await http.post(
    Uri.parse('$_baseUrl/feedback'),
    headers: await _jsonHeaders(),
    body: jsonEncode(body),
  );

  if (response.statusCode != 200) {
    throw Exception('피드백 전송 실패: ${response.statusCode}');
  }
}

// ─── Savings (절감액 트래커) ──────────────────────────────────────────────

class SavingsHistoryItem {
  final int predictionId;
  final String? subscriptionName;
  final String? emoji;
  final String subscriptionType;
  final int effectiveMonthlyCost;
  final DateTime? feedbackAt;

  SavingsHistoryItem({
    required this.predictionId,
    this.subscriptionName,
    this.emoji,
    required this.subscriptionType,
    required this.effectiveMonthlyCost,
    required this.feedbackAt,
  });

  factory SavingsHistoryItem.fromJson(Map<String, dynamic> json) {
    final fb = json['feedback_at'] as String?;
    return SavingsHistoryItem(
      predictionId: (json['prediction_id'] as num).toInt(),
      subscriptionName: json['subscription_name'] as String?,
      emoji: json['emoji'] as String?,
      subscriptionType: json['subscription_type'] as String? ?? '',
      effectiveMonthlyCost: (json['effective_monthly'] as num?)?.toInt() ?? 0,
      feedbackAt: fb != null ? DateTime.tryParse(fb) : null,
    );
  }
}

class SavingsSummary {
  final int cancelledCount;
  final int keptCount;
  final int monthlySavings;
  final int cumulativeSavings;
  final List<SavingsHistoryItem> history;

  SavingsSummary({
    required this.cancelledCount,
    required this.keptCount,
    required this.monthlySavings,
    required this.cumulativeSavings,
    required this.history,
  });

  factory SavingsSummary.fromJson(Map<String, dynamic> json) {
    return SavingsSummary(
      cancelledCount: (json['cancelled_count'] as num?)?.toInt() ?? 0,
      keptCount: (json['kept_count'] as num?)?.toInt() ?? 0,
      monthlySavings: (json['monthly_savings'] as num?)?.toInt() ?? 0,
      cumulativeSavings: (json['cumulative_savings'] as num?)?.toInt() ?? 0,
      history: ((json['history'] as List<dynamic>?) ?? [])
          .map((e) => SavingsHistoryItem.fromJson(e as Map<String, dynamic>))
          .toList(),
    );
  }
}

Future<SavingsSummary> fetchSavings() async {
  if (!backendEnabled) {
    return _mockSavingsSummary();
  }

  final response = await http.get(
    Uri.parse('$_baseUrl/savings'),
    headers: await _jsonHeaders(),
  );
  if (response.statusCode != 200) {
    throw Exception('절감액 조회 실패: ${response.statusCode}');
  }
  return SavingsSummary.fromJson(
    jsonDecode(utf8.decode(response.bodyBytes)) as Map<String, dynamic>,
  );
}

Future<bool> checkServerHealth() async {
  if (!backendEnabled) return false;

  try {
    final response = await http
        .get(Uri.parse('$_baseUrl/health'))
        .timeout(const Duration(seconds: 3));
    return response.statusCode == 200;
  } catch (_) {
    return false;
  }
}

final _mockSubscriptions = <Subscription>[
  Subscription(
    id: 'demo-netflix',
    name: 'Netflix',
    type: 'Video',
    monthlyCost: 17000,
    useFrequency: UseFrequency.weekly,
    lastUseRecency: LastUseRecency.between1and7d,
    perceivedNecessity: 3,
    replacementAvailable: true,
    billingDay: 12,
    emoji: '🎬',
  ),
  Subscription(
    id: 'demo-melon',
    name: 'Melon',
    type: 'Music',
    monthlyCost: 10900,
    useFrequency: UseFrequency.frequent,
    lastUseRecency: LastUseRecency.under1d,
    perceivedNecessity: 5,
    replacementAvailable: false,
    billingDay: 8,
    emoji: '🎧',
  ),
  Subscription(
    id: 'demo-icloud',
    name: 'iCloud+',
    type: 'Cloud',
    monthlyCost: 3300,
    useFrequency: UseFrequency.frequent,
    lastUseRecency: LastUseRecency.under1d,
    perceivedNecessity: 4,
    replacementAvailable: false,
    billingDay: 20,
    emoji: '☁️',
  ),
  Subscription(
    id: 'demo-news',
    name: 'Digital News',
    type: 'News',
    monthlyCost: 12000,
    useFrequency: UseFrequency.rare,
    lastUseRecency: LastUseRecency.over30d,
    perceivedNecessity: 2,
    replacementAvailable: true,
    billingDay: 27,
    emoji: '📰',
  ),
];

int _mockId = 1000;
int _mockPredictionId = 2000;

String _nextMockId() {
  _mockId += 1;
  return 'demo-$_mockId';
}

Map<String, ChurnResult> _mockPredictBatch(List<Subscription> subscriptions) {
  return {
    for (final subscription in subscriptions)
      subscription.id: _mockPrediction(subscription),
  };
}

ChurnResult _mockPrediction(Subscription subscription) {
  final lowUsage = subscription.useFrequency == UseFrequency.rare ||
      subscription.lastUseRecency == LastUseRecency.over30d;
  final replaceable = subscription.replacementAvailable;
  final expensive = subscription.effectiveMonthlyCost >= 12000;
  final lowNeed = subscription.perceivedNecessity <= 2;
  final isCandidate = (lowUsage && replaceable) || (expensive && lowNeed);
  final confidence = isCandidate
      ? (0.72 + (replaceable ? 0.08 : 0) + (lowNeed ? 0.07 : 0)).clamp(
          0.0,
          0.94,
        )
      : (0.38 + (subscription.perceivedNecessity / 20)).clamp(0.0, 0.68);

  _mockPredictionId += 1;
  return ChurnResult(
    predictionId: _mockPredictionId,
    isChurnCandidate: isCandidate,
    confidence: confidence,
    reason: isCandidate
        ? '최근 사용 빈도와 필요도가 낮고 대체 서비스가 있어 해지 후보로 분류했어요.'
        : '사용 빈도와 필요도가 높아 현재는 유지 가치가 더 높아 보여요.',
    userFeedbackKept: subscription.lastFeedbackKept,
  );
}

SavingsSummary _mockSavingsSummary() {
  final cancelled = _mockSubscriptions
      .where((subscription) => subscription.lastFeedbackKept == false)
      .toList();
  final kept = _mockSubscriptions
      .where((subscription) => subscription.lastFeedbackKept == true)
      .toList();
  final monthlySavings = cancelled.fold<int>(
    0,
    (sum, subscription) => sum + subscription.effectiveMonthlyCost,
  );

  return SavingsSummary(
    cancelledCount: cancelled.length,
    keptCount: kept.length,
    monthlySavings: monthlySavings,
    cumulativeSavings: monthlySavings * 3,
    history: [
      for (var i = 0; i < cancelled.length; i++)
        SavingsHistoryItem(
          predictionId: 3000 + i,
          subscriptionName: cancelled[i].name,
          emoji: cancelled[i].emoji,
          subscriptionType: cancelled[i].type,
          effectiveMonthlyCost: cancelled[i].effectiveMonthlyCost,
          feedbackAt: cancelled[i].lastFeedbackAt,
        ),
    ],
  );
}
