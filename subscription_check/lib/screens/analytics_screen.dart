import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';

import '../models/subscription.dart';
import '../providers/subscription_provider.dart';
import '../services/predict_service.dart';
import '../theme/app_theme.dart';
import '../widgets/app_top_bar.dart';

const double _maxContentWidth = 460;

const _categoryLabels = {
  'Video': '영상',
  'Music': '음악',
  'Cloud': '생활',
  'Education': '교육',
  'Game': '게임',
  'Fitness': '운동',
  'News': '뉴스',
};

const _categoryColors = {
  'Video': Color(0xFF53B2FF),
  'Music': Color(0xFF7C5CFA),
  'Cloud': Color(0xFF00B386),
  'Education': Color(0xFFF6A63B),
  'Game': Color(0xFFF04452),
  'Fitness': Color(0xFF19BCD4),
  'News': Color(0xFF8B95A1),
};

const Color _etcColor = Color(0xFFB0B8C1);

String _categoryLabel(String key) => _categoryLabels[key] ?? key;
Color _categoryColor(String key) => _categoryColors[key] ?? _etcColor;

class AnalyticsScreen extends StatefulWidget {
  final bool showBackButton;

  const AnalyticsScreen({super.key, this.showBackButton = true});

  @override
  State<AnalyticsScreen> createState() => _AnalyticsScreenState();
}

class _AnalyticsScreenState extends State<AnalyticsScreen> {
  late Future<SavingsSummary> _savingsFuture;

  @override
  void initState() {
    super.initState();
    _savingsFuture = fetchSavings();
  }

  Future<void> _refresh() async {
    setState(() {
      _savingsFuture = fetchSavings();
    });
    await _savingsFuture;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: Consumer<SubscriptionProvider>(
        builder: (context, provider, _) {
          return Column(
            children: [
              _Header(showBackButton: widget.showBackButton),
              FutureBuilder<SavingsSummary>(
                future: _savingsFuture,
                builder: (context, snapshot) {
                  return _SavingsHeroSection(snapshot: snapshot);
                },
              ),
              Expanded(
                child: RefreshIndicator(
                  onRefresh: _refresh,
                  color: AppColors.primary,
                  child: CustomScrollView(
                    physics: const AlwaysScrollableScrollPhysics(),
                    slivers: [
                      SliverToBoxAdapter(
                        child: _CategoryBreakdownSection(provider: provider),
                      ),
                      SliverToBoxAdapter(
                        child: FutureBuilder<SavingsSummary>(
                          future: _savingsFuture,
                          builder: (context, snapshot) {
                            final summary = snapshot.data;
                            if (summary == null || summary.history.isEmpty) {
                              return const SizedBox.shrink();
                            }
                            return _CancellationHistorySection(
                                history: summary.history);
                          },
                        ),
                      ),
                      const SliverToBoxAdapter(child: SizedBox(height: 138)),
                    ],
                  ),
                ),
              ),
            ],
          );
        },
      ),
    );
  }
}

class _Header extends StatelessWidget {
  final bool showBackButton;
  const _Header({required this.showBackButton});

  @override
  Widget build(BuildContext context) {
    return AppTopBar(
      padding: const EdgeInsets.symmetric(horizontal: 8),
      child: Row(
        children: [
          if (showBackButton) ...[
            IconButton(
              icon: const Icon(Icons.arrow_back_ios_new,
                  size: 18, color: AppColors.textPrimary),
              onPressed: () => Navigator.of(context).pop(),
            ),
            const SizedBox(width: 4),
          ] else
            const SizedBox(width: 16),
          const Text(
            '지출 분석',
            style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
              letterSpacing: 0,
            ),
          ),
        ],
      ),
    );
  }
}

// ─── 1. 누적 절감액 헤로 ────────────────────────────────────────────────

class _SavingsHeroSection extends StatelessWidget {
  final AsyncSnapshot<SavingsSummary> snapshot;
  const _SavingsHeroSection({required this.snapshot});

  @override
  Widget build(BuildContext context) {
    final isLoading = snapshot.connectionState == ConnectionState.waiting;
    final hasError = snapshot.hasError;
    final summary = snapshot.data;

    return Container(
      color: AppColors.surface,
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: _maxContentWidth),
          child: SizedBox(
            height: 144,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(24, 18, 24, 22),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '지금까지 절약한 금액',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: AppColors.textTertiary,
                    ),
                  ),
                  const SizedBox(height: 8),
                  if (isLoading)
                    const _BigAmountPlaceholder()
                  else if (hasError || summary == null)
                    const _BigAmountError()
                  else
                    _BigAmount(value: summary.cumulativeSavings),
                  const SizedBox(height: 10),
                  if (!isLoading &&
                      !hasError &&
                      summary != null &&
                      summary.cancelledCount == 0)
                    const Text(
                      '해지 피드백을 남기면 절감액이 쌓여요',
                      style: TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                        color: AppColors.textDisabled,
                      ),
                    )
                  else if (!isLoading && !hasError && summary != null)
                    Text(
                      '해지 피드백 ${summary.cancelledCount}건 반영',
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w500,
                        color: AppColors.textDisabled,
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _BigAmount extends StatelessWidget {
  final int value;
  const _BigAmount({required this.value});

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Flexible(
          child: Text(
            formatKRW(value),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            textAlign: TextAlign.left,
            style: const TextStyle(
              fontSize: 40,
              fontWeight: FontWeight.w800,
              color: AppColors.success,
              letterSpacing: 0,
              height: 1.0,
            ),
          ),
        ),
        const SizedBox(width: 6),
        const Padding(
          padding: EdgeInsets.only(bottom: 3),
          child: Text(
            '원',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w600,
              color: AppColors.textSecondary,
            ),
          ),
        ),
      ],
    );
  }
}

class _BigAmountPlaceholder extends StatelessWidget {
  const _BigAmountPlaceholder();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 180,
      height: 44,
      decoration: BoxDecoration(
        color: AppColors.neutralSoft,
        borderRadius: BorderRadius.circular(8),
      ),
    );
  }
}

class _BigAmountError extends StatelessWidget {
  const _BigAmountError();

  @override
  Widget build(BuildContext context) {
    return const Text(
      '-',
      style: TextStyle(
        fontSize: 40,
        fontWeight: FontWeight.w800,
        color: AppColors.textDisabled,
        height: 1.0,
      ),
    );
  }
}

// ─── 2. 카테고리 파이 차트 ──────────────────────────────────────────────

class _CategoryBreakdownSection extends StatefulWidget {
  final SubscriptionProvider provider;
  const _CategoryBreakdownSection({required this.provider});

  @override
  State<_CategoryBreakdownSection> createState() =>
      _CategoryBreakdownSectionState();
}

class _CategoryBreakdownSectionState extends State<_CategoryBreakdownSection> {
  int? _touchedIndex;

  @override
  Widget build(BuildContext context) {
    final totals = _aggregateByCategory(widget.provider.items);
    final total = totals.fold<int>(0, (sum, e) => sum + e.amount);

    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: _maxContentWidth),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
          child: Container(
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(20),
            ),
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '카테고리별 지출',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  total > 0 ? '이번 달 ${formatKRW(total)}원 지출 중' : '등록된 구독이 없어요',
                  style: const TextStyle(
                    fontSize: 12,
                    color: AppColors.textTertiary,
                  ),
                ),
                const SizedBox(height: 20),
                if (totals.isEmpty)
                  const _EmptyChart()
                else
                  SizedBox(
                    height: 200,
                    child: PieChart(
                      PieChartData(
                        sectionsSpace: 2,
                        centerSpaceRadius: 56,
                        pieTouchData: PieTouchData(
                          touchCallback: (event, response) {
                            setState(() {
                              if (!event.isInterestedForInteractions ||
                                  response == null ||
                                  response.touchedSection == null) {
                                _touchedIndex = null;
                                return;
                              }
                              _touchedIndex =
                                  response.touchedSection!.touchedSectionIndex;
                            });
                          },
                        ),
                        sections: [
                          for (var i = 0; i < totals.length; i++)
                            _buildSection(totals[i], i == _touchedIndex, total),
                        ],
                      ),
                    ),
                  ),
                if (totals.isNotEmpty) ...[
                  const SizedBox(height: 20),
                  _Legend(totals: totals, total: total),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  PieChartSectionData _buildSection(
      _CategoryTotal item, bool touched, int total) {
    final radius = touched ? 58.0 : 48.0;
    final percent = total > 0 ? (item.amount / total * 100) : 0.0;

    return PieChartSectionData(
      color: _categoryColor(item.key),
      value: item.amount.toDouble(),
      title: percent >= 8 ? '${percent.toStringAsFixed(0)}%' : '',
      radius: radius,
      titleStyle: const TextStyle(
        fontSize: 11,
        fontWeight: FontWeight.w700,
        color: Colors.white,
      ),
    );
  }
}

class _EmptyChart extends StatelessWidget {
  const _EmptyChart();

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 200,
      alignment: Alignment.center,
      child: const Text(
        '구독을 추가하면 차트가 나타나요',
        style: TextStyle(
          fontSize: 13,
          color: AppColors.textDisabled,
        ),
      ),
    );
  }
}

class _Legend extends StatelessWidget {
  final List<_CategoryTotal> totals;
  final int total;
  const _Legend({required this.totals, required this.total});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        for (final item in totals) ...[
          _LegendRow(item: item, total: total),
          if (item != totals.last) const SizedBox(height: 10),
        ],
      ],
    );
  }
}

class _LegendRow extends StatelessWidget {
  final _CategoryTotal item;
  final int total;
  const _LegendRow({required this.item, required this.total});

  @override
  Widget build(BuildContext context) {
    final percent = total > 0 ? (item.amount / total * 100) : 0.0;

    return Row(
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(
            color: _categoryColor(item.key),
            borderRadius: BorderRadius.circular(3),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Text(
            '${_categoryLabel(item.key)} · ${item.count}개',
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w500,
              color: AppColors.textSecondary,
            ),
          ),
        ),
        Text(
          '${formatKRW(item.amount)}원',
          style: const TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        const SizedBox(width: 8),
        SizedBox(
          width: 40,
          child: Text(
            '${percent.toStringAsFixed(0)}%',
            textAlign: TextAlign.end,
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: AppColors.textTertiary,
            ),
          ),
        ),
      ],
    );
  }
}

class _CategoryTotal {
  final String key;
  final int amount;
  final int count;
  const _CategoryTotal(this.key, this.amount, this.count);
}

List<_CategoryTotal> _aggregateByCategory(List<Subscription> items) {
  final amountMap = <String, int>{};
  final countMap = <String, int>{};

  for (final s in items) {
    amountMap.update(s.type, (v) => v + s.effectiveMonthlyCost,
        ifAbsent: () => s.effectiveMonthlyCost);
    countMap.update(s.type, (v) => v + 1, ifAbsent: () => 1);
  }

  final result = amountMap.entries
      .map((e) => _CategoryTotal(e.key, e.value, countMap[e.key] ?? 0))
      .where((t) => t.amount > 0)
      .toList();

  result.sort((a, b) => b.amount.compareTo(a.amount));
  return result;
}

// ─── 3. 해지 내역 ──────────────────────────────────────────────────────

class _CancellationHistorySection extends StatelessWidget {
  final List<SavingsHistoryItem> history;
  const _CancellationHistorySection({required this.history});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: _maxContentWidth),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
          child: Container(
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(20),
            ),
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '최근 해지 내역',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 12),
                for (final item in history) _HistoryRow(item: item),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _HistoryRow extends StatelessWidget {
  final SavingsHistoryItem item;
  const _HistoryRow({required this.item});

  @override
  Widget build(BuildContext context) {
    final dateText = item.feedbackAt != null
        ? DateFormat('yyyy.MM.dd').format(item.feedbackAt!.toLocal())
        : '';

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: Row(
        children: [
          Container(
            width: 32,
            height: 32,
            decoration: BoxDecoration(
              color:
                  _categoryColor(item.subscriptionType).withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(10),
            ),
            alignment: Alignment.center,
            child: Transform.rotate(
              angle: -0.785,
              child: Icon(
                Icons.content_cut,
                size: 14,
                color: _categoryColor(item.subscriptionType),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _categoryLabel(item.subscriptionType),
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  dateText,
                  style: const TextStyle(
                    fontSize: 11,
                    color: AppColors.textTertiary,
                  ),
                ),
              ],
            ),
          ),
          Text(
            '-${formatKRW(item.effectiveMonthlyCost)}원/월',
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: AppColors.success,
            ),
          ),
        ],
      ),
    );
  }
}
