import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/subscription.dart';
import '../providers/subscription_provider.dart';
import '../theme/app_theme.dart';
import '../widgets/app_top_bar.dart';

const double _maxContentWidth = 460;

class RecommendationsScreen extends StatelessWidget {
  const RecommendationsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: Consumer<SubscriptionProvider>(
        builder: (context, provider, _) {
          final candidates = provider.items
              .where(
                  (item) => provider.results[item.id]?.isChurnCandidate == true)
              .toList()
            ..sort((a, b) {
              final ar = provider.results[a.id]!;
              final br = provider.results[b.id]!;
              final confidence = br.confidence.compareTo(ar.confidence);
              if (confidence != 0) return confidence;
              return b.effectiveMonthlyCost.compareTo(a.effectiveMonthlyCost);
            });
          final saveable = candidates.fold<int>(
            0,
            (sum, item) => sum + item.effectiveMonthlyCost,
          );

          return Column(
            children: [
              const _Header(),
              _Hero(
                count: candidates.length,
                saveable: saveable,
                isAnalyzing: provider.isAnalyzing,
              ),
              Expanded(
                child: RefreshIndicator(
                  onRefresh: provider.analyzeAll,
                  color: AppColors.primary,
                  child: CustomScrollView(
                    physics: const AlwaysScrollableScrollPhysics(),
                    slivers: [
                      if (provider.items.isEmpty)
                        const SliverToBoxAdapter(child: _EmptySubscriptions())
                      else if (candidates.isEmpty)
                        SliverToBoxAdapter(
                          child: _NoCandidates(
                            isAnalyzing: provider.isAnalyzing,
                            onAnalyze: provider.analyzeAll,
                          ),
                        )
                      else
                        SliverToBoxAdapter(
                          child: _CandidateList(
                            items: candidates,
                            provider: provider,
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
  const _Header();

  @override
  Widget build(BuildContext context) {
    return const AppTopBar(
      child: Align(
        alignment: Alignment.centerLeft,
        child: Text(
          '해지 추천',
          style: TextStyle(
            fontSize: 17,
            fontWeight: FontWeight.w800,
            color: AppColors.textPrimary,
            letterSpacing: 0,
          ),
        ),
      ),
    );
  }
}

class _Hero extends StatelessWidget {
  final int count;
  final int saveable;
  final bool isAnalyzing;

  const _Hero({
    required this.count,
    required this.saveable,
    required this.isAnalyzing,
  });

  @override
  Widget build(BuildContext context) {
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
                  Row(
                    children: [
                      const Text(
                        '이번 달 줄일 수 있는 금액',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                          color: AppColors.textTertiary,
                        ),
                      ),
                      if (isAnalyzing) ...[
                        const SizedBox(width: 8),
                        const SizedBox(
                          width: 12,
                          height: 12,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        ),
                      ],
                    ],
                  ),
                  const SizedBox(height: 8),
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Flexible(
                        child: Text(
                          formatKRW(saveable),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          textAlign: TextAlign.left,
                          style: const TextStyle(
                            fontSize: 40,
                            fontWeight: FontWeight.w800,
                            color: AppColors.danger,
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
                  ),
                  const SizedBox(height: 10),
                  Text(
                    count > 0 ? '$count개 구독이 해지 후보예요' : '아직 해지 후보가 없어요',
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

class _CandidateList extends StatelessWidget {
  final List<Subscription> items;
  final SubscriptionProvider provider;

  const _CandidateList({required this.items, required this.provider});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: _maxContentWidth),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 14, 16, 0),
          child: Column(
            children: [
              for (final item in items)
                _CandidateCard(
                  item: item,
                  result: provider.results[item.id]!,
                  onKeep: () => provider.submitChurnFeedback(
                    subscriptionId: item.id,
                    actualKept: true,
                  ),
                  onCancel: () => provider.submitChurnFeedback(
                    subscriptionId: item.id,
                    actualKept: false,
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _CandidateCard extends StatelessWidget {
  final Subscription item;
  final ChurnResult result;
  final VoidCallback onKeep;
  final VoidCallback onCancel;

  const _CandidateCard({
    required this.item,
    required this.result,
    required this.onKeep,
    required this.onCancel,
  });

  @override
  Widget build(BuildContext context) {
    final feedback = item.lastFeedbackKept ?? result.userFeedbackKept;

    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: AppColors.dangerSoft,
                  borderRadius: BorderRadius.circular(14),
                ),
                alignment: Alignment.center,
                child: Text(
                  item.emoji ?? '📦',
                  style: const TextStyle(fontSize: 24),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      item.name,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w800,
                        color: AppColors.textPrimary,
                      ),
                    ),
                    const SizedBox(height: 3),
                    Text(
                      '월 ${formatKRW(item.effectiveMonthlyCost)}원',
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: AppColors.textTertiary,
                      ),
                    ),
                  ],
                ),
              ),
              _ConfidencePill(value: result.confidence),
            ],
          ),
          const SizedBox(height: 16),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: AppColors.dangerTint,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Text(
              result.reason,
              style: const TextStyle(
                fontSize: 13,
                height: 1.5,
                color: AppColors.textSecondary,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _DecisionButton(
                  label: '유지했어요',
                  icon: CupertinoIcons.check_mark,
                  selected: feedback == true,
                  color: AppColors.success,
                  onTap: onKeep,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _DecisionButton(
                  label: '직접 해지',
                  icon: CupertinoIcons.scissors,
                  selected: feedback == false,
                  color: AppColors.danger,
                  onTap: onCancel,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _ConfidencePill extends StatelessWidget {
  final double value;
  const _ConfidencePill({required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: AppColors.dangerSoft,
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        '${(value * 100).toStringAsFixed(0)}%',
        style: const TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.w800,
          color: AppColors.danger,
        ),
      ),
    );
  }
}

class _DecisionButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final bool selected;
  final Color color;
  final VoidCallback onTap;

  const _DecisionButton({
    required this.label,
    required this.icon,
    required this.selected,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      selected: selected,
      label: label,
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          height: 42,
          decoration: BoxDecoration(
            color: selected ? color : AppColors.neutralSoft,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: selected ? color : AppColors.border),
          ),
          alignment: Alignment.center,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 15, color: selected ? Colors.white : color),
              const SizedBox(width: 6),
              Text(
                label,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w800,
                  color: selected ? Colors.white : color,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _NoCandidates extends StatelessWidget {
  final bool isAnalyzing;
  final VoidCallback onAnalyze;

  const _NoCandidates({required this.isAnalyzing, required this.onAnalyze});

  @override
  Widget build(BuildContext context) {
    return _CenteredCard(
      icon: CupertinoIcons.check_mark_circled,
      title: isAnalyzing ? '분석 중이에요' : '지금은 유지 가치가 높아요',
      body: isAnalyzing
          ? '구독 사용 패턴을 다시 확인하고 있어요'
          : '새 구독을 추가하거나 사용 패턴을 바꾸면 다시 추천해드려요',
      actionLabel: '다시 분석',
      onAction: isAnalyzing ? null : onAnalyze,
    );
  }
}

class _EmptySubscriptions extends StatelessWidget {
  const _EmptySubscriptions();

  @override
  Widget build(BuildContext context) {
    return const _CenteredCard(
      icon: CupertinoIcons.plus_circle,
      title: '구독을 먼저 추가해주세요',
      body: '홈에서 구독을 추가하면 해지 후보를 분석할 수 있어요',
    );
  }
}

class _CenteredCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String body;
  final String? actionLabel;
  final VoidCallback? onAction;

  const _CenteredCard({
    required this.icon,
    required this.title,
    required this.body,
    this.actionLabel,
    this.onAction,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: _maxContentWidth),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
          child: Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 42),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Column(
              children: [
                Icon(icon, size: 38, color: AppColors.textDisabled),
                const SizedBox(height: 14),
                Text(
                  title,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  body,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 13,
                    height: 1.5,
                    color: AppColors.textTertiary,
                  ),
                ),
                if (actionLabel != null) ...[
                  const SizedBox(height: 18),
                  Semantics(
                    button: true,
                    enabled: onAction != null,
                    label: actionLabel,
                    child: GestureDetector(
                      onTap: onAction,
                      child: Container(
                        height: 42,
                        padding: const EdgeInsets.symmetric(horizontal: 18),
                        decoration: BoxDecoration(
                          color: onAction == null
                              ? AppColors.neutralChipDark
                              : AppColors.primary,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        alignment: Alignment.center,
                        child: Text(
                          actionLabel!,
                          style: TextStyle(
                            fontSize: 14,
                            fontWeight: FontWeight.w800,
                            color: onAction == null
                                ? AppColors.textDisabled
                                : Colors.white,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}
