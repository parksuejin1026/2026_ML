import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../models/subscription.dart';
import '../providers/subscription_provider.dart';
import '../theme/app_theme.dart';
import '../widgets/app_top_bar.dart';
import '../widgets/subscription_card.dart';
import '../widgets/add_subscription_sheet.dart';

const double _maxContentWidth = 460;

enum _HomeSort { recommended, priceDesc, priceAsc, name }

const _categoryFilters = [
  _CategoryFilter('all', '전체'),
  _CategoryFilter('Video', '영상'),
  _CategoryFilter('Music', '음악'),
  _CategoryFilter('Cloud', '생활'),
  _CategoryFilter('Education', '교육'),
  _CategoryFilter('Game', '게임'),
  _CategoryFilter('Fitness', '운동'),
  _CategoryFilter('News', '뉴스'),
];

class _CategoryFilter {
  final String key;
  final String label;
  const _CategoryFilter(this.key, this.label);
}

String _sortLabel(_HomeSort sort) {
  switch (sort) {
    case _HomeSort.recommended:
      return '추천순';
    case _HomeSort.priceDesc:
      return '높은 금액순';
    case _HomeSort.priceAsc:
      return '낮은 금액순';
    case _HomeSort.name:
      return '이름순';
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _category = 'all';
  _HomeSort _sort = _HomeSort.recommended;

  List<Subscription> _visibleItems(SubscriptionProvider provider) {
    final items = provider.items.where((item) {
      return _category == 'all' || item.type == _category;
    }).toList();

    items.sort((a, b) {
      switch (_sort) {
        case _HomeSort.recommended:
          final ar = provider.results[a.id];
          final br = provider.results[b.id];
          final aScore =
              (ar?.isChurnCandidate == true ? 1000 : 0) + (ar?.confidence ?? 0);
          final bScore =
              (br?.isChurnCandidate == true ? 1000 : 0) + (br?.confidence ?? 0);
          return bScore.compareTo(aScore);
        case _HomeSort.priceDesc:
          return b.effectiveMonthlyCost.compareTo(a.effectiveMonthlyCost);
        case _HomeSort.priceAsc:
          return a.effectiveMonthlyCost.compareTo(b.effectiveMonthlyCost);
        case _HomeSort.name:
          return a.name.compareTo(b.name);
      }
    });
    return items;
  }

  void _openAddSheet(BuildContext context) {
    Navigator.of(context).push(
      PageRouteBuilder(
        opaque: false,
        barrierColor: Colors.black.withValues(alpha: 0.25),
        transitionDuration: const Duration(milliseconds: 320),
        reverseTransitionDuration: const Duration(milliseconds: 260),
        pageBuilder: (_, __, ___) => const AddSubscriptionSheet(),
        transitionsBuilder: (_, anim, __, child) {
          final slide = Tween<Offset>(
            begin: const Offset(0, 1),
            end: Offset.zero,
          ).animate(CurvedAnimation(
            parent: anim,
            curve: Curves.easeOutCubic,
            reverseCurve: Curves.easeInCubic,
          ));
          return SlideTransition(position: slide, child: child);
        },
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: Consumer<SubscriptionProvider>(
        builder: (context, provider, _) {
          return Stack(
            children: [
              Column(
                children: [
                  _Header(onAdd: () => _openAddSheet(context)),
                  _HeroSection(provider: provider),
                  Expanded(
                    child: RefreshIndicator(
                      onRefresh: () => provider.loadFromServer(),
                      color: AppColors.primary,
                      child: CustomScrollView(
                        physics: const AlwaysScrollableScrollPhysics(),
                        slivers: [
                          SliverToBoxAdapter(
                            child: _SubscriptionListSection(
                              provider: provider,
                              items: _visibleItems(provider),
                              category: _category,
                              sort: _sort,
                              onCategoryChanged: (value) {
                                setState(() => _category = value);
                              },
                              onSortChanged: (value) {
                                setState(() => _sort = value);
                              },
                              onAdd: () => _openAddSheet(context),
                            ),
                          ),
                          if (provider.errorMessage != null)
                            SliverToBoxAdapter(
                              child:
                                  _ErrorBanner(message: provider.errorMessage!),
                            ),
                          const SliverToBoxAdapter(
                              child: SizedBox(height: 138)),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
              if (provider.items.isNotEmpty)
                Positioned(
                  left: 0,
                  right: 0,
                  bottom: MediaQuery.of(context).padding.bottom + 94,
                  child: Align(
                    alignment: Alignment.center,
                    child: ConstrainedBox(
                      constraints:
                          const BoxConstraints(maxWidth: _maxContentWidth),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 16),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.end,
                          children: [
                            _FloatingAddButton(
                                onTap: () => _openAddSheet(context)),
                          ],
                        ),
                      ),
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
  final VoidCallback onAdd;
  const _Header({required this.onAdd});

  @override
  Widget build(BuildContext context) {
    return AppTopBar(
      child: Row(
        children: [
          _Logo(),
          const SizedBox(width: 12),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.center,
            children: const [
              Text(
                'SubCut',
                style: TextStyle(
                  fontSize: 17,
                  fontWeight: FontWeight.w800,
                  color: AppColors.textPrimary,
                  letterSpacing: -0.34,
                  height: 1.1,
                ),
              ),
              SizedBox(height: 2),
              Text(
                '스마트 구독 관리',
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w500,
                  color: AppColors.textTertiary,
                  height: 1.2,
                ),
              ),
            ],
          ),
          const Spacer(),
          _HeaderAddButton(onTap: onAdd),
        ],
      ),
    );
  }
}

class _HeaderAddButton extends StatelessWidget {
  final VoidCallback onTap;
  const _HeaderAddButton({required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Container(
        width: 40,
        height: 40,
        decoration: BoxDecoration(
          color: AppColors.neutralChip,
          borderRadius: BorderRadius.circular(12),
        ),
        alignment: Alignment.center,
        child: const Icon(
          Icons.add_rounded,
          size: 20,
          color: AppColors.textSecondary,
        ),
      ),
    );
  }
}

class _Logo extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      width: 36,
      height: 36,
      decoration: BoxDecoration(
        gradient: AppColors.primaryGradient,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: AppColors.primary.withValues(alpha: 0.3),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      alignment: Alignment.center,
      child: Transform.rotate(
        angle: -0.785, // -45°
        child: const Icon(Icons.content_cut, size: 18, color: Colors.white),
      ),
    );
  }
}

class _HeroSection extends StatelessWidget {
  final SubscriptionProvider provider;
  const _HeroSection({required this.provider});

  @override
  Widget build(BuildContext context) {
    final total = provider.totalMonthlyCost;
    final saveable = provider.saveableCost;

    return Container(
      color: AppColors.surface,
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: _maxContentWidth),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(24, 16, 24, 28),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '매달 빠져나가는 돈',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                    color: AppColors.textTertiary,
                  ),
                ),
                const SizedBox(height: 6),
                AnimatedSwitcher(
                  duration: const Duration(milliseconds: 240),
                  transitionBuilder: (child, animation) => FadeTransition(
                    opacity: animation,
                    child: SlideTransition(
                      position: Tween<Offset>(
                              begin: const Offset(0, 0.3), end: Offset.zero)
                          .animate(animation),
                      child: child,
                    ),
                  ),
                  child: Row(
                    key: ValueKey(total),
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Text(
                        formatKRW(total),
                        style: const TextStyle(
                          fontSize: 40,
                          fontWeight: FontWeight.w800,
                          color: AppColors.textPrimary,
                          letterSpacing: -1.6,
                          height: 1.0,
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
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    Text(
                      '${provider.items.length}개 구독 서비스 이용 중',
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w400,
                        color: AppColors.textDisabled,
                      ),
                    ),
                    const SizedBox(width: 8),
                    if (saveable > 0)
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 2),
                        decoration: BoxDecoration(
                          color: AppColors.dangerSoft,
                          borderRadius: BorderRadius.circular(999),
                        ),
                        child: Text(
                          '월 ${formatKRW(saveable)}원 절약 가능',
                          style: const TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            color: AppColors.danger,
                          ),
                        ),
                      ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _SubscriptionListSection extends StatelessWidget {
  final SubscriptionProvider provider;
  final List<Subscription> items;
  final String category;
  final _HomeSort sort;
  final ValueChanged<String> onCategoryChanged;
  final ValueChanged<_HomeSort> onSortChanged;
  final VoidCallback onAdd;
  const _SubscriptionListSection({
    required this.provider,
    required this.items,
    required this.category,
    required this.sort,
    required this.onCategoryChanged,
    required this.onSortChanged,
    required this.onAdd,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: _maxContentWidth),
        child: Padding(
          padding: const EdgeInsets.only(top: 8),
          child: (() {
            if (provider.isLoading && provider.items.isEmpty) {
              return const Padding(
                padding: EdgeInsets.symmetric(vertical: 80),
                child: Center(child: CircularProgressIndicator()),
              );
            }
            if (provider.items.isEmpty) {
              return _EmptyState(onAdd: onAdd);
            }
            return _buildList(context);
          })(),
        ),
      ),
    );
  }

  Widget _buildList(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Container(
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(20),
        ),
        clipBehavior: Clip.antiAlias,
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 20, 12, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        '내 구독 ${items.length}',
                        style: const TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w700,
                          color: AppColors.textPrimary,
                        ),
                      ),
                      TextButton(
                        style: TextButton.styleFrom(
                          minimumSize: Size.zero,
                          padding: const EdgeInsets.symmetric(
                              horizontal: 8, vertical: 4),
                          tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                          foregroundColor: AppColors.primary,
                        ),
                        onPressed: onAdd,
                        child: const Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(
                              '추가',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w600,
                                color: AppColors.primary,
                              ),
                            ),
                            Icon(Icons.chevron_right,
                                size: 16, color: AppColors.primary),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  _FilterBar(
                    active: category,
                    onChanged: onCategoryChanged,
                  ),
                  const SizedBox(height: 10),
                  _SortBar(
                    sort: sort,
                    onChanged: onSortChanged,
                    visibleCount: items.length,
                    totalCount: provider.items.length,
                  ),
                ],
              ),
            ),
            if (items.isEmpty)
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 28, 20, 36),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 28),
                  decoration: BoxDecoration(
                    color: AppColors.neutralSoft,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: const Column(
                    children: [
                      Icon(Icons.filter_alt_off_outlined,
                          size: 26, color: AppColors.textDisabled),
                      SizedBox(height: 8),
                      Text(
                        '조건에 맞는 구독이 없어요',
                        style: TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: AppColors.textTertiary,
                        ),
                      ),
                    ],
                  ),
                ),
              )
            else
              for (var i = 0; i < items.length; i++) ...[
                SubscriptionCard(
                  subscription: items[i],
                  result: provider.results[items[i].id],
                  onDelete: () => provider.removeSubscription(items[i].id),
                  onUpdate: provider.updateSubscription,
                  onFeedback: (kept) => provider.submitChurnFeedback(
                    subscriptionId: items[i].id,
                    actualKept: kept,
                  ),
                ),
                if (i < items.length - 1)
                  const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 20),
                    child: Divider(
                      height: 1,
                      thickness: 1,
                      color: AppColors.divider,
                    ),
                  ),
              ],
          ],
        ),
      ),
    );
  }
}

class _FilterBar extends StatelessWidget {
  final String active;
  final ValueChanged<String> onChanged;

  const _FilterBar({required this.active, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 34,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: _categoryFilters.length,
        separatorBuilder: (_, __) => const SizedBox(width: 6),
        itemBuilder: (_, i) {
          final item = _categoryFilters[i];
          final selected = active == item.key;
          return GestureDetector(
            onTap: () => onChanged(item.key),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 160),
              padding: const EdgeInsets.symmetric(horizontal: 14),
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: selected ? AppColors.textPrimary : AppColors.neutralChip,
                borderRadius: BorderRadius.circular(999),
              ),
              child: Text(
                item.label,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w700,
                  color: selected ? Colors.white : AppColors.textTertiary,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class _SortBar extends StatelessWidget {
  final _HomeSort sort;
  final ValueChanged<_HomeSort> onChanged;
  final int visibleCount;
  final int totalCount;

  const _SortBar({
    required this.sort,
    required this.onChanged,
    required this.visibleCount,
    required this.totalCount,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Text(
          totalCount == visibleCount
              ? '전체 $totalCount개'
              : '$visibleCount / $totalCount개',
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: AppColors.textTertiary,
          ),
        ),
        const Spacer(),
        PopupMenuButton<_HomeSort>(
          initialValue: sort,
          onSelected: onChanged,
          color: AppColors.surface,
          surfaceTintColor: Colors.transparent,
          position: PopupMenuPosition.under,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          itemBuilder: (context) => [
            for (final option in _HomeSort.values)
              PopupMenuItem(
                value: option,
                child: Text(
                  _sortLabel(option),
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textSecondary,
                  ),
                ),
              ),
          ],
          child: Container(
            height: 34,
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: BoxDecoration(
              color: AppColors.neutralSoft,
              borderRadius: BorderRadius.circular(999),
              border: Border.all(color: AppColors.border),
            ),
            child: Row(
              children: [
                const Icon(Icons.swap_vert_rounded,
                    size: 16, color: AppColors.textTertiary),
                const SizedBox(width: 5),
                Text(
                  _sortLabel(sort),
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textSecondary,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _EmptyState extends StatelessWidget {
  final VoidCallback onAdd;
  const _EmptyState({required this.onAdd});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: Container(
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(20),
        ),
        padding: const EdgeInsets.symmetric(vertical: 80),
        child: Column(
          children: [
            const Text('💳', style: TextStyle(fontSize: 52)),
            const SizedBox(height: 16),
            const Text(
              '아직 구독이 없어요',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              '매달 빠져나가는 구독료,\n한눈에 확인해보세요',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                height: 1.5,
                fontWeight: FontWeight.w400,
                color: AppColors.textTertiary,
              ),
            ),
            const SizedBox(height: 24),
            GestureDetector(
              onTap: onAdd,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                height: 44,
                decoration: BoxDecoration(
                  color: AppColors.primary,
                  borderRadius: BorderRadius.circular(12),
                ),
                alignment: Alignment.center,
                child: const Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(Icons.add, size: 16, color: Colors.white),
                    SizedBox(width: 6),
                    Text(
                      '구독 추가하기',
                      style: TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _FloatingAddButton extends StatelessWidget {
  final VoidCallback onTap;
  const _FloatingAddButton({required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 56,
        height: 56,
        decoration: BoxDecoration(
          gradient: AppColors.primarySimpleGradient,
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color: AppColors.primary.withValues(alpha: 0.45),
              blurRadius: 24,
              offset: const Offset(0, 6),
            ),
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.08),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: const Icon(Icons.add, size: 26, color: Colors.white),
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  final String message;
  const _ErrorBanner({required this.message});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: _maxContentWidth),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
          child: Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: AppColors.dangerSoft,
              borderRadius: BorderRadius.circular(14),
              border:
                  Border.all(color: AppColors.danger.withValues(alpha: 0.25)),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.error_outline,
                    size: 18, color: AppColors.danger),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    message,
                    style: const TextStyle(
                      fontSize: 13,
                      height: 1.5,
                      color: AppColors.danger,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
