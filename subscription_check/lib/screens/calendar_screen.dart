import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';

import '../models/subscription.dart';
import '../providers/subscription_provider.dart';
import '../theme/app_theme.dart';
import '../widgets/app_top_bar.dart';

const double _maxContentWidth = 460;
const int _minPickerYear = 1900;
const int _maxPickerYear = 2900;

class CalendarScreen extends StatefulWidget {
  const CalendarScreen({super.key});

  @override
  State<CalendarScreen> createState() => _CalendarScreenState();
}

class _CalendarScreenState extends State<CalendarScreen> {
  late DateTime _focused;
  late int _pickerYear;
  late int _pickerMonth;
  final LayerLink _monthPickerLink = LayerLink();
  OverlayEntry? _monthPickerEntry;
  bool _isMonthPickerOpen = false;

  @override
  void initState() {
    super.initState();
    final now = DateTime.now();
    _focused = DateTime(now.year, now.month);
    _pickerYear = _focused.year;
    _pickerMonth = _focused.month;
  }

  void _setFocused(DateTime value) {
    _removeMonthPicker(updateState: false);
    setState(() {
      _focused = DateTime(value.year, value.month);
      _pickerYear = _focused.year;
      _pickerMonth = _focused.month;
      _isMonthPickerOpen = false;
    });
  }

  @override
  void dispose() {
    _removeMonthPicker(updateState: false);
    super.dispose();
  }

  void _showDayEvents(DateTime date, List<_BillingEvent> events) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => _DayEventsSheet(date: date, events: events),
    );
  }

  void _toggleMonthPicker() {
    if (_isMonthPickerOpen) {
      _removeMonthPicker();
      return;
    }
    _showMonthPicker();
  }

  void _showMonthPicker() {
    _pickerYear = _focused.year;
    _pickerMonth = _focused.month;
    setState(() => _isMonthPickerOpen = true);

    _monthPickerEntry = OverlayEntry(
      builder: (context) {
        final screenWidth = MediaQuery.of(context).size.width;
        final popoverWidth = (screenWidth - 24).clamp(292.0, 336.0);

        return Stack(
          children: [
            Positioned.fill(
              child: GestureDetector(
                behavior: HitTestBehavior.translucent,
                onTap: _removeMonthPicker,
              ),
            ),
            CompositedTransformFollower(
              link: _monthPickerLink,
              showWhenUnlinked: false,
              targetAnchor: Alignment.bottomRight,
              followerAnchor: Alignment.topRight,
              offset: const Offset(0, 10),
              child: Material(
                color: Colors.transparent,
                child: _MonthPopover(
                  width: popoverWidth,
                  initialYear: _pickerYear,
                  initialMonth: _pickerMonth,
                  onYearChanged: (value) => _pickerYear = value,
                  onMonthChanged: (value) => _pickerMonth = value,
                  onCancel: _removeMonthPicker,
                  onDone: _applyPickedMonth,
                ),
              ),
            ),
          ],
        );
      },
    );
    Overlay.of(context).insert(_monthPickerEntry!);
  }

  void _removeMonthPicker({bool updateState = true}) {
    _monthPickerEntry?.remove();
    _monthPickerEntry = null;
    if (updateState && mounted) {
      setState(() => _isMonthPickerOpen = false);
    }
  }

  void _applyPickedMonth() {
    _setFocused(DateTime(_pickerYear, _pickerMonth));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: Consumer<SubscriptionProvider>(
        builder: (context, provider, _) {
          final events = _buildEvents(provider.items, _focused);
          final total = events.fold<int>(
              0, (sum, e) => sum + e.subscription.effectiveMonthlyCost);

          return Column(
            children: [
              _Header(
                focused: _focused,
                isPickerOpen: _isMonthPickerOpen,
                monthPickerLink: _monthPickerLink,
                onPickMonth: _toggleMonthPicker,
                onToday: () {
                  final now = DateTime.now();
                  _setFocused(DateTime(now.year, now.month));
                },
              ),
              _MonthSummary(total: total, count: events.length),
              Expanded(
                child: CustomScrollView(
                  physics: const AlwaysScrollableScrollPhysics(),
                  slivers: [
                    SliverToBoxAdapter(child: _UpcomingSection(events: events)),
                    SliverToBoxAdapter(
                      child: _CalendarCard(
                        focused: _focused,
                        events: events,
                        onDayTap: _showDayEvents,
                      ),
                    ),
                    const SliverToBoxAdapter(child: SizedBox(height: 138)),
                  ],
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
  final DateTime focused;
  final bool isPickerOpen;
  final LayerLink monthPickerLink;
  final VoidCallback onPickMonth;
  final VoidCallback onToday;

  const _Header({
    required this.focused,
    required this.isPickerOpen,
    required this.monthPickerLink,
    required this.onPickMonth,
    required this.onToday,
  });

  @override
  Widget build(BuildContext context) {
    return AppTopBar(
      padding: const EdgeInsets.fromLTRB(24, 0, 12, 0),
      child: Row(
        children: [
          const Text(
            '일정',
            style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
              letterSpacing: 0,
            ),
          ),
          const Spacer(),
          Semantics(
            button: true,
            label: '월 선택',
            child: CompositedTransformTarget(
              link: monthPickerLink,
              child: GestureDetector(
                onTap: onPickMonth,
                behavior: HitTestBehavior.opaque,
                child: Container(
                  height: 36,
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  decoration: BoxDecoration(
                    color: AppColors.neutralChip,
                    borderRadius: BorderRadius.circular(999),
                  ),
                  alignment: Alignment.center,
                  child: Row(
                    children: [
                      Text(
                        DateFormat('yyyy년 M월').format(focused),
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w800,
                          color: AppColors.textSecondary,
                        ),
                      ),
                      const SizedBox(width: 4),
                      AnimatedRotation(
                        turns: isPickerOpen ? 0.5 : 0,
                        duration: const Duration(milliseconds: 180),
                        child: const Icon(
                          CupertinoIcons.chevron_down,
                          size: 13,
                          color: AppColors.textTertiary,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          const SizedBox(width: 6),
          Semantics(
            button: true,
            label: '오늘로 이동',
            child: GestureDetector(
              onTap: onToday,
              child: Container(
                height: 36,
                padding: const EdgeInsets.symmetric(horizontal: 11),
                decoration: BoxDecoration(
                  color: AppColors.primarySoft,
                  borderRadius: BorderRadius.circular(999),
                ),
                alignment: Alignment.center,
                child: const Text(
                  '오늘',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w800,
                    color: AppColors.primaryDark,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MonthSummary extends StatelessWidget {
  final int total;
  final int count;

  const _MonthSummary({required this.total, required this.count});

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
                  const Text(
                    '이번 달 결제 예정',
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w500,
                      color: AppColors.textTertiary,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Flexible(
                        child: Text(
                          formatKRW(total),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          textAlign: TextAlign.left,
                          style: const TextStyle(
                            fontSize: 40,
                            fontWeight: FontWeight.w800,
                            color: AppColors.textPrimary,
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
                    count > 0 ? '$count건의 결제가 예정되어 있어요' : '예정된 결제가 없어요',
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

class _UpcomingSection extends StatelessWidget {
  final List<_BillingEvent> events;

  const _UpcomingSection({required this.events});

  @override
  Widget build(BuildContext context) {
    final sorted = [...events]..sort((a, b) => a.date.compareTo(b.date));

    return Container(
      color: AppColors.bg,
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: _maxContentWidth),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 0),
            child: Container(
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(18),
              ),
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '결제 예정 목록',
                    style: TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w800,
                      color: AppColors.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 6),
                  if (sorted.isEmpty)
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: 18),
                      child: Center(
                        child: Text(
                          '결제일이 등록된 구독이 없어요',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: AppColors.textDisabled,
                          ),
                        ),
                      ),
                    )
                  else
                    for (final event in sorted.take(4))
                      _BillingRow(event: event, compact: true),
                  if (sorted.length > 4)
                    Padding(
                      padding: const EdgeInsets.only(top: 4, bottom: 8),
                      child: Text(
                        '외 ${sorted.length - 4}건은 캘린더에서 확인하세요',
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: AppColors.textTertiary,
                        ),
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

class _CalendarCard extends StatelessWidget {
  final DateTime focused;
  final List<_BillingEvent> events;
  final void Function(DateTime date, List<_BillingEvent> events) onDayTap;

  const _CalendarCard({
    required this.focused,
    required this.events,
    required this.onDayTap,
  });

  @override
  Widget build(BuildContext context) {
    final first = DateTime(focused.year, focused.month, 1);
    final daysInMonth = DateTime(focused.year, focused.month + 1, 0).day;
    final leading = first.weekday % 7;
    final cells = leading + daysInMonth;
    final rows = (cells / 7).ceil();

    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: _maxContentWidth),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 14, 16, 0),
          child: Container(
            padding: const EdgeInsets.fromLTRB(16, 18, 16, 16),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(20),
            ),
            child: Column(
              children: [
                const _WeekHeader(),
                const SizedBox(height: 8),
                GridView.builder(
                  itemCount: rows * 7,
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 7,
                    mainAxisSpacing: 6,
                    crossAxisSpacing: 4,
                    childAspectRatio: 0.9,
                  ),
                  itemBuilder: (context, index) {
                    final day = index - leading + 1;
                    if (day < 1 || day > daysInMonth) {
                      return const SizedBox.shrink();
                    }
                    final date = DateTime(focused.year, focused.month, day);
                    final dayEvents =
                        events.where((e) => e.date.day == day).toList();
                    return _DayCell(
                      day: day,
                      events: dayEvents,
                      isToday: _isSameDay(DateTime.now(), date),
                      onTap: () => onDayTap(date, dayEvents),
                    );
                  },
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _WeekHeader extends StatelessWidget {
  const _WeekHeader();

  @override
  Widget build(BuildContext context) {
    const labels = ['일', '월', '화', '수', '목', '금', '토'];
    return Row(
      children: [
        for (final label in labels)
          Expanded(
            child: Center(
              child: Text(
                label,
                style: TextStyle(
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                  color:
                      label == '일' ? AppColors.danger : AppColors.textDisabled,
                ),
              ),
            ),
          ),
      ],
    );
  }
}

class _DayCell extends StatelessWidget {
  final int day;
  final List<_BillingEvent> events;
  final bool isToday;
  final VoidCallback onTap;

  const _DayCell({
    required this.day,
    required this.events,
    required this.isToday,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final hasEvent = events.isNotEmpty;
    return Semantics(
      button: true,
      label: hasEvent ? '$day일, 결제 ${events.length}건' : '$day일',
      child: GestureDetector(
        onTap: onTap,
        behavior: HitTestBehavior.opaque,
        child: Container(
          decoration: BoxDecoration(
            color: hasEvent ? AppColors.primarySoftBg : Colors.transparent,
            borderRadius: BorderRadius.circular(12),
            border: isToday ? Border.all(color: AppColors.primary) : null,
          ),
          padding: const EdgeInsets.symmetric(vertical: 5),
          child: Column(
            children: [
              Text(
                '$day',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w800,
                  color: hasEvent
                      ? AppColors.primaryDark
                      : AppColors.textSecondary,
                ),
              ),
              const Spacer(),
              if (hasEvent)
                Wrap(
                  alignment: WrapAlignment.center,
                  spacing: 2,
                  runSpacing: 2,
                  children: [
                    for (final event in events.take(3))
                      Container(
                        width: 5,
                        height: 5,
                        decoration: BoxDecoration(
                          color: event.subscription.isAnnual
                              ? AppColors.danger
                              : AppColors.primary,
                          shape: BoxShape.circle,
                        ),
                      ),
                  ],
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _BillingRow extends StatelessWidget {
  final _BillingEvent event;
  final bool compact;

  const _BillingRow({required this.event, this.compact = false});

  @override
  Widget build(BuildContext context) {
    final sub = event.subscription;
    return Padding(
      padding: EdgeInsets.symmetric(vertical: compact ? 8 : 11),
      child: Row(
        children: [
          Container(
            width: compact ? 34 : 38,
            height: compact ? 34 : 38,
            decoration: BoxDecoration(
              color:
                  sub.isAnnual ? AppColors.dangerSoft : AppColors.primarySoft,
              borderRadius: BorderRadius.circular(12),
            ),
            alignment: Alignment.center,
            child:
                Text(sub.emoji ?? '📦', style: const TextStyle(fontSize: 19)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  sub.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: AppColors.textPrimary,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  '${DateFormat('M월 d일').format(event.date)} · ${sub.isAnnual ? '연간 갱신' : '월간 결제'}',
                  style: const TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    color: AppColors.textTertiary,
                  ),
                ),
              ],
            ),
          ),
          Text(
            '${formatKRW(sub.effectiveMonthlyCost)}원',
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
            ),
          ),
        ],
      ),
    );
  }
}

class _DayEventsSheet extends StatelessWidget {
  final DateTime date;
  final List<_BillingEvent> events;

  const _DayEventsSheet({required this.date, required this.events});

  @override
  Widget build(BuildContext context) {
    final sorted = [...events]..sort((a, b) => a.date.compareTo(b.date));

    return SafeArea(
      top: false,
      child: Container(
        margin: const EdgeInsets.all(10),
        padding: const EdgeInsets.fromLTRB(20, 10, 20, 20),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(26),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 36,
                height: 4,
                decoration: BoxDecoration(
                  color: AppColors.neutralChipDark,
                  borderRadius: BorderRadius.circular(999),
                ),
              ),
            ),
            const SizedBox(height: 18),
            Text(
              DateFormat('yyyy년 M월 d일').format(date),
              style: const TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w800,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              sorted.isEmpty ? '결제 예정 구독이 없어요' : '${sorted.length}건 결제 예정',
              style: const TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: AppColors.textTertiary,
              ),
            ),
            const SizedBox(height: 14),
            if (sorted.isEmpty)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 28),
                decoration: BoxDecoration(
                  color: AppColors.neutralSoft,
                  borderRadius: BorderRadius.circular(16),
                ),
                alignment: Alignment.center,
                child: const Text(
                  '선택한 날짜에 결제 예정인 구독이 없습니다.',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: AppColors.textDisabled,
                  ),
                ),
              )
            else
              for (final event in sorted) _BillingRow(event: event),
          ],
        ),
      ),
    );
  }
}

class _MonthPopover extends StatefulWidget {
  final double width;
  final int initialYear;
  final int initialMonth;
  final ValueChanged<int> onYearChanged;
  final ValueChanged<int> onMonthChanged;
  final VoidCallback onCancel;
  final VoidCallback onDone;

  const _MonthPopover({
    required this.width,
    required this.initialYear,
    required this.initialMonth,
    required this.onYearChanged,
    required this.onMonthChanged,
    required this.onCancel,
    required this.onDone,
  });

  @override
  State<_MonthPopover> createState() => _MonthPopoverState();
}

class _MonthPopoverState extends State<_MonthPopover> {
  late final FixedExtentScrollController _yearController;
  late final FixedExtentScrollController _monthController;

  @override
  void initState() {
    super.initState();
    _yearController = FixedExtentScrollController(
      initialItem: (widget.initialYear - _minPickerYear)
          .clamp(0, _maxPickerYear - _minPickerYear)
          .toInt(),
    );
    _monthController =
        FixedExtentScrollController(initialItem: widget.initialMonth - 1);
  }

  @override
  void dispose() {
    _yearController.dispose();
    _monthController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: widget.width,
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Positioned(
            top: -8,
            right: 30,
            child: CustomPaint(
              size: const Size(18, 9),
              painter: _PopoverArrowPainter(),
            ),
          ),
          Container(
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(22),
              border: Border.all(color: AppColors.border),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.16),
                  blurRadius: 24,
                  offset: const Offset(0, 12),
                ),
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.06),
                  blurRadius: 4,
                  offset: const Offset(0, 1),
                ),
              ],
            ),
            clipBehavior: Clip.antiAlias,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(8, 6, 8, 0),
                  child: Row(
                    children: [
                      CupertinoButton(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        onPressed: widget.onCancel,
                        child: const Text('취소'),
                      ),
                      const Spacer(),
                      const Text(
                        '월 선택',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w800,
                          color: AppColors.textPrimary,
                        ),
                      ),
                      const Spacer(),
                      CupertinoButton(
                        padding: const EdgeInsets.symmetric(horizontal: 12),
                        onPressed: widget.onDone,
                        child: const Text(
                          '완료',
                          style: TextStyle(fontWeight: FontWeight.w800),
                        ),
                      ),
                    ],
                  ),
                ),
                SizedBox(
                  height: 190,
                  child: Row(
                    children: [
                      Expanded(
                        child: CupertinoPicker(
                          scrollController: _yearController,
                          itemExtent: 38,
                          useMagnifier: true,
                          magnification: 1.08,
                          onSelectedItemChanged: (index) {
                            widget.onYearChanged(_minPickerYear + index);
                          },
                          children: [
                            for (var year = _minPickerYear;
                                year <= _maxPickerYear;
                                year++)
                              Center(child: Text('$year년')),
                          ],
                        ),
                      ),
                      Expanded(
                        child: CupertinoPicker(
                          scrollController: _monthController,
                          itemExtent: 38,
                          useMagnifier: true,
                          magnification: 1.08,
                          onSelectedItemChanged: (index) {
                            widget.onMonthChanged(index + 1);
                          },
                          children: [
                            for (var month = 1; month <= 12; month++)
                              Center(child: Text('$month월')),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 10),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _PopoverArrowPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final path = Path()
      ..moveTo(size.width / 2, 0)
      ..lineTo(size.width, size.height)
      ..lineTo(0, size.height)
      ..close();
    final paint = Paint()..color = AppColors.surface;
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _BillingEvent {
  final DateTime date;
  final Subscription subscription;

  const _BillingEvent({required this.date, required this.subscription});
}

List<_BillingEvent> _buildEvents(List<Subscription> items, DateTime focused) {
  return [
    for (final item in items)
      if (_billingDateForMonth(item, focused) != null)
        _BillingEvent(
          date: _billingDateForMonth(item, focused)!,
          subscription: item,
        ),
  ]..sort((a, b) => a.date.compareTo(b.date));
}

bool _isSameDay(DateTime a, DateTime b) {
  return a.year == b.year && a.month == b.month && a.day == b.day;
}

DateTime? _billingDateForMonth(Subscription item, DateTime focused) {
  final nextBillingAt = item.nextBillingAt;
  if (nextBillingAt != null &&
      nextBillingAt.year == focused.year &&
      nextBillingAt.month == focused.month) {
    return nextBillingAt;
  }

  final billingDay = item.billingDay;
  if (billingDay == null) return null;

  final daysInMonth = DateTime(focused.year, focused.month + 1, 0).day;
  return DateTime(
    focused.year,
    focused.month,
    billingDay.clamp(1, daysInMonth),
  );
}
