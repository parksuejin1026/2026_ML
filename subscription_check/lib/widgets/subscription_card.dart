import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../models/subscription.dart';
import '../providers/subscription_provider.dart';
import '../theme/app_theme.dart';

class SubscriptionCard extends StatefulWidget {
  final Subscription subscription;
  final ChurnResult? result;
  final VoidCallback onDelete;
  final ValueChanged<Subscription> onUpdate;
  final ValueChanged<bool>? onFeedback;

  const SubscriptionCard({
    super.key,
    required this.subscription,
    this.result,
    required this.onDelete,
    required this.onUpdate,
    this.onFeedback,
  });

  @override
  State<SubscriptionCard> createState() => _SubscriptionCardState();
}

class _SubscriptionCardState extends State<SubscriptionCard> {
  bool _expanded = false;
  bool _editing = false;
  bool _controllersReady = false;

  // Edit state
  late bool _replacement;
  late bool _isAnnual;
  late TextEditingController _remainingCtrl;
  late TextEditingController _discountCtrl;

  @override
  void initState() {
    super.initState();
    _resetEditState();
  }

  @override
  void didUpdateWidget(covariant SubscriptionCard oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!_editing && oldWidget.subscription != widget.subscription) {
      _resetEditState();
    }
  }

  void _resetEditState() {
    final s = widget.subscription;
    _replacement = s.replacementAvailable;
    _isAnnual = s.isAnnual;
    final remainingText = s.remainingMonths.toString();
    final discountText =
        s.discountAmount > 0 ? s.discountAmount.toString() : '';
    if (_controllersReady) {
      _remainingCtrl.text = remainingText;
      _discountCtrl.text = discountText;
    } else {
      _remainingCtrl = TextEditingController(text: remainingText);
      _discountCtrl = TextEditingController(text: discountText);
      _controllersReady = true;
    }
  }

  @override
  void dispose() {
    _remainingCtrl.dispose();
    _discountCtrl.dispose();
    super.dispose();
  }

  void _save() {
    widget.onUpdate(widget.subscription.copyWith(
      replacementAvailable: _replacement,
      isAnnual: _isAnnual,
      remainingMonths: double.tryParse(_remainingCtrl.text.trim()) ?? 0,
      discountAmount: int.tryParse(_discountCtrl.text.trim()) ?? 0,
    ));
    setState(() => _editing = false);
  }

  void _cancel() {
    setState(() {
      _resetEditState();
      _editing = false;
    });
  }

  Future<void> _confirmDelete() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppColors.surface,
        surfaceTintColor: Colors.transparent,
        title: const Text('구독을 삭제할까요?'),
        content: Text(
          '${widget.subscription.name} 구독 정보와 분석 결과가 목록에서 제거됩니다.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('취소'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(true),
            style: TextButton.styleFrom(foregroundColor: AppColors.danger),
            child: const Text('삭제'),
          ),
        ],
      ),
    );
    if (confirmed == true) widget.onDelete();
  }

  @override
  Widget build(BuildContext context) {
    final sub = widget.subscription;
    final result = widget.result;
    final effective = sub.effectiveMonthlyCost;

    return InkWell(
      onTap: () => setState(() => _expanded = !_expanded),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                _IconBox(emoji: sub.emoji ?? '📦', result: result),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Flexible(
                            child: Text(
                              sub.name,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.w600,
                                color: AppColors.textPrimary,
                              ),
                            ),
                          ),
                          if (result != null) ...[
                            const SizedBox(width: 8),
                            _StatusBadge(isChurn: result.isChurnCandidate),
                          ],
                        ],
                      ),
                      const SizedBox(height: 2),
                      Text(
                        '${freqShortLabel(sub.useFrequency)} · 최근 ${recencyShortLabel(sub.lastUseRecency)}',
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                          color: AppColors.textTertiary,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      '${formatKRW(effective)}원',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary,
                      ),
                    ),
                    if (sub.discountAmount > 0)
                      Text(
                        '${formatKRW(sub.monthlyCost)}원',
                        style: const TextStyle(
                          fontSize: 11,
                          color: AppColors.textDisabled,
                          decoration: TextDecoration.lineThrough,
                        ),
                      ),
                  ],
                ),
                const SizedBox(width: 6),
                Icon(
                  _expanded
                      ? Icons.keyboard_arrow_up
                      : Icons.keyboard_arrow_down,
                  size: 18,
                  color: AppColors.textPlaceholder,
                ),
              ],
            ),
            AnimatedSize(
              duration: const Duration(milliseconds: 220),
              curve: Curves.easeInOut,
              child: _expanded
                  ? _buildExpanded(context)
                  : const SizedBox(width: double.infinity, height: 0),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildExpanded(BuildContext context) {
    final sub = widget.subscription;
    final result = widget.result;
    return Padding(
      padding: const EdgeInsets.only(top: 12, left: 62),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (result != null) _ResultCard(result: result),
          if (result != null) const SizedBox(height: 12),
          if (result != null && widget.onFeedback != null) ...[
            _FeedbackRow(
              // 서버에 영구 저장된 값을 우선 사용, 없으면 세션 내 상태로 fallback
              currentFeedback: sub.lastFeedbackKept ?? result.userFeedbackKept,
              onSelect: widget.onFeedback!,
            ),
            const SizedBox(height: 12),
          ],
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: [
              _InfoTag(label: sub.type),
              _InfoTag(label: sub.isAnnual ? '연간 구독' : '월간 구독'),
              if (sub.remainingMonths > 0)
                _InfoTag(
                    label:
                        '${sub.remainingMonths.toStringAsFixed(sub.remainingMonths.truncateToDouble() == sub.remainingMonths ? 0 : 1)}개월 남음'),
              _InfoTag(label: '필요도 ${sub.perceivedNecessity}/5'),
              if (sub.replacementAvailable)
                const _InfoTag(label: '대체 가능', accent: true),
            ],
          ),
          const SizedBox(height: 12),
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 200),
            child: _editing ? _buildEditSection() : _buildEditButton(),
          ),
          GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTap: _confirmDelete,
            child: const Padding(
              padding: EdgeInsets.symmetric(vertical: 6),
              child: Row(
                children: [
                  Icon(Icons.delete_outline, size: 14, color: AppColors.danger),
                  SizedBox(width: 6),
                  Text(
                    '구독 삭제',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: AppColors.danger,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEditButton() {
    return GestureDetector(
      key: const ValueKey('edit-btn'),
      behavior: HitTestBehavior.opaque,
      onTap: () => setState(() => _editing = true),
      child: const Padding(
        padding: EdgeInsets.symmetric(vertical: 6),
        child: Row(
          children: [
            Icon(Icons.tune, size: 14, color: AppColors.primary),
            SizedBox(width: 6),
            Text(
              '상세 설정',
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: AppColors.primary,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEditSection() {
    return Container(
      key: const ValueKey('edit-section'),
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.neutralSoft,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _toggleRow('대체 서비스 있음', _replacement,
              (v) => setState(() => _replacement = v)),
          const SizedBox(height: 14),
          _toggleRow('연간 구독', _isAnnual, (v) => setState(() => _isAnnual = v)),
          const SizedBox(height: 14),
          _inputRow('남은 기간', _remainingCtrl, '0', '개월',
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true)),
          const SizedBox(height: 14),
          _inputRow('할인/환급액', _discountCtrl, '0', '원',
              keyboardType: TextInputType.number,
              formatters: [FilteringTextInputFormatter.digitsOnly]),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: _ActionButton(
                  label: '저장',
                  onTap: _save,
                  background: AppColors.primary,
                  foreground: Colors.white,
                ),
              ),
              const SizedBox(width: 8),
              _ActionButton(
                label: '취소',
                onTap: _cancel,
                background: AppColors.neutralChipDark,
                foreground: AppColors.textMuted,
                paddingH: 16,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _toggleRow(String label, bool value, ValueChanged<bool> onChanged) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: AppColors.textSecondary,
          ),
        ),
        _TossToggle(value: value, onChanged: onChanged),
      ],
    );
  }

  Widget _inputRow(
    String label,
    TextEditingController ctrl,
    String hint,
    String suffix, {
    TextInputType? keyboardType,
    List<TextInputFormatter>? formatters,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: AppColors.textTertiary,
          ),
        ),
        const SizedBox(height: 4),
        Container(
          height: 40,
          padding: const EdgeInsets.symmetric(horizontal: 12),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(10),
          ),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  controller: ctrl,
                  keyboardType: keyboardType,
                  inputFormatters: formatters,
                  decoration: InputDecoration(
                    hintText: hint,
                    hintStyle: const TextStyle(
                      fontSize: 14,
                      color: AppColors.textPlaceholder,
                    ),
                    isDense: true,
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.zero,
                  ),
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                    color: AppColors.textPrimary,
                  ),
                ),
              ),
              Text(
                suffix,
                style: const TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textTertiary,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class _IconBox extends StatelessWidget {
  final String emoji;
  final ChurnResult? result;
  const _IconBox({required this.emoji, required this.result});

  @override
  Widget build(BuildContext context) {
    Gradient gradient;
    if (result == null) {
      gradient = const LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: [Color(0xFFF2F4F6), Color(0xFFE9ECEF)],
      );
    } else if (result!.isChurnCandidate) {
      gradient = const LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: [Color(0xFFFFF5F5), Color(0xFFFFE8E8)],
      );
    } else {
      gradient = const LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: [Color(0xFFE8F7F0), Color(0xFFD3F1E4)],
      );
    }
    return Container(
      width: 48,
      height: 48,
      decoration: BoxDecoration(
        gradient: gradient,
        borderRadius: BorderRadius.circular(14),
      ),
      alignment: Alignment.center,
      child: Text(emoji, style: const TextStyle(fontSize: 24)),
    );
  }
}

class _StatusBadge extends StatelessWidget {
  final bool isChurn;
  const _StatusBadge({required this.isChurn});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: isChurn ? AppColors.dangerSofter : AppColors.successSoft,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        isChurn ? '해지 추천' : '유지',
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w700,
          color: isChurn ? AppColors.danger : AppColors.success,
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final ChurnResult result;
  const _ResultCard({required this.result});

  @override
  Widget build(BuildContext context) {
    final isChurn = result.isChurnCandidate;
    final bgGradient = LinearGradient(
      begin: Alignment.topLeft,
      end: Alignment.bottomRight,
      colors: isChurn
          ? const [AppColors.dangerTint, Color(0xFFFFF2F2)]
          : const [AppColors.successTint, AppColors.successSofter],
    );

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: bgGradient,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            result.reason,
            style: const TextStyle(
              fontSize: 13,
              height: 1.6,
              color: AppColors.textSecondary,
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              const Text(
                '확신도',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: AppColors.textTertiary,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(999),
                  child: Container(
                    height: 6,
                    color: Colors.white.withValues(alpha: 0.6),
                    alignment: Alignment.centerLeft,
                    child: TweenAnimationBuilder<double>(
                      tween: Tween(begin: 0, end: result.confidence),
                      duration: const Duration(milliseconds: 800),
                      curve: Curves.easeOut,
                      builder: (_, v, __) => FractionallySizedBox(
                        widthFactor: v,
                        child: Container(
                          decoration: BoxDecoration(
                            gradient: LinearGradient(
                              colors: isChurn
                                  ? const [
                                      Color(0xFFFF8A8A),
                                      AppColors.danger,
                                    ]
                                  : const [
                                      AppColors.successLight,
                                      AppColors.success,
                                    ],
                            ),
                            borderRadius: BorderRadius.circular(999),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Text(
                '${(result.confidence * 100).toStringAsFixed(0)}%',
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                  color: isChurn ? AppColors.danger : AppColors.success,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _InfoTag extends StatelessWidget {
  final String label;
  final bool accent;
  const _InfoTag({required this.label, this.accent = false});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: accent ? AppColors.dangerSoft : AppColors.neutralChip,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w600,
          color: accent ? AppColors.danger : AppColors.textMuted,
        ),
      ),
    );
  }
}

class _TossToggle extends StatelessWidget {
  final bool value;
  final ValueChanged<bool> onChanged;
  const _TossToggle({required this.value, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => onChanged(!value),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 220),
        width: 46,
        height: 28,
        padding: const EdgeInsets.all(2),
        decoration: BoxDecoration(
          color: value ? AppColors.primary : const Color(0xFFD1D6DB),
          borderRadius: BorderRadius.circular(999),
        ),
        alignment: value ? Alignment.centerRight : Alignment.centerLeft,
        child: Container(
          width: 24,
          height: 24,
          decoration: BoxDecoration(
            color: Colors.white,
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: Colors.black.withValues(alpha: 0.12),
                blurRadius: 4,
                offset: const Offset(0, 1),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _FeedbackRow extends StatelessWidget {
  final bool? currentFeedback;
  final ValueChanged<bool> onSelect;

  const _FeedbackRow({
    required this.currentFeedback,
    required this.onSelect,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.neutralSoft,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '실제 결정은 어떻게 하셨나요?',
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: AppColors.textTertiary,
            ),
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              Expanded(
                child: _FeedbackButton(
                  label: '유지함',
                  selected: currentFeedback == true,
                  onTap: () => onSelect(true),
                  accent: AppColors.success,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: _FeedbackButton(
                  label: '해지함',
                  selected: currentFeedback == false,
                  onTap: () => onSelect(false),
                  accent: AppColors.danger,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _FeedbackButton extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;
  final Color accent;

  const _FeedbackButton({
    required this.label,
    required this.selected,
    required this.onTap,
    required this.accent,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        height: 36,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: selected ? accent : Colors.white,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(
            color: selected ? accent : const Color(0xFFE5E8EB),
          ),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w700,
            color: selected ? Colors.white : AppColors.textMuted,
          ),
        ),
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  final String label;
  final VoidCallback onTap;
  final Color background;
  final Color foreground;
  final double paddingH;
  const _ActionButton({
    required this.label,
    required this.onTap,
    required this.background,
    required this.foreground,
    this.paddingH = 0,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 38,
        padding: EdgeInsets.symmetric(horizontal: paddingH),
        decoration: BoxDecoration(
          color: background,
          borderRadius: BorderRadius.circular(10),
        ),
        alignment: Alignment.center,
        child: Text(
          label,
          style: TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w700,
            color: foreground,
          ),
        ),
      ),
    );
  }
}
