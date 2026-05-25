// ignore_for_file: invalid_use_of_protected_member

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import '../data/service_presets.dart';
import '../models/subscription.dart';
import '../providers/subscription_provider.dart';
import '../theme/app_theme.dart';

const double _maxContentWidth = 460;
const int _totalSteps = 2;

const _categoryTabs = [
  _CategoryTab(key: 'all', label: '전체'),
  _CategoryTab(key: 'Video', label: '영상'),
  _CategoryTab(key: 'Music', label: '음악'),
  _CategoryTab(key: 'Cloud', label: '생활'),
  _CategoryTab(key: 'Education', label: '교육'),
  _CategoryTab(key: 'Game', label: '게임'),
  _CategoryTab(key: 'Fitness', label: '운동'),
];

class _CategoryTab {
  final String key;
  final String label;
  const _CategoryTab({required this.key, required this.label});
}

const _types = [
  'Music',
  'Video',
  'Education',
  'Fitness',
  'Game',
  'News',
  'Cloud'
];

class _FreqOption {
  final UseFrequency value;
  final String label;
  final String emoji;
  const _FreqOption(this.value, this.label, this.emoji);
}

const _freqOptions = [
  _FreqOption(UseFrequency.rare, '거의 안 씀', '🥱'),
  _FreqOption(UseFrequency.monthly, '월 1~2회', '📅'),
  _FreqOption(UseFrequency.weekly, '주 1~2회', '📆'),
  _FreqOption(UseFrequency.frequent, '거의 매일', '🔥'),
];

class _RecencyOption {
  final LastUseRecency value;
  final String label;
  const _RecencyOption(this.value, this.label);
}

const _recencyOptions = [
  _RecencyOption(LastUseRecency.over30d, '한 달 넘음'),
  _RecencyOption(LastUseRecency.between7and30d, '일주일~한달'),
  _RecencyOption(LastUseRecency.between1and7d, '이번 주'),
  _RecencyOption(LastUseRecency.under1d, '오늘'),
];

class AddSubscriptionSheet extends StatefulWidget {
  const AddSubscriptionSheet({super.key});

  @override
  State<AddSubscriptionSheet> createState() => _AddSubscriptionSheetState();
}

class _AddSubscriptionSheetState extends State<AddSubscriptionSheet> {
  int _step = 0;

  // Step 1 state
  ServicePreset? _selectedPreset;
  bool _editingPreset = false;
  final _nameCtrl = TextEditingController();
  final _costCtrl = TextEditingController();
  final _searchCtrl = TextEditingController();
  String _type = _types[0];
  String _activeTab = 'all';

  // Step 2 state
  UseFrequency _freq = UseFrequency.weekly;
  LastUseRecency _recency = LastUseRecency.under1d;
  int _necessity = 3;

  @override
  void dispose() {
    _nameCtrl.dispose();
    _costCtrl.dispose();
    _searchCtrl.dispose();
    super.dispose();
  }

  List<ServicePreset> get _filteredPresets {
    final q = _searchCtrl.text.trim().toLowerCase();
    return servicePresets.where((p) {
      final matchQuery = q.isEmpty || p.name.toLowerCase().contains(q);
      final matchTab = _activeTab == 'all' || p.type == _activeTab;
      return matchQuery && matchTab;
    }).toList();
  }

  void _selectPreset(ServicePreset p) {
    setState(() {
      _selectedPreset = p;
      _nameCtrl.text = p.name;
      _costCtrl.text = p.monthlyCost.toString();
      _type = p.type;
      _searchCtrl.clear();
      _editingPreset = false;
    });
  }

  void _clearPreset() {
    setState(() {
      _selectedPreset = null;
      _nameCtrl.clear();
      _costCtrl.clear();
      _type = _types[0];
      _editingPreset = false;
    });
  }

  void _refreshForm() => setState(() {});

  void _setActiveTab(String value) =>
      setState(() => _activeTab = value);

  void _setEditingPreset(bool value) =>
      setState(() => _editingPreset = value);

  void _setType(String value) => setState(() => _type = value);

  void _setFrequency(UseFrequency value) =>
      setState(() => _freq = value);

  void _setRecency(LastUseRecency value) =>
      setState(() => _recency = value);

  void _setNecessity(int value) => setState(() => _necessity = value);

  bool get _canGoNext {
    if (_step == 0) {
      return _nameCtrl.text.trim().isNotEmpty &&
          (int.tryParse(_costCtrl.text.trim()) ?? 0) > 0;
    }
    return true;
  }

  void _goNext() {
    if (!_canGoNext) return;
    if (_step < _totalSteps - 1) {
      setState(() => _step = _step + 1);
    }
  }

  void _goBack() {
    if (_step > 0) {
      setState(() => _step = _step - 1);
    } else {
      Navigator.of(context).pop();
    }
  }

  void _submit() {
    final name = _nameCtrl.text.trim();
    final cost = int.tryParse(_costCtrl.text.trim()) ?? 0;
    if (name.isEmpty || cost <= 0) return;

    final provider = context.read<SubscriptionProvider>();
    provider.addSubscription(Subscription(
      id: provider.nextId,
      name: name,
      type: _type,
      monthlyCost: cost,
      useFrequency: _freq,
      lastUseRecency: _recency,
      perceivedNecessity: _necessity,
      replacementAvailable: false,
      isAnnual: _selectedPreset?.isAnnual ?? false,
      remainingMonths: 0,
      discountAmount: _selectedPreset?.discountAmount ?? 0,
      emoji: _selectedPreset?.emoji,
    ));
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final stepTitles = ['서비스 선택', '사용 패턴'];
    final safePaddingBottom = MediaQuery.of(context).viewInsets.bottom;

    return Material(
      color: AppColors.surface,
      child: SafeArea(
        bottom: false,
        child: Column(
          children: [
            _Header(
              title: stepTitles[_step],
              onBack: _goBack,
              isFirstStep: _step == 0,
              step: _step,
              totalSteps: _totalSteps,
            ),
            Expanded(
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: _maxContentWidth),
                  child: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 260),
                    transitionBuilder: (child, animation) => FadeTransition(
                      opacity: animation,
                      child: SlideTransition(
                        position: Tween<Offset>(
                                begin: const Offset(0.08, 0), end: Offset.zero)
                            .animate(animation),
                        child: child,
                      ),
                    ),
                    child: _step == 0
                        ? _Step1(
                            key: const ValueKey('step1'),
                            state: this,
                          )
                        : _Step2(
                            key: const ValueKey('step2'),
                            state: this,
                          ),
                  ),
                ),
              ),
            ),
            _BottomCta(
              isLast: _step >= _totalSteps - 1,
              canGoNext: _canGoNext,
              onNext: _goNext,
              onSubmit: _submit,
              extraBottomInset: safePaddingBottom,
            ),
          ],
        ),
      ),
    );
  }
}

class _Header extends StatelessWidget {
  final String title;
  final VoidCallback onBack;
  final bool isFirstStep;
  final int step;
  final int totalSteps;
  const _Header({
    required this.title,
    required this.onBack,
    required this.isFirstStep,
    required this.step,
    required this.totalSteps,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: _maxContentWidth),
        child: Column(
          children: [
            SizedBox(
              height: 56,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Row(
                  children: [
                    GestureDetector(
                      onTap: onBack,
                      child: Container(
                        width: 40,
                        height: 40,
                        alignment: Alignment.center,
                        child: Icon(
                          isFirstStep ? Icons.close : Icons.chevron_left,
                          size: 22,
                          color: AppColors.textSecondary,
                        ),
                      ),
                    ),
                    Expanded(
                      child: Center(
                        child: Text(
                          title,
                          style: const TextStyle(
                            fontSize: 17,
                            fontWeight: FontWeight.w700,
                            color: AppColors.textPrimary,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 40),
                  ],
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(999),
                    child: Container(
                      height: 3,
                      color: AppColors.divider,
                      alignment: Alignment.centerLeft,
                      child: TweenAnimationBuilder<double>(
                        tween: Tween(begin: 0, end: (step + 1) / totalSteps),
                        duration: const Duration(milliseconds: 280),
                        curve: Curves.easeOutCubic,
                        builder: (_, v, __) => FractionallySizedBox(
                          widthFactor: v,
                          child: Container(
                            decoration: BoxDecoration(
                              color: AppColors.primary,
                              borderRadius: BorderRadius.circular(999),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '${step + 1} / $totalSteps',
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                      color: AppColors.textTertiary,
                    ),
                  ),
                  const SizedBox(height: 4),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _BottomCta extends StatelessWidget {
  final bool isLast;
  final bool canGoNext;
  final VoidCallback onNext;
  final VoidCallback onSubmit;
  final double extraBottomInset;

  const _BottomCta({
    required this.isLast,
    required this.canGoNext,
    required this.onNext,
    required this.onSubmit,
    required this.extraBottomInset,
  });

  @override
  Widget build(BuildContext context) {
    final bottomPad = MediaQuery.of(context).padding.bottom;
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 12,
            offset: const Offset(0, -1),
          ),
        ],
      ),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: _maxContentWidth),
          child: Padding(
            padding: EdgeInsets.fromLTRB(
                20, 12, 20, 16 + bottomPad + extraBottomInset),
            child: GestureDetector(
              onTap: isLast ? onSubmit : (canGoNext ? onNext : null),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 160),
                height: 54,
                decoration: BoxDecoration(
                  color: (isLast || canGoNext)
                      ? AppColors.primary
                      : AppColors.neutralChipDark,
                  borderRadius: BorderRadius.circular(16),
                ),
                alignment: Alignment.center,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      isLast ? '구독 추가하기' : '다음',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w700,
                        color: (isLast || canGoNext)
                            ? Colors.white
                            : AppColors.textDisabled,
                      ),
                    ),
                    if (!isLast) ...[
                      const SizedBox(width: 6),
                      Icon(Icons.arrow_forward,
                          size: 18,
                          color: canGoNext
                              ? Colors.white
                              : AppColors.textDisabled),
                    ],
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _Step1 extends StatelessWidget {
  final _AddSubscriptionSheetState state;
  const _Step1({super.key, required this.state});

  @override
  Widget build(BuildContext context) {
    final presets = state._filteredPresets;
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '어떤 서비스를\n구독하고 계신가요?',
            style: TextStyle(
              fontSize: 22,
              height: 1.3,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            '서비스를 선택하면 자동으로 채워져요',
            style: TextStyle(
              fontSize: 14,
              color: AppColors.textTertiary,
            ),
          ),
          const SizedBox(height: 18),
          _SearchField(
            controller: state._searchCtrl,
            onChanged: (_) => state._refreshForm(),
          ),
          const SizedBox(height: 12),
          _CategoryTabsBar(
            active: state._activeTab,
            onSelect: state._setActiveTab,
          ),
          const SizedBox(height: 12),
          _PresetGrid(
            presets: presets,
            selected: state._selectedPreset,
            showManualName: state._nameCtrl.text.isNotEmpty,
            onSelect: state._selectPreset,
            onManual: state._clearPreset,
          ),
          const SizedBox(height: 16),
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 220),
            child: state._selectedPreset != null && !state._editingPreset
                ? _PresetInfoCard(
                    key: const ValueKey('preset-info'),
                    preset: state._selectedPreset!,
                    name: state._nameCtrl.text,
                    type: state._type,
                    cost: int.tryParse(state._costCtrl.text) ?? 0,
                    onEdit: () => state._setEditingPreset(true),
                  )
                : _ManualForm(
                    key: ValueKey(state._selectedPreset == null
                        ? 'manual'
                        : 'preset-edit'),
                    title:
                        state._selectedPreset != null ? '서비스 정보 수정' : '직접 입력',
                    nameCtrl: state._nameCtrl,
                    costCtrl: state._costCtrl,
                    type: state._type,
                    onTypeChanged: state._setType,
                    showConfirm: state._editingPreset,
                    onConfirm: () => state._setEditingPreset(false),
                    onChanged: state._refreshForm,
                  ),
          ),
        ],
      ),
    );
  }
}

class _SearchField extends StatelessWidget {
  final TextEditingController controller;
  final ValueChanged<String> onChanged;
  const _SearchField({required this.controller, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 44,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      decoration: BoxDecoration(
        color: AppColors.neutralChip,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        children: [
          const Icon(Icons.search, size: 16, color: AppColors.textTertiary),
          const SizedBox(width: 10),
          Expanded(
            child: TextField(
              controller: controller,
              onChanged: onChanged,
              decoration: const InputDecoration(
                hintText: '서비스 이름 검색',
                hintStyle: TextStyle(
                  fontSize: 14,
                  color: AppColors.textPlaceholder,
                  fontWeight: FontWeight.w500,
                ),
                isDense: true,
                border: InputBorder.none,
                contentPadding: EdgeInsets.zero,
              ),
              style: const TextStyle(
                fontSize: 14,
                color: AppColors.textPrimary,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _CategoryTabsBar extends StatelessWidget {
  final String active;
  final ValueChanged<String> onSelect;
  const _CategoryTabsBar({required this.active, required this.onSelect});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 34,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: _categoryTabs.length,
        separatorBuilder: (_, __) => const SizedBox(width: 6),
        itemBuilder: (_, i) {
          final tab = _categoryTabs[i];
          final isActive = active == tab.key;
          return GestureDetector(
            onTap: () => onSelect(tab.key),
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 160),
              padding: const EdgeInsets.symmetric(horizontal: 14),
              alignment: Alignment.center,
              decoration: BoxDecoration(
                color: isActive ? AppColors.textPrimary : AppColors.neutralChip,
                borderRadius: BorderRadius.circular(999),
              ),
              child: Text(
                tab.label,
                style: TextStyle(
                  fontSize: 13,
                  fontWeight: FontWeight.w600,
                  color: isActive ? Colors.white : AppColors.textTertiary,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}

class _PresetGrid extends StatelessWidget {
  final List<ServicePreset> presets;
  final ServicePreset? selected;
  final bool showManualName;
  final ValueChanged<ServicePreset> onSelect;
  final VoidCallback onManual;

  const _PresetGrid({
    required this.presets,
    required this.selected,
    required this.showManualName,
    required this.onSelect,
    required this.onManual,
  });

  @override
  Widget build(BuildContext context) {
    // Arrange into columns of 3, horizontally scrollable
    final all = [...presets, null];
    final columns = <List<ServicePreset?>>[];
    for (var i = 0; i < all.length; i += 3) {
      columns.add(all.sublist(i, (i + 3).clamp(0, all.length)));
    }

    return SizedBox(
      height: 144,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: columns.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (_, i) {
          final col = columns[i];
          return Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              for (var j = 0; j < col.length; j++) ...[
                if (j > 0) const SizedBox(height: 8),
                if (col[j] == null)
                  _ManualChip(
                    active: selected == null && showManualName,
                    onTap: onManual,
                  )
                else
                  _PresetChip(
                    preset: col[j]!,
                    selected: selected?.name == col[j]!.name,
                    onTap: () => onSelect(col[j]!),
                  ),
              ],
            ],
          );
        },
      ),
    );
  }
}

class _PresetChip extends StatelessWidget {
  final ServicePreset preset;
  final bool selected;
  final VoidCallback onTap;
  const _PresetChip({
    required this.preset,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        height: 40,
        padding: const EdgeInsets.symmetric(horizontal: 14),
        decoration: BoxDecoration(
          color: selected ? AppColors.primary : AppColors.neutralSoft,
          border: Border.all(
            color: selected ? AppColors.primary : AppColors.border,
            width: selected ? 1.5 : 1,
          ),
          borderRadius: BorderRadius.circular(999),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(preset.emoji, style: const TextStyle(fontSize: 15)),
            const SizedBox(width: 8),
            Text(
              preset.name,
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: selected ? Colors.white : const Color(0xFF333D4B),
              ),
            ),
            const SizedBox(width: 6),
            Text(
              '${preset.monthlyCost.toString().replaceAllMapped(
                    RegExp(r'(\d)(?=(\d{3})+(?!\d))'),
                    (m) => '${m[1]},',
                  )}원',
              style: TextStyle(
                fontSize: 11,
                fontWeight: FontWeight.w500,
                color: selected
                    ? Colors.white.withValues(alpha: 0.7)
                    : AppColors.textTertiary,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ManualChip extends StatelessWidget {
  final bool active;
  final VoidCallback onTap;
  const _ManualChip({required this.active, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 160),
        height: 40,
        padding: const EdgeInsets.symmetric(horizontal: 14),
        decoration: BoxDecoration(
          color: active ? AppColors.primarySoftBg : AppColors.neutralSoft,
          border: Border.all(
            color: active ? AppColors.primary : AppColors.border,
            width: active ? 1.5 : 1,
          ),
          borderRadius: BorderRadius.circular(999),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: const [
            Text('✏️', style: TextStyle(fontSize: 14)),
            SizedBox(width: 8),
            Text(
              '직접 입력',
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: Color(0xFF333D4B),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _PresetInfoCard extends StatelessWidget {
  final ServicePreset preset;
  final String name;
  final String type;
  final int cost;
  final VoidCallback onEdit;

  const _PresetInfoCard({
    super.key,
    required this.preset,
    required this.name,
    required this.type,
    required this.cost,
    required this.onEdit,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.divider),
      ),
      padding: const EdgeInsets.all(20),
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppColors.neutralSoft,
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFFF2F4F6), Color(0xFFE9ECEF)],
                ),
                borderRadius: BorderRadius.circular(12),
              ),
              alignment: Alignment.center,
              child: Text(preset.emoji, style: const TextStyle(fontSize: 22)),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    name,
                    style: const TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: AppColors.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    '$type · 월 ${formatKRW(cost)}원',
                    style: const TextStyle(
                      fontSize: 12,
                      color: AppColors.textTertiary,
                    ),
                  ),
                ],
              ),
            ),
            GestureDetector(
              onTap: onEdit,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: AppColors.primarySoft,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Text(
                  '수정',
                  style: TextStyle(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: AppColors.primary,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ManualForm extends StatelessWidget {
  final String title;
  final TextEditingController nameCtrl;
  final TextEditingController costCtrl;
  final String type;
  final ValueChanged<String> onTypeChanged;
  final bool showConfirm;
  final VoidCallback onConfirm;
  final VoidCallback onChanged;

  const _ManualForm({
    super.key,
    required this.title,
    required this.nameCtrl,
    required this.costCtrl,
    required this.type,
    required this.onTypeChanged,
    required this.showConfirm,
    required this.onConfirm,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.divider),
      ),
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color: AppColors.textPrimary,
            ),
          ),
          const SizedBox(height: 12),
          _FloatingInput(
            label: '서비스 이름',
            placeholder: '예: 넷플릭스, 멜론',
            controller: nameCtrl,
            onChanged: (_) => onChanged(),
          ),
          const SizedBox(height: 12),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: _TypeDropdown(type: type, onChanged: onTypeChanged),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _FloatingInput(
                  label: '월 구독료',
                  placeholder: '0',
                  controller: costCtrl,
                  suffix: '원',
                  keyboardType: TextInputType.number,
                  formatters: [FilteringTextInputFormatter.digitsOnly],
                  onChanged: (_) => onChanged(),
                ),
              ),
            ],
          ),
          if (showConfirm) ...[
            const SizedBox(height: 16),
            GestureDetector(
              onTap: onConfirm,
              child: Container(
                height: 44,
                decoration: BoxDecoration(
                  color: AppColors.primary,
                  borderRadius: BorderRadius.circular(12),
                ),
                alignment: Alignment.center,
                child: const Text(
                  '확인',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _FloatingInput extends StatelessWidget {
  final String label;
  final String placeholder;
  final TextEditingController controller;
  final String? suffix;
  final TextInputType? keyboardType;
  final List<TextInputFormatter>? formatters;
  final ValueChanged<String>? onChanged;

  const _FloatingInput({
    required this.label,
    required this.placeholder,
    required this.controller,
    this.suffix,
    this.keyboardType,
    this.formatters,
    this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
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
        const SizedBox(height: 6),
        Container(
          height: 48,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          decoration: BoxDecoration(
            color: AppColors.neutralChip,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            children: [
              Expanded(
                child: TextField(
                  controller: controller,
                  keyboardType: keyboardType,
                  inputFormatters: formatters,
                  onChanged: onChanged,
                  decoration: InputDecoration(
                    hintText: placeholder,
                    hintStyle: const TextStyle(
                      fontSize: 15,
                      color: AppColors.textPlaceholder,
                      fontWeight: FontWeight.w500,
                    ),
                    isDense: true,
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.zero,
                  ),
                  style: const TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w500,
                    color: AppColors.textPrimary,
                  ),
                ),
              ),
              if (suffix != null)
                Padding(
                  padding: const EdgeInsets.only(left: 6),
                  child: Text(
                    suffix!,
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: AppColors.textTertiary,
                    ),
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }
}

class _TypeDropdown extends StatelessWidget {
  final String type;
  final ValueChanged<String> onChanged;
  const _TypeDropdown({required this.type, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          '유형',
          style: TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w600,
            color: AppColors.textTertiary,
          ),
        ),
        const SizedBox(height: 6),
        Container(
          height: 48,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          decoration: BoxDecoration(
            color: AppColors.neutralChip,
            borderRadius: BorderRadius.circular(12),
          ),
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: type,
              isExpanded: true,
              icon: const Icon(Icons.keyboard_arrow_down,
                  size: 16, color: AppColors.textTertiary),
              style: const TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w500,
                color: AppColors.textPrimary,
              ),
              dropdownColor: AppColors.surface,
              items: _types
                  .map((t) => DropdownMenuItem(value: t, child: Text(t)))
                  .toList(),
              onChanged: (v) {
                if (v != null) onChanged(v);
              },
            ),
          ),
        ),
      ],
    );
  }
}

class _Step2 extends StatelessWidget {
  final _AddSubscriptionSheetState state;
  const _Step2({super.key, required this.state});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '얼마나 자주\n사용하시나요?',
            style: TextStyle(
              fontSize: 22,
              height: 1.3,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            '사용 패턴을 기반으로 분석해드려요',
            style: TextStyle(
              fontSize: 14,
              color: AppColors.textTertiary,
            ),
          ),
          const SizedBox(height: 28),
          const _Step2Label('사용 빈도'),
          const SizedBox(height: 10),
          _FreqGrid(
            value: state._freq,
            onSelect: state._setFrequency,
          ),
          const SizedBox(height: 24),
          const _Step2Label('마지막 사용'),
          const SizedBox(height: 10),
          _RecencyRow(
            value: state._recency,
            onSelect: state._setRecency,
          ),
          const SizedBox(height: 24),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const _Step2Label('필요도'),
              Text(
                '${state._necessity}/5',
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w800,
                  color: AppColors.primary,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          _NecessityRow(
            value: state._necessity,
            onSelect: state._setNecessity,
          ),
          const SizedBox(height: 8),
          const Padding(
            padding: EdgeInsets.symmetric(horizontal: 4),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  '전혀 안 필요해',
                  style: TextStyle(
                    fontSize: 11,
                    color: AppColors.textPlaceholder,
                  ),
                ),
                Text(
                  '없으면 안 돼!',
                  style: TextStyle(
                    fontSize: 11,
                    color: AppColors.textPlaceholder,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _Step2Label extends StatelessWidget {
  final String text;
  const _Step2Label(this.text);

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: const TextStyle(
        fontSize: 13,
        fontWeight: FontWeight.w600,
        color: AppColors.textTertiary,
      ),
    );
  }
}

class _FreqGrid extends StatelessWidget {
  final UseFrequency value;
  final ValueChanged<UseFrequency> onSelect;
  const _FreqGrid({required this.value, required this.onSelect});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Row(
          children: [
            _freqTile(_freqOptions[0]),
            const SizedBox(width: 8),
            _freqTile(_freqOptions[1]),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            _freqTile(_freqOptions[2]),
            const SizedBox(width: 8),
            _freqTile(_freqOptions[3]),
          ],
        ),
      ],
    );
  }

  Widget _freqTile(_FreqOption o) {
    final sel = value == o.value;
    return Expanded(
      child: GestureDetector(
        onTap: () => onSelect(o.value),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 160),
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: sel ? AppColors.primary : AppColors.neutralChip,
            borderRadius: BorderRadius.circular(14),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(o.emoji, style: const TextStyle(fontSize: 18)),
              const SizedBox(height: 6),
              Text(
                o.label,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: sel ? Colors.white : AppColors.textSecondary,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _RecencyRow extends StatelessWidget {
  final LastUseRecency value;
  final ValueChanged<LastUseRecency> onSelect;
  const _RecencyRow({required this.value, required this.onSelect});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        for (var i = 0; i < _recencyOptions.length; i++) ...[
          if (i > 0) const SizedBox(width: 8),
          Expanded(
            child: GestureDetector(
              onTap: () => onSelect(_recencyOptions[i].value),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 160),
                height: 44,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: value == _recencyOptions[i].value
                      ? AppColors.primary
                      : AppColors.neutralChip,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Text(
                  _recencyOptions[i].label,
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: value == _recencyOptions[i].value
                        ? Colors.white
                        : AppColors.textSecondary,
                  ),
                ),
              ),
            ),
          ),
        ],
      ],
    );
  }
}

class _NecessityRow extends StatelessWidget {
  final int value;
  final ValueChanged<int> onSelect;
  const _NecessityRow({required this.value, required this.onSelect});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        for (var n = 1; n <= 5; n++) ...[
          if (n > 1) const SizedBox(width: 6),
          Expanded(
            child: GestureDetector(
              onTap: () => onSelect(n),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 160),
                height: 48,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  gradient: n <= value
                      ? LinearGradient(
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight,
                          colors: n <= 2
                              ? const [
                                  AppColors.primaryLight,
                                  AppColors.primary,
                                ]
                              : const [
                                  AppColors.primary,
                                  Color(0xFF2272EB),
                                ],
                        )
                      : null,
                  color: n <= value ? null : AppColors.neutralChip,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Text(
                  '$n',
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                    color: n <= value ? Colors.white : AppColors.textDisabled,
                  ),
                ),
              ),
            ),
          ),
        ],
      ],
    );
  }
}
