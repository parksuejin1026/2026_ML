import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../config/app_brand.dart';
import '../providers/subscription_provider.dart';
import '../services/app_preferences.dart';
import '../theme/app_theme.dart';
import '../widgets/app_top_bar.dart';
import '../widgets/pin_keypad.dart';
import 'terms_screen.dart';

const double _maxContentWidth = 460;

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _billingAlert = true;
  bool _weeklyReport = false;
  bool _lockEnabled = false;
  bool _biometricsEnabled = false;
  AppBiometricKind _biometricKind = AppBiometricKind.biometric;

  @override
  void initState() {
    super.initState();
    _loadLockSettings();
  }

  Future<void> _loadLockSettings() async {
    final lock = await AppPreferences.isLockEnabled();
    final bio = await AppPreferences.isBiometricsEnabled();
    final kind = await AppPreferences.biometricKind();
    if (!mounted) return;
    setState(() {
      _lockEnabled = lock;
      _biometricsEnabled = bio;
      _biometricKind = kind;
    });
  }

  Future<void> _toggleLock(bool value) async {
    if (!value) {
      await AppPreferences.setLockEnabled(false);
      await AppPreferences.setBiometricsEnabled(false);
      if (!mounted) return;
      setState(() {
        _lockEnabled = false;
        _biometricsEnabled = false;
      });
      return;
    }

    final pin = await _showPinSetupSheet();
    if (pin == null) return;
    await AppPreferences.setPin(pin);
    await AppPreferences.setLockEnabled(true);
    if (!mounted) return;
    setState(() => _lockEnabled = true);
  }

  Future<void> _changePin() async {
    final pin = await _showPinSetupSheet();
    if (pin == null) return;
    await AppPreferences.setPin(pin);
    if (!mounted) return;
    _showSnack('PIN 번호가 변경되었습니다.');
  }

  Future<void> _toggleBiometrics(bool value) async {
    if (!value) {
      await AppPreferences.setBiometricsEnabled(false);
      if (!mounted) return;
      setState(() => _biometricsEnabled = false);
      return;
    }

    final canUse = await AppPreferences.canUseBiometrics();
    if (!canUse) {
      if (!mounted) return;
      _showSnack('이 기기에서 Face ID 또는 Touch ID를 사용할 수 없습니다.');
      return;
    }
    final ok = await AppPreferences.authenticateWithBiometrics();
    if (!ok) return;
    final kind = await AppPreferences.biometricKind();
    await AppPreferences.setBiometricsEnabled(true);
    if (!mounted) return;
    setState(() {
      _biometricsEnabled = true;
      _biometricKind = kind;
    });
  }

  String get _biometricTitle {
    return switch (_biometricKind) {
      AppBiometricKind.faceId => 'Face ID',
      AppBiometricKind.touchId => 'Touch ID',
      AppBiometricKind.biometric => '기기 생체인증',
    };
  }

  IconData get _biometricIcon {
    return switch (_biometricKind) {
      AppBiometricKind.faceId => Icons.face_retouching_natural_rounded,
      AppBiometricKind.touchId => Icons.fingerprint_rounded,
      AppBiometricKind.biometric => Icons.fingerprint_rounded,
    };
  }

  Future<String?> _showPinSetupSheet() {
    return showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => const _PinSetupSheet(),
    );
  }

  void _showSnack(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        behavior: SnackBarBehavior.floating,
        backgroundColor: AppColors.textPrimary,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: Consumer<SubscriptionProvider>(
        builder: (context, provider, _) {
          return Column(
            children: [
              const _Header(),
              _ProfileCard(
                count: provider.items.length,
                total: provider.totalMonthlyCost,
                saveable: provider.saveableCost,
              ),
              Expanded(
                child: CustomScrollView(
                  physics: const AlwaysScrollableScrollPhysics(),
                  slivers: [
                    SliverToBoxAdapter(
                      child: _Section(
                        title: '알림',
                        children: [
                          _SwitchRow(
                            icon: CupertinoIcons.bell_fill,
                            title: '결제 전 알림',
                            subtitle: '예정 결제 3일 전에 알려줘요',
                            value: _billingAlert,
                            onChanged: (v) => setState(() => _billingAlert = v),
                          ),
                          _SwitchRow(
                            icon: CupertinoIcons.chart_bar_alt_fill,
                            title: '주간 리포트',
                            subtitle: '구독료 변화와 추천을 요약해요',
                            value: _weeklyReport,
                            onChanged: (v) => setState(() => _weeklyReport = v),
                          ),
                        ],
                      ),
                    ),
                    SliverToBoxAdapter(
                      child: _Section(
                        title: '데이터',
                        children: [
                          _ActionRow(
                            icon: CupertinoIcons.arrow_down_doc_fill,
                            title: '데이터 내보내기',
                            subtitle: 'CSV 내보내기 기능 자리',
                            onTap: () => _showStub(context, '데이터 내보내기'),
                          ),
                          _ActionRow(
                            icon: CupertinoIcons.arrow_2_circlepath,
                            title: '구독 목록 새로고침',
                            subtitle: '서버에 저장된 목록을 다시 불러와요',
                            onTap: provider.loadFromServer,
                          ),
                          _ActionRow(
                            icon: CupertinoIcons.delete_solid,
                            title: '삭제된 구독 보관함',
                            subtitle: '해지 기록 관리 기능 자리',
                            destructive: true,
                            onTap: () => _showStub(context, '보관함'),
                          ),
                        ],
                      ),
                    ),
                    SliverToBoxAdapter(
                      child: _Section(
                        title: '보안',
                        children: [
                          _SwitchRow(
                            icon: CupertinoIcons.lock_fill,
                            title: '앱 잠금',
                            subtitle: 'PIN 또는 기기 인증으로 보호해요',
                            value: _lockEnabled,
                            onChanged: _toggleLock,
                          ),
                          if (_lockEnabled)
                            _ActionRow(
                              icon: CupertinoIcons.number,
                              title: 'PIN 번호 변경',
                              subtitle: '4자리 PIN을 다시 설정해요',
                              onTap: _changePin,
                            ),
                          if (_lockEnabled)
                            _SwitchRow(
                              icon: _biometricIcon,
                              title: _biometricTitle,
                              subtitle: '기기 인식 방식으로 잠금을 해제해요',
                              value: _biometricsEnabled,
                              onChanged: _toggleBiometrics,
                            ),
                        ],
                      ),
                    ),
                    SliverToBoxAdapter(
                      child: _Section(
                        title: '정보',
                        children: [
                          _ActionRow(
                            icon: CupertinoIcons.doc_text_fill,
                            title: '이용약관 및 개인정보 안내',
                            subtitle: '데이터 처리와 모델 추천 안내',
                            onTap: () => Navigator.of(context).push(
                              MaterialPageRoute(
                                builder: (_) => const TermsScreen(),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SliverToBoxAdapter(child: _AppInfo()),
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

  void _showStub(BuildContext context, String title) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('$title 기능은 백엔드 연동 시 활성화됩니다.'),
        behavior: SnackBarBehavior.floating,
        backgroundColor: AppColors.textPrimary,
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
          '설정',
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

class _ProfileCard extends StatelessWidget {
  final int count;
  final int total;
  final int saveable;

  const _ProfileCard({
    required this.count,
    required this.total,
    required this.saveable,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      color: AppColors.surface,
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: _maxContentWidth),
          child: SizedBox(
            height: 154,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(24, 18, 24, 22),
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: AppColors.neutralSoft,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(color: AppColors.border),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Expanded(
                          child: Text(
                            AppBrand.displayName,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: TextStyle(
                              fontSize: 18,
                              fontWeight: FontWeight.w800,
                              color: AppColors.textPrimary,
                            ),
                          ),
                        ),
                        if (saveable > 0) ...[
                          const SizedBox(width: 10),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 10,
                              vertical: 6,
                            ),
                            decoration: BoxDecoration(
                              color: AppColors.dangerSoft,
                              borderRadius: BorderRadius.circular(999),
                            ),
                            child: Text(
                              '${formatKRW(saveable)}원 절약 가능',
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w800,
                                color: AppColors.danger,
                              ),
                            ),
                          ),
                        ],
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '$count개 구독 · 월 ${formatKRW(total)}원',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textSecondary,
                      ),
                    ),
                    if (saveable > 0) ...[
                      const SizedBox(height: 4),
                      Text(
                        '해지 후보 기준 월 ${formatKRW(saveable)}원까지 줄일 수 있어요',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: AppColors.textTertiary,
                        ),
                      ),
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

class _Section extends StatelessWidget {
  final String title;
  final List<Widget> children;

  const _Section({required this.title, required this.children});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: _maxContentWidth),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 14, 16, 0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(4, 0, 4, 8),
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w800,
                    color: AppColors.textTertiary,
                  ),
                ),
              ),
              Container(
                decoration: BoxDecoration(
                  color: AppColors.surface,
                  borderRadius: BorderRadius.circular(20),
                ),
                clipBehavior: Clip.antiAlias,
                child: Column(
                  children: [
                    for (var i = 0; i < children.length; i++) ...[
                      children[i],
                      if (i < children.length - 1)
                        const Padding(
                          padding: EdgeInsets.only(left: 62),
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
            ],
          ),
        ),
      ),
    );
  }
}

class _SwitchRow extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final bool value;
  final ValueChanged<bool> onChanged;

  const _SwitchRow({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 14, 12),
      child: Row(
        children: [
          _SettingIcon(icon: icon, color: AppColors.primary),
          const SizedBox(width: 12),
          Expanded(child: _SettingText(title: title, subtitle: subtitle)),
          CupertinoSwitch(
            value: value,
            activeTrackColor: AppColors.primary,
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }
}

class _ActionRow extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final bool destructive;
  final VoidCallback onTap;

  const _ActionRow({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.onTap,
    this.destructive = false,
  });

  @override
  Widget build(BuildContext context) {
    final color = destructive ? AppColors.danger : AppColors.primary;
    return GestureDetector(
      onTap: onTap,
      behavior: HitTestBehavior.opaque,
      child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 14, 14, 14),
        child: Row(
          children: [
            _SettingIcon(icon: icon, color: color),
            const SizedBox(width: 12),
            Expanded(child: _SettingText(title: title, subtitle: subtitle)),
            const Icon(
              CupertinoIcons.chevron_right,
              size: 16,
              color: AppColors.textDisabled,
            ),
          ],
        ),
      ),
    );
  }
}

class _SettingIcon extends StatelessWidget {
  final IconData icon;
  final Color color;

  const _SettingIcon({required this.icon, required this.color});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 34,
      height: 34,
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(10),
      ),
      alignment: Alignment.center,
      child: Icon(icon, size: 17, color: color),
    );
  }
}

class _SettingText extends StatelessWidget {
  final String title;
  final String subtitle;

  const _SettingText({required this.title, required this.subtitle});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            fontSize: 14,
            fontWeight: FontWeight.w700,
            color: AppColors.textPrimary,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          subtitle,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: const TextStyle(
            fontSize: 12,
            fontWeight: FontWeight.w500,
            color: AppColors.textTertiary,
          ),
        ),
      ],
    );
  }
}

class _PinSetupSheet extends StatefulWidget {
  const _PinSetupSheet();

  @override
  State<_PinSetupSheet> createState() => _PinSetupSheetState();
}

class _PinSetupSheetState extends State<_PinSetupSheet> {
  String _first = '';
  String _current = '';
  bool _confirming = false;
  bool _hasError = false;

  String get _title => _confirming ? 'PIN 번호 확인' : 'PIN 번호 설정';
  String get _subtitle {
    if (_hasError) return 'PIN 번호가 일치하지 않습니다';
    return _confirming ? '같은 PIN을 한 번 더 입력하세요' : '앱 잠금에 사용할 4자리 PIN을 입력하세요';
  }

  void _appendDigit(String digit) {
    if (_current.length >= 4) return;
    final next = '$_current$digit';
    setState(() {
      _current = next;
      _hasError = false;
    });

    if (next.length == 4) {
      if (!_confirming) {
        Future<void>.delayed(const Duration(milliseconds: 180), () {
          if (!mounted) return;
          setState(() {
            _first = next;
            _current = '';
            _confirming = true;
          });
        });
        return;
      }

      if (next == _first) {
        Navigator.of(context).pop(next);
      } else {
        Future<void>.delayed(const Duration(milliseconds: 180), () {
          if (!mounted) return;
          setState(() {
            _current = '';
            _first = '';
            _confirming = false;
            _hasError = true;
          });
        });
      }
    }
  }

  void _backspace() {
    if (_current.isEmpty) return;
    setState(() {
      _current = _current.substring(0, _current.length - 1);
      _hasError = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      top: false,
      child: Container(
        margin: const EdgeInsets.all(10),
        padding: const EdgeInsets.fromLTRB(20, 10, 20, 20),
        decoration: BoxDecoration(
          color: AppColors.bg,
          borderRadius: BorderRadius.circular(26),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 36,
              height: 4,
              decoration: BoxDecoration(
                color: AppColors.neutralChipDark,
                borderRadius: BorderRadius.circular(999),
              ),
            ),
            const SizedBox(height: 22),
            Text(
              _title,
              style: const TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.w800,
                color: AppColors.textPrimary,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              _subtitle,
              style: TextStyle(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: _hasError ? AppColors.danger : AppColors.textTertiary,
              ),
            ),
            const SizedBox(height: 24),
            PinDots(filled: _current.length, hasError: _hasError),
            const SizedBox(height: 24),
            PinKeypad(
              onDigit: _appendDigit,
              onBackspace: _backspace,
            ),
          ],
        ),
      ),
    );
  }
}

class _AppInfo extends StatelessWidget {
  const _AppInfo();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: _maxContentWidth),
        child: const Padding(
          padding: EdgeInsets.fromLTRB(16, 18, 16, 0),
          child: Text(
            '${AppBrand.displayName} 1.0.0 · 프론트엔드 프로토타입',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: AppColors.textDisabled,
            ),
          ),
        ),
      ),
    );
  }
}
