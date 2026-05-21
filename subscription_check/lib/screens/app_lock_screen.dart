import 'package:flutter/material.dart';

import '../services/app_preferences.dart';
import '../theme/app_theme.dart';
import '../widgets/pin_keypad.dart';

class AppLockScreen extends StatefulWidget {
  final VoidCallback onUnlocked;

  const AppLockScreen({super.key, required this.onUnlocked});

  @override
  State<AppLockScreen> createState() => _AppLockScreenState();
}

class _AppLockScreenState extends State<AppLockScreen> {
  String _pin = '';
  bool _hasError = false;
  bool _canUseBiometrics = false;
  bool _authInProgress = false;
  AppBiometricKind _biometricKind = AppBiometricKind.biometric;

  @override
  void initState() {
    super.initState();
    _loadBiometrics();
  }

  Future<void> _loadBiometrics() async {
    final enabled = await AppPreferences.isBiometricsEnabled();
    final canUse = enabled && await AppPreferences.canUseBiometrics();
    final kind = await AppPreferences.biometricKind();
    if (!mounted) return;
    setState(() {
      _canUseBiometrics = canUse;
      _biometricKind = kind;
    });
    if (canUse) {
      Future<void>.delayed(const Duration(milliseconds: 450), _authenticate);
    }
  }

  Future<void> _authenticate() async {
    if (_authInProgress || !_canUseBiometrics) return;
    setState(() => _authInProgress = true);
    final ok = await AppPreferences.authenticateWithBiometrics();
    if (!mounted) return;
    setState(() => _authInProgress = false);
    if (ok) widget.onUnlocked();
  }

  Future<void> _appendDigit(String digit) async {
    if (_pin.length >= 4) return;
    final next = '$_pin$digit';
    setState(() {
      _pin = next;
      _hasError = false;
    });

    if (next.length == 4) {
      final ok = await AppPreferences.verifyPin(next);
      if (!mounted) return;
      if (ok) {
        widget.onUnlocked();
      } else {
        setState(() => _hasError = true);
        await Future<void>.delayed(const Duration(milliseconds: 420));
        if (mounted) {
          setState(() {
            _pin = '';
            _hasError = false;
          });
        }
      }
    }
  }

  void _backspace() {
    if (_pin.isEmpty) return;
    setState(() {
      _pin = _pin.substring(0, _pin.length - 1);
      _hasError = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 460),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 28),
              child: Column(
                children: [
                  const Spacer(),
                  Container(
                    width: 62,
                    height: 62,
                    decoration: BoxDecoration(
                      gradient: AppColors.primaryGradient,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    alignment: Alignment.center,
                    child: Transform.rotate(
                      angle: -0.785,
                      child: const Icon(
                        Icons.content_cut,
                        size: 25,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  const SizedBox(height: 22),
                  const Text(
                    'SubCut 잠금 해제',
                    style: TextStyle(
                      fontSize: 22,
                      fontWeight: FontWeight.w800,
                      color: AppColors.textPrimary,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _hasError ? 'PIN 번호가 올바르지 않습니다' : 'PIN 번호를 입력하세요',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color:
                          _hasError ? AppColors.danger : AppColors.textTertiary,
                    ),
                  ),
                  const SizedBox(height: 30),
                  PinDots(filled: _pin.length, hasError: _hasError),
                  const Spacer(),
                  PinKeypad(
                    onDigit: _appendDigit,
                    onBackspace: _backspace,
                    showBiometric: _canUseBiometrics,
                    biometricKind: _biometricKind,
                    onBiometric: _authenticate,
                  ),
                  const SizedBox(height: 20),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
