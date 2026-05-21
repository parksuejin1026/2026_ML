import 'package:flutter/material.dart';

import '../services/app_preferences.dart';
import '../theme/app_theme.dart';
import 'app_lock_screen.dart';
import 'main_shell.dart';
import 'terms_screen.dart';

class AppGate extends StatefulWidget {
  const AppGate({super.key});

  @override
  State<AppGate> createState() => _AppGateState();
}

class _AppGateState extends State<AppGate> {
  bool _loading = true;
  bool _termsAccepted = false;
  bool _needsUnlock = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    final termsAccepted = await AppPreferences.isTermsAccepted();
    final lockEnabled = await AppPreferences.isLockEnabled();
    final pin = await AppPreferences.getPin();
    if (!mounted) return;
    setState(() {
      _termsAccepted = termsAccepted;
      _needsUnlock = lockEnabled && pin != null;
      _loading = false;
    });
  }

  Future<void> _acceptTerms() async {
    await AppPreferences.setTermsAccepted(true);
    if (!mounted) return;
    setState(() => _termsAccepted = true);
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(
        backgroundColor: AppColors.bg,
        body: Center(child: CircularProgressIndicator()),
      );
    }

    if (!_termsAccepted) {
      return TermsScreen(
        showAgreeButton: true,
        onAgree: _acceptTerms,
      );
    }

    if (_needsUnlock) {
      return AppLockScreen(
        onUnlocked: () => setState(() => _needsUnlock = false),
      );
    }

    return const MainShell();
  }
}
