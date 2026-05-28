import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/app_preferences.dart';
import '../theme/app_theme.dart';

class PinDots extends StatelessWidget {
  final int length;
  final int filled;
  final bool hasError;

  const PinDots({
    super.key,
    this.length = 4,
    required this.filled,
    this.hasError = false,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        for (var i = 0; i < length; i++)
          AnimatedContainer(
            duration: const Duration(milliseconds: 140),
            width: 14,
            height: 14,
            margin: const EdgeInsets.symmetric(horizontal: 7),
            decoration: BoxDecoration(
              color: i < filled
                  ? (hasError ? AppColors.danger : AppColors.textPrimary)
                  : AppColors.neutralChipDark,
              shape: BoxShape.circle,
            ),
          ),
      ],
    );
  }
}

class PinKeypad extends StatelessWidget {
  final ValueChanged<String> onDigit;
  final VoidCallback onBackspace;
  final VoidCallback? onBiometric;
  final bool showBiometric;
  final AppBiometricKind biometricKind;

  const PinKeypad({
    super.key,
    required this.onDigit,
    required this.onBackspace,
    this.onBiometric,
    this.showBiometric = false,
    this.biometricKind = AppBiometricKind.biometric,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        for (final row in const [
          ['1', '2', '3'],
          ['4', '5', '6'],
          ['7', '8', '9'],
        ])
          _KeypadRow(
            children: [
              for (final digit in row)
                _KeyButton(label: digit, onTap: () => onDigit(digit)),
            ],
          ),
        _KeypadRow(
          children: [
            _IconKeyButton(
              enabled: showBiometric,
              onTap: onBiometric,
              semanticsLabel: '생체인증',
              child: BiometricIcon(
                kind: biometricKind,
                size: 25,
                color: showBiometric
                    ? AppColors.textSecondary
                    : Colors.transparent,
              ),
            ),
            _KeyButton(label: '0', onTap: () => onDigit('0')),
            _IconKeyButton(
              enabled: true,
              onTap: onBackspace,
              semanticsLabel: '한 글자 지우기',
              child: const Icon(CupertinoIcons.delete_left),
            ),
          ],
        ),
      ],
    );
  }
}

class BiometricIcon extends StatelessWidget {
  final AppBiometricKind kind;
  final double size;
  final Color color;

  const BiometricIcon({
    super.key,
    required this.kind,
    this.size = 22,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final icon = switch (kind) {
      AppBiometricKind.faceId => Icons.face_retouching_natural_rounded,
      AppBiometricKind.touchId => Icons.fingerprint_rounded,
      AppBiometricKind.biometric => Icons.fingerprint_rounded,
    };

    return Icon(icon, size: size, color: color);
  }
}

class _KeypadRow extends StatelessWidget {
  final List<Widget> children;

  const _KeypadRow({required this.children});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          for (final child in children)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 9),
              child: child,
            ),
        ],
      ),
    );
  }
}

class _KeyButton extends StatelessWidget {
  final String label;
  final VoidCallback onTap;

  const _KeyButton({required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: '$label 입력',
      child: GestureDetector(
        onTap: () {
          HapticFeedback.lightImpact();
          onTap();
        },
        child: Container(
          width: 72,
          height: 56,
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(18),
            border: Border.all(color: AppColors.border),
          ),
          alignment: Alignment.center,
          child: Text(
            label,
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w700,
              color: AppColors.textPrimary,
            ),
          ),
        ),
      ),
    );
  }
}

class _IconKeyButton extends StatelessWidget {
  final Widget child;
  final bool enabled;
  final VoidCallback? onTap;
  final String semanticsLabel;

  const _IconKeyButton({
    required this.child,
    required this.enabled,
    required this.semanticsLabel,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      enabled: enabled,
      label: semanticsLabel,
      child: GestureDetector(
        onTap: enabled
            ? () {
                HapticFeedback.selectionClick();
                onTap?.call();
              }
            : null,
        child: Container(
          width: 72,
          height: 56,
          alignment: Alignment.center,
          child: IconTheme(
            data: IconThemeData(
              size: 25,
              color: enabled ? AppColors.textSecondary : Colors.transparent,
            ),
            child: child,
          ),
        ),
      ),
    );
  }
}
