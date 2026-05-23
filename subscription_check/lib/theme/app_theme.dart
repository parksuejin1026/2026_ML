import 'package:flutter/material.dart';

class AppColors {
  const AppColors._();

  static const Color bg = Color(0xFFF3F4F6);
  static const Color surface = Colors.white;
  static const Color divider = Color(0xFFF2F4F6);
  static const Color border = Color(0xFFECEEF0);

  static const Color primary = Color(0xFF53B2FF);
  static const Color primaryLight = Color(0xFF7CC5FF);
  static const Color primaryDark = Color(0xFF258FE8);
  static const Color primarySoft = Color(0xFFEAF6FF);
  static const Color primarySoftBg = Color(0xFFF2FAFF);

  static const Color textPrimary = Color(0xFF191F28);
  static const Color textSecondary = Color(0xFF4E5968);
  static const Color textMuted = Color(0xFF6B7684);
  static const Color textTertiary = Color(0xFF8B95A1);
  static const Color textDisabled = Color(0xFFB0B8C1);
  static const Color textPlaceholder = Color(0xFFC4CAD4);

  static const Color neutralChip = Color(0xFFF2F4F6);
  static const Color neutralChipDark = Color(0xFFE5E8EB);
  static const Color neutralSoft = Color(0xFFF8F9FA);

  static const Color danger = Color(0xFFF04452);
  static const Color dangerSoft = Color(0xFFFFF5F5);
  static const Color dangerSofter = Color(0xFFFFF0F0);
  static const Color dangerTint = Color(0xFFFFF8F8);

  static const Color success = Color(0xFF00B386);
  static const Color successLight = Color(0xFF5CD9A8);
  static const Color successSoft = Color(0xFFEAF8F0);
  static const Color successSofter = Color(0xFFE8F7F0);
  static const Color successTint = Color(0xFFF0FAF5);

  static const LinearGradient primaryGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [primaryLight, primary, primaryDark],
    stops: [0.0, 0.5, 1.0],
  );

  static const LinearGradient primarySimpleGradient = LinearGradient(
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    colors: [primaryLight, primary],
  );
}

class AppTheme {
  const AppTheme._();

  static ThemeData light() {
    const fontFamily = 'Pretendard';

    final base = ThemeData(
      useMaterial3: true,
      fontFamilyFallback: const [
        '-apple-system',
        'BlinkMacSystemFont',
        'Apple SD Gothic Neo',
        'Segoe UI',
        'Roboto',
        'Helvetica Neue',
        'Arial',
        'sans-serif',
      ],
      scaffoldBackgroundColor: AppColors.bg,
      colorScheme: const ColorScheme.light(
        primary: AppColors.primary,
        onPrimary: Colors.white,
        surface: AppColors.surface,
        onSurface: AppColors.textPrimary,
        error: AppColors.danger,
      ),
      splashColor: Colors.transparent,
      highlightColor: Colors.transparent,
    );

    return base.copyWith(
      textTheme: base.textTheme
          .apply(
            fontFamily: fontFamily,
            bodyColor: AppColors.textPrimary,
            displayColor: AppColors.textPrimary,
          )
          .copyWith(
            bodyLarge: const TextStyle(
                fontSize: 16, color: AppColors.textPrimary, height: 1.4),
            bodyMedium: const TextStyle(
                fontSize: 14, color: AppColors.textSecondary, height: 1.4),
            bodySmall: const TextStyle(
                fontSize: 12, color: AppColors.textTertiary, height: 1.4),
          ),
    );
  }
}

String formatKRW(num value) {
  final s = value.toStringAsFixed(0);
  final reversed = s.split('').reversed.toList();
  final buf = StringBuffer();
  for (var i = 0; i < reversed.length; i++) {
    if (i > 0 && i % 3 == 0) buf.write(',');
    buf.write(reversed[i]);
  }
  return buf.toString().split('').reversed.join();
}
