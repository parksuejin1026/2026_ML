import 'package:local_auth/local_auth.dart';
import 'package:shared_preferences/shared_preferences.dart';

enum AppBiometricKind { faceId, touchId, biometric }

class AppPreferences {
  AppPreferences._();

  static const _termsAcceptedKey = 'terms_accepted_v1';
  static const _lockEnabledKey = 'app_lock_enabled';
  static const _lockPinKey = 'app_lock_pin';
  static const _biometricsEnabledKey = 'app_lock_biometrics_enabled';

  static final LocalAuthentication _auth = LocalAuthentication();

  static Future<bool> isTermsAccepted() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_termsAcceptedKey) ?? false;
  }

  static Future<void> setTermsAccepted(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_termsAcceptedKey, value);
  }

  static Future<bool> isLockEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_lockEnabledKey) ?? false;
  }

  static Future<void> setLockEnabled(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_lockEnabledKey, value);
    if (!value) {
      await prefs.setBool(_biometricsEnabledKey, false);
    }
  }

  static Future<String?> getPin() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_lockPinKey);
  }

  static Future<void> setPin(String pin) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_lockPinKey, pin);
  }

  static Future<bool> verifyPin(String pin) async {
    final saved = await getPin();
    return saved != null && saved == pin;
  }

  static Future<bool> isBiometricsEnabled() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getBool(_biometricsEnabledKey) ?? false;
  }

  static Future<void> setBiometricsEnabled(bool value) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool(_biometricsEnabledKey, value);
  }

  static Future<bool> canUseBiometrics() async {
    try {
      return await _auth.isDeviceSupported() && await _auth.canCheckBiometrics;
    } catch (_) {
      return false;
    }
  }

  static Future<AppBiometricKind> biometricKind() async {
    try {
      final types = await _auth.getAvailableBiometrics();
      if (types.contains(BiometricType.face)) {
        return AppBiometricKind.faceId;
      }
      if (types.contains(BiometricType.fingerprint)) {
        return AppBiometricKind.touchId;
      }
      return AppBiometricKind.biometric;
    } catch (_) {
      return AppBiometricKind.biometric;
    }
  }

  static Future<bool> authenticateWithBiometrics() async {
    try {
      return _auth.authenticate(
        localizedReason: 'SubCut 앱 잠금을 해제합니다.',
        options: const AuthenticationOptions(
          biometricOnly: true,
          stickyAuth: true,
        ),
      );
    } catch (_) {
      return false;
    }
  }
}
