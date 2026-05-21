import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:subscription_check/services/predict_service.dart';

void main() {
  test('default frontend mode returns mock subscriptions without device id',
      () async {
    SharedPreferences.setMockInitialValues({});

    final subscriptions = await fetchSubscriptions();
    final savings = await fetchSavings();
    final prefs = await SharedPreferences.getInstance();

    expect(subscriptions, isNotEmpty);
    expect(savings.cumulativeSavings, 0);
    expect(prefs.getString('device_id'), isNull);
  });
}
