import 'dart:ui' show Tristate;
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter/widgets.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:subscription_check/main.dart';
import 'package:subscription_check/providers/subscription_provider.dart';

void main() {
  testWidgets('SubCut shell renders bottom tabs', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues({
      'terms_accepted_v1': true,
    });

    await tester.pumpWidget(
      ChangeNotifierProvider(
        create: (_) => SubscriptionProvider(),
        child: const MyApp(),
      ),
    );
    await tester.pump();

    expect(find.text('섭컷 SubCut'), findsOneWidget);
    expect(find.text('홈'), findsOneWidget);
    expect(find.text('추천'), findsOneWidget);
    expect(find.text('일정'), findsOneWidget);
    expect(find.text('분석'), findsOneWidget);
    expect(find.text('설정'), findsOneWidget);

    await tester.tap(find.byKey(const ValueKey('tab-설정')));
    for (var i = 0; i < 8; i++) {
      await tester.pump(const Duration(milliseconds: 100));
    }

    expect(find.text('앱 데이터'), findsOneWidget);
  });

  testWidgets('SubCut shell changes tabs with horizontal swipes',
      (WidgetTester tester) async {
    final semantics = tester.ensureSemantics();
    try {
      SharedPreferences.setMockInitialValues({
        'terms_accepted_v1': true,
      });

      await tester.pumpWidget(
        ChangeNotifierProvider(
          create: (_) => SubscriptionProvider(),
          child: const MyApp(),
        ),
      );
      await tester.pump();

      expect(_isTabSelected(tester, '홈'), isTrue);

      await tester.drag(find.byType(PageView), const Offset(-500, 0));
      await tester.pumpAndSettle();

      expect(_isTabSelected(tester, '추천'), isTrue);
    } finally {
      semantics.dispose();
    }
  });
}

bool _isTabSelected(WidgetTester tester, String label) {
  return tester
          .getSemantics(find.byKey(ValueKey('tab-$label')))
          .flagsCollection
          .isSelected ==
      Tristate.isTrue;
}
