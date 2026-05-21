import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

class AppTopBar extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;

  const AppTopBar({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.fromLTRB(24, 0, 12, 0),
  });

  @override
  Widget build(BuildContext context) {
    final top = MediaQuery.of(context).padding.top;

    return Container(
      height: top + 60,
      padding: EdgeInsets.only(top: top),
      decoration: const BoxDecoration(
        color: AppColors.surface,
        border: Border(bottom: BorderSide(color: AppColors.divider)),
      ),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 460),
          child: Padding(
            padding: padding,
            child: SizedBox(height: 60, child: child),
          ),
        ),
      ),
    );
  }
}
