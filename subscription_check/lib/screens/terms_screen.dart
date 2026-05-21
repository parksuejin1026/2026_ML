import 'package:flutter/material.dart';

import '../theme/app_theme.dart';
import '../widgets/app_top_bar.dart';

const double _maxContentWidth = 460;

class TermsScreen extends StatelessWidget {
  final bool showAgreeButton;
  final VoidCallback? onAgree;

  const TermsScreen({
    super.key,
    this.showAgreeButton = false,
    this.onAgree,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.bg,
      body: Column(
        children: [
          _Header(canClose: !showAgreeButton),
          Expanded(
            child: SingleChildScrollView(
              padding: EdgeInsets.fromLTRB(
                20,
                18,
                20,
                showAgreeButton
                    ? 24 + MediaQuery.of(context).padding.bottom
                    : 120,
              ),
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: _maxContentWidth),
                  child: const _TermsContent(),
                ),
              ),
            ),
          ),
          if (showAgreeButton)
            _AgreeBar(onAgree: onAgree ?? () => Navigator.of(context).pop()),
        ],
      ),
    );
  }
}

class _Header extends StatelessWidget {
  final bool canClose;

  const _Header({required this.canClose});

  @override
  Widget build(BuildContext context) {
    return AppTopBar(
      padding: const EdgeInsets.symmetric(horizontal: 8),
      child: Row(
        children: [
          if (canClose)
            IconButton(
              icon: const Icon(Icons.arrow_back_ios_new,
                  size: 18, color: AppColors.textPrimary),
              onPressed: () => Navigator.of(context).pop(),
            )
          else
            const SizedBox(width: 16),
          const SizedBox(width: 4),
          const Text(
            '이용약관',
            style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
              letterSpacing: -0.34,
            ),
          ),
        ],
      ),
    );
  }
}

class _TermsContent extends StatelessWidget {
  const _TermsContent();

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: const [
        _IntroCard(),
        SizedBox(height: 16),
        _TermsBlock(
          title: '1. 서비스 성격',
          body:
              'SubCut은 사용자가 직접 입력한 구독 정보를 바탕으로 월 지출, 결제 예정일, 해지 추천 가능성을 보여주는 구독 관리 도구입니다. 앱의 분석과 추천은 참고용이며, 실제 결제 취소, 환불, 계약 해지, 소비자 권리 행사 여부는 사용자가 각 서비스의 공식 채널에서 직접 확인하고 결정해야 합니다.',
        ),
        _TermsBlock(
          title: '2. 사용자 입력 데이터',
          body:
              '앱은 구독명, 카테고리, 금액, 사용 빈도, 최근 사용 시점, 필요도, 대체 가능 여부, 유지 또는 해지 피드백 등 사용자가 입력하거나 선택한 정보를 처리합니다. 민감정보, 결제카드 전체번호, 계정 비밀번호, 주민등록번호 등 서비스 이용에 필요하지 않은 정보는 입력하지 않아야 합니다.',
        ),
        _TermsBlock(
          title: '3. 데이터 활용 및 모델 학습',
          body:
              '사용자가 제공한 구독 정보와 피드백은 서비스 품질 개선, 통계 분석, 해지 추천 모델의 성능 개선 및 재학습에 활용될 수 있습니다. 모델 학습에는 개인을 직접 식별하기 어려운 형태의 사용 패턴과 피드백 데이터가 우선 사용되며, 앱은 추천 정확도 개선을 위해 누적 데이터를 분석할 수 있습니다.',
        ),
        _TermsBlock(
          title: '4. 로컬 기능과 서버 기능',
          body:
              '일부 기능은 기기 내 로컬 상태로 동작하고, 일부 기능은 추론 서버 또는 저장 서버와 연동될 수 있습니다. 현재 프론트엔드 프로토타입에서 제공되는 캘린더, 설정, 잠금, 약관 동의 등은 백엔드 없이 기기 내 상태를 우선 사용합니다.',
        ),
        _TermsBlock(
          title: '5. 추천 결과의 한계',
          body:
              '해지 추천, 절감액, 결제 예정일, 분석 차트는 사용자가 입력한 정보와 모델 판단에 기반한 추정값입니다. 실제 청구 금액, 결제일, 환불 가능 여부, 위약금, 연간 구독 조건과 다를 수 있으므로 중요한 결정 전에는 각 구독 서비스의 공식 정보를 확인해야 합니다.',
        ),
        _TermsBlock(
          title: '6. 사용자 책임',
          body:
              '사용자는 본인의 구독 정보가 정확하게 입력되었는지 확인해야 하며, 앱에서 제공하는 정보에만 의존해 발생한 결제, 해지 지연, 환불 실패, 서비스 이용 중단 등 결과에 대해 앱은 법령상 허용되는 범위 내에서 책임을 제한합니다.',
        ),
        _TermsBlock(
          title: '7. 동의 철회 및 데이터 관리',
          body:
              '사용자는 앱 내 설정 또는 향후 제공되는 데이터 관리 기능을 통해 일부 데이터를 삭제하거나 내보내는 기능을 사용할 수 있습니다. 서버에 이미 반영된 학습용 통계 데이터는 모델 품질과 무결성을 위해 즉시 개별 분리 삭제가 어려울 수 있습니다.',
        ),
        _TermsBlock(
          title: '8. 약관 변경',
          body:
              '서비스 기능, 데이터 처리 방식, 법령 또는 운영 정책이 변경되면 본 안내도 변경될 수 있습니다. 중요한 변경 사항은 앱 내 화면 또는 공지 방식으로 안내할 수 있습니다.',
        ),
        SizedBox(height: 8),
        Text(
          '위 내용은 앱 사용을 위한 일반 안내이며, 개별 법률 자문이나 금융 자문을 대체하지 않습니다.',
          style: TextStyle(
            fontSize: 12,
            height: 1.5,
            fontWeight: FontWeight.w600,
            color: AppColors.textDisabled,
          ),
        ),
      ],
    );
  }
}

class _IntroCard extends StatelessWidget {
  const _IntroCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'SubCut 이용 및 데이터 활용 안내',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
            ),
          ),
          SizedBox(height: 8),
          Text(
            '앱을 사용하기 전에 구독 정보 처리, 모델 학습 활용, 추천 결과의 한계를 확인해주세요.',
            style: TextStyle(
              fontSize: 14,
              height: 1.5,
              fontWeight: FontWeight.w500,
              color: AppColors.textTertiary,
            ),
          ),
        ],
      ),
    );
  }
}

class _TermsBlock extends StatelessWidget {
  final String title;
  final String body;

  const _TermsBlock({required this.title, required this.body});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: const TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            body,
            style: const TextStyle(
              fontSize: 13,
              height: 1.6,
              fontWeight: FontWeight.w500,
              color: AppColors.textSecondary,
            ),
          ),
        ],
      ),
    );
  }
}

class _AgreeBar extends StatelessWidget {
  final VoidCallback onAgree;

  const _AgreeBar({required this.onAgree});

  @override
  Widget build(BuildContext context) {
    final bottom = MediaQuery.of(context).padding.bottom;
    return Container(
      padding: EdgeInsets.fromLTRB(20, 12, 20, 14 + bottom),
      decoration: BoxDecoration(
        color: AppColors.surface,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.08),
            blurRadius: 18,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: _maxContentWidth),
          child: GestureDetector(
            onTap: onAgree,
            child: Container(
              height: 54,
              decoration: BoxDecoration(
                color: AppColors.primary,
                borderRadius: BorderRadius.circular(16),
              ),
              alignment: Alignment.center,
              child: const Text(
                '동의하고 시작하기',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w800,
                  color: Colors.white,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
