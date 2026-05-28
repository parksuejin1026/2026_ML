import 'package:flutter/material.dart';

import '../config/app_brand.dart';
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
                  child: showAgreeButton
                      ? _ConsentSummary(
                          onViewFull: () => Navigator.of(context).push(
                            MaterialPageRoute(
                              builder: (_) => const TermsScreen(),
                            ),
                          ),
                        )
                      : const _TermsContent(),
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
            '이용약관 및 개인정보 안내',
            style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
              letterSpacing: 0,
            ),
          ),
        ],
      ),
    );
  }
}

class _ConsentSummary extends StatelessWidget {
  final VoidCallback onViewFull;

  const _ConsentSummary({required this.onViewFull});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const _IntroCard(),
        const SizedBox(height: 16),
        const _SummaryCard(
          icon: Icons.lock_outline,
          title: '입력한 구독 정보로만 분석해요',
          body: '구독명, 금액, 사용 빈도, 최근 사용 시점, 필요도 같은 정보를 저장하고 추천에 사용합니다.',
        ),
        const _SummaryCard(
          icon: Icons.auto_graph,
          title: '추천은 참고용이에요',
          body: '실제 해지, 환불, 계약 조건은 각 서비스의 공식 채널에서 직접 확인해야 합니다.',
        ),
        const _SummaryCard(
          icon: Icons.fingerprint,
          title: '생체정보 원본은 저장하지 않아요',
          body: 'Face ID 또는 Touch ID를 쓰더라도 앱은 기기 인증 결과만 확인합니다.',
        ),
        const SizedBox(height: 8),
        Semantics(
          button: true,
          label: '전체 이용약관 및 개인정보 안내 보기',
          child: GestureDetector(
            onTap: onViewFull,
            behavior: HitTestBehavior.opaque,
            child: Container(
              height: 48,
              padding: const EdgeInsets.symmetric(horizontal: 16),
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppColors.border),
              ),
              child: const Row(
                children: [
                  Expanded(
                    child: Text(
                      '전체 이용약관 및 개인정보 안내 보기',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: AppColors.textPrimary,
                      ),
                    ),
                  ),
                  Icon(Icons.chevron_right,
                      size: 18, color: AppColors.textDisabled),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _SummaryCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String body;

  const _SummaryCard({
    required this.icon,
    required this.title,
    required this.body,
  });

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
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              color: AppColors.primarySoft,
              borderRadius: BorderRadius.circular(10),
            ),
            alignment: Alignment.center,
            child: Icon(icon, size: 18, color: AppColors.primaryDark),
          ),
          const SizedBox(width: 12),
          Expanded(
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
                const SizedBox(height: 5),
                Text(
                  body,
                  style: const TextStyle(
                    fontSize: 13,
                    height: 1.5,
                    fontWeight: FontWeight.w500,
                    color: AppColors.textSecondary,
                  ),
                ),
              ],
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
              '섭컷 SubCut은 사용자가 직접 입력한 구독 정보를 바탕으로 월 지출, 결제 예정일, 해지 추천 가능성을 보여주는 구독 관리 도구입니다. 앱의 분석과 추천은 참고용이며, 실제 결제 취소, 환불, 계약 해지, 소비자 권리 행사 여부는 사용자가 각 서비스의 공식 채널에서 직접 확인하고 결정해야 합니다.',
        ),
        _TermsBlock(
          title: '2. 수집 및 처리하는 정보',
          body:
              '앱은 구독명, 카테고리, 금액, 사용 빈도, 최근 사용 시점, 필요도, 대체 가능 여부, 유지 또는 해지 피드백, 앱 잠금 설정, 약관 동의 상태 등 사용자가 입력하거나 선택한 정보를 처리합니다. 백엔드 기능을 켠 경우 기기별 구분을 위한 임의 식별자와 서버 요청 기록이 함께 처리될 수 있습니다.',
        ),
        _TermsBlock(
          title: '3. 이용 목적과 보유 기간',
          body:
              '입력 정보는 구독 목록 저장, 지출 분석, 결제 일정 표시, 해지 추천, 피드백 반영, 앱 잠금, 장애 대응 및 서비스 품질 개선을 위해 사용됩니다. 기기 내 저장 정보는 사용자가 앱을 삭제하거나 데이터를 초기화할 때까지 보관될 수 있고, 서버 저장 정보는 삭제 요청, 서비스 종료 또는 관련 법령에 따른 보존 기간 만료 시까지 보관될 수 있습니다.',
        ),
        _TermsBlock(
          title: '4. 로컬 기능과 서버 기능',
          body:
              '일부 기능은 기기 내 로컬 상태로 동작하고, 일부 기능은 사용자가 설정한 추론 서버 또는 저장 서버와 연동될 수 있습니다. PIN, 생체인증 사용 여부, 약관 동의 상태 등 앱 잠금 관련 설정은 기기 내에 저장됩니다. Face ID 또는 Touch ID의 생체정보 원본은 앱이 수집하거나 저장하지 않고, 기기 운영체제의 인증 결과만 사용합니다.',
        ),
        _TermsBlock(
          title: '5. 모델 학습과 자동화 추천',
          body:
              '해지 추천은 사용자가 입력한 정보와 모델 판단에 기반한 자동화된 참고 결과입니다. 추천 결과는 사용자의 법적 권리나 의무를 확정하지 않으며, 실제 결제 취소나 계약 해지는 사용자가 직접 진행해야 합니다. 모델 성능 개선에는 개인을 직접 식별하기 어려운 통계 또는 익명화된 데이터가 우선 사용되며, 개인 식별이 가능한 데이터를 별도 목적의 학습에 사용하는 경우에는 필요한 고지와 동의를 받아야 합니다.',
        ),
        _TermsBlock(
          title: '6. 추천 결과의 한계',
          body:
              '해지 추천, 절감액, 결제 예정일, 분석 차트는 추정값이며 실제 청구 금액, 결제일, 환불 가능 여부, 위약금, 연간 구독 조건과 다를 수 있습니다. 중요한 결정을 하기 전에는 각 구독 서비스의 공식 정보를 확인해야 합니다.',
        ),
        _TermsBlock(
          title: '7. 사용자 책임과 입력 제한',
          body:
              '사용자는 본인의 구독 정보가 정확하게 입력되었는지 확인해야 합니다. 민감정보, 결제카드 전체번호, 계정 비밀번호, 주민등록번호, 타인의 개인정보 등 서비스 이용에 필요하지 않은 정보는 입력하지 않아야 합니다.',
        ),
        _TermsBlock(
          title: '8. 동의 거부, 철회 및 데이터 관리',
          body:
              '사용자는 개인정보 수집·이용에 대한 동의를 거부할 수 있으나, 필수 정보 처리가 필요한 구독 저장, 분석, 추천 등 핵심 기능 이용이 제한될 수 있습니다. 사용자는 앱 내 설정, 데이터 관리 기능 또는 앱스토어 지원 연락처 등 별도 고지된 고객지원 채널을 통해 데이터 열람, 정정, 삭제, 처리정지, 동의 철회를 요청할 수 있습니다. 익명화되어 개인을 식별할 수 없는 통계 데이터는 개별 사용자 기준으로 분리 삭제가 어려울 수 있습니다.',
        ),
        _TermsBlock(
          title: '9. 제3자 제공 및 처리 위탁',
          body:
              '앱은 법령에 근거가 있거나 사용자가 동의한 경우를 제외하고 개인을 식별할 수 있는 정보를 제3자에게 제공하지 않습니다. 서버 운영, 분석, 장애 대응 등 처리를 외부 업체에 위탁하거나 국외 이전이 필요한 경우에는 제공받는 자, 목적, 항목, 보유 기간, 거부권 등 필요한 사항을 별도로 고지합니다.',
        ),
        _TermsBlock(
          title: '10. 책임 제한',
          body:
              '앱은 안정적인 서비스를 제공하기 위해 합리적인 보안 조치를 취합니다. 다만 사용자의 입력 오류, 외부 구독 서비스 정책 변경, 네트워크 또는 서버 장애, 사용자가 공식 채널에서 해지 절차를 완료하지 않아 발생한 손해에 대해서는 법령상 허용되는 범위에서 책임이 제한됩니다.',
        ),
        _TermsBlock(
          title: '11. 약관 변경',
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
            '${AppBrand.displayName} 이용 및 데이터 활용 안내',
            style: TextStyle(
              fontSize: 20,
              fontWeight: FontWeight.w800,
              color: AppColors.textPrimary,
            ),
          ),
          SizedBox(height: 8),
          Text(
            '앱을 사용하기 전에 구독 정보 처리, 개인정보 이용, 모델 추천 결과의 한계를 확인해주세요.',
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
          child: Semantics(
            button: true,
            label: '동의하고 시작하기',
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
      ),
    );
  }
}
