"""
config
 - 프로젝트 전역 설정 모듈

코드 구성
 0) 라이브러리 임포트
 1) 경로 설정 (프로젝트 루트, 샘플/결과 폴더)
 2) 유틸 함수: 경로 보정/폴더 생성
 3) 품질 지표 설정 (quality_metrics.py 관련)
 4) 보행(HS/TO) 검출 파라미터
 5) STS 검출 파라미터
 6) 시각화/표시 옵션
 7) 실행 환경 옵션
"""



# 0) ───────────────────────────────────────────────────────────────────────────
#    라이브러리 임포트
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path


# 1) ───────────────────────────────────────────────────────────────────────────
#    경로 설정 (프로젝트 루트, 샘플/결과 폴더)
# -----------------------------------------------------------------------------
# ▶ config.py는 src/ 안에 있으므로, 부모의 부모가 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class Paths:
    """프로젝트 폴더 경로 모음."""
    root: Path = PROJECT_ROOT
    samples: Path = PROJECT_ROOT / "samples"
    results: Path = PROJECT_ROOT / "results"
    results_figures: Path = PROJECT_ROOT / "results" / "figures"
    results_keypoints: Path = PROJECT_ROOT / "results" / "keypoints"

    def ensure_dirs(self) -> None:
        """필요한 폴더 생성(존재하면 무시)."""
        self.results.mkdir(parents=True, exist_ok=True)
        self.results_figures.mkdir(parents=True, exist_ok=True)
        self.results_keypoints.mkdir(parents=True, exist_ok=True)

# 전역 PATH 객체 (import해서 바로 사용)
PATHS = Paths()
PATHS.ensure_dirs()


# 2) ───────────────────────────────────────────────────────────────────────────
#    유틸 함수: 경로 보정/폴더 생성
# -----------------------------------------------------------------------------
def resolve_path(p: str | Path) -> Path:
    """
    상대경로가 들어오면 프로젝트 루트 기준으로 보정해 절대경로 반환.
    절대경로면 그대로 반환.
    """
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (PROJECT_ROOT / pp).resolve()


# 3) ───────────────────────────────────────────────────────────────────────────
#    품질 지표 설정 (quality_metrics.py 관련)
# -----------------------------------------------------------------------------
@dataclass
class QualityConfig:
    """
    품질 필터 기준
     - avg_visibility_thr : 평균 visibility 최소 기준
     - visible_thr        : 프레임을 'usable'로 판단하는 visibility 임계값
     - visible_ratio_min  : usable 프레임 비율 최소 기준
     - bbox_ratio_min     : 프레임 대비 bbox 면적 비율 최소 기준
    """
    avg_visibility_thr: float = 0.55
    visible_thr: float = 0.50
    visible_ratio_min: float = 0.60
    bbox_ratio_min: float = 0.10

QUALITY = QualityConfig()


# 4) ───────────────────────────────────────────────────────────────────────────
#    보행(HS/TO) 검출 파라미터
# -----------------------------------------------------------------------------
@dataclass
class GaitConfig:
    """
    HS/TO 피크 검출 파라미터
     - min_dist_frames : 피크 간 최소 간격(프레임)
     - prominence      : 피크 prominence 임계값(정규화 y에서)
     - default_side    : 기본 분석 다리
    """
    min_dist_frames: int = 10
    prominence: float = 0.01
    default_side: str = "left"

GAIT = GaitConfig()


# 5) ───────────────────────────────────────────────────────────────────────────
#    STS 검출 파라미터
# -----------------------------------------------------------------------------
@dataclass
class STSConfig:
    """
    STS (sit→stand) 검출 파라미터
     - vel_sign_min_dur_s : 속도 부호 전환 최소 간격(초)
     - pair_min_s/pair_max_s : sit→stand 유효 구간 길이(초) 범위
     - default_side        : 기본 분석 다리
    """
    vel_sign_min_dur_s: float = 0.3
    pair_min_s: float = 0.4
    pair_max_s: float = 5.0
    default_side: str = "left"

STS = STSConfig()


# 6) ───────────────────────────────────────────────────────────────────────────
#    시각화/표시 옵션
# -----------------------------------------------------------------------------
@dataclass
class DisplayConfig:
    """
    미리보기/오버레이 관련 표시 옵션
     - show_preview  : 실시간 미리보기 기본값
     - show_overlay  : 검출 결과 오버레이 재생 기본값
     - win_title_gait: 보행 창 제목
     - win_title_sts : STS 창 제목
    """
    show_preview: bool = False
    show_overlay: bool = False
    win_title_gait: str = "Gait Events (q/ESC to quit)"
    win_title_sts: str = "STS Events (q/ESC to quit)"

DISPLAY = DisplayConfig()


# 7) ───────────────────────────────────────────────────────────────────────────
#    실행 환경 옵션
# -----------------------------------------------------------------------------
@dataclass
class RuntimeConfig:
    """
    실행 환경 관련 옵션
     - default_fps_csv : CSV만 입력될 때 사용할 기본 FPS
     - seed            : 난수 고정(필요시)
    """
    default_fps_csv: float = 30.0
    seed: int = 42

RUNTIME = RuntimeConfig()
