"""

config.py ----- 프로젝트 경로와 공통 설정

실행환경이나 패스가 달라지면 이 파일만 고치면 되겠금 만들 것임.
노트북에서는 
form config import *
print(PKLOT_ORIGN)

"""

from pathlib import Path
#================================================
# 1. 프로젝트 폴더
# 이 파일(config.py)가 있는 폴더를 상위 폴더(root)라고 본다.
# 루트
ROOT = Path(__file__).resolve().parent
#print(f'ROOT: {ROOT}')

# ===============================================
# 2. 원본 데이터. PKLot, PKLotSegmented
PKLOT = ROOT / "PKLOT" / "PKLOT" / "PKLOT" # 주자창 전체 사진
PKLOT_SEG = ROOT / "PKLOT" / "PKLOTSegmented" # 주차면 크롭사진

# ==============================================
# 3. 주차장 데이터

MYLOT = ROOT / "data" / "mylot"
MYLOT_IMAGES = MYLOT / "images" # 우리 주자창 사진 
MYLOT_ROIS = MYLOT / "rois"   #우리 주차장 좌표 파일 (*.json, *.xml 등)이 될것

# =================================================
# 4. 학습용 데이터, 모델 등
DATASETS = ROOT / "datasets" #학습 데이터 셋 train/val/test
MODELS = ROOT / "models" #쓸만한 가중치 모아두는 곳
RUNS = ROOT / "runs" # Ultralytics 학습결과(자동 생성됨) 가중치도 여기있음
# RESULT = ROOT / ""

# 분류 모델 만든 거 A, B
MODEL_A_PT = MODELS / "cls_v1" / "weights" / "best.pt" #PUC주차장 파인튜닝 안함. 기본 모델로 두고 하는 중
MODEL_B_PT = MODELS / "cls_ft_puc5d" / "weights" / "best.pt"  #PUC주차장 파인튜닝 한 실험본.
# ====================================================
# 5. 공통 설정
IMGSIZE = 96   #분류 모델 입력 크기(사진 사이즈)
CROP_METHOD = "minrect" # 주차면 자르는 방식
SEED = 0 

def check_paths():
    """경로가 실제로 있는지 한 번에 확인"""
    items = {
        "ROOT": ROOT, "PKLOT": PKLOT, "PKLOT_SEG": PKLOT_SEG,
        "MYLOT_IMAGES": MYLOT_IMAGES, "MYLOT_ROIS": MYLOT_ROIS,
        "DATASETS": DATASETS, "MODEL_A_PT": MODEL_A_PT,
    }
    for name, p in items.items():
        print(f"{'O' if Path(p).exists() else 'X'}  {name:13s} {p}")

def pathsList():
    items = {
            "ROOT": ROOT, "PKLOT": PKLOT, "PKLOT_SEG": PKLOT_SEG, "MYLOT" : MYLOT, 
            "MYLOT_IMAGES": MYLOT_IMAGES, "MYLOT_ROIS": MYLOT_ROIS, "MODELS" : MODELS, "RUNS" : RUNS,
            "DATASETS": DATASETS, "MODEL_A_PT": MODEL_A_PT, "MODEL_B_PT" : MODEL_B_PT, 
        }
    for name, p in items.items():
        print(f"{name:13s} {p}")

for d in [MODELS, MYLOT_IMAGES, MYLOT_ROIS]:
    d.mkdir(parents=True, exist_ok=True)
print(check_paths())