"""
parking_utils.py =================== 주차장 관련 함수들 모음

"""
import cv2
from config import *
import matplotlib.pyplot as plt

import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from ultralytics import YOLO




# ===========================================
# 1.파일 읽기.

# 이미지 읽기
# cv2는 한글 path를 못읽음 그래서 np로 파일 그대로 읽고 그대로 바이트로 읽어서 1차원 numpy배열 생성.
# 변환한 바이트를 cv2가 컬러이미지로 변환함.
def imread(imgpath):
    img = cv2.imdecode(np.fromfile(str(imgpath), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(imgpath)
    return img

# xml파일 읽기
"""
  <space id="1" occupied="1">
    <rotatedRect>
      <center x="724" y="623" />
      <size w="93" h="154" />
      <angle d="-31" />
    </rotatedRect>
    <contour>
      <point x="720" y="549" />
      <point x="805" y="665" />
      <point x="715" y="698" />
      <point x="644" y="582" />
    </contour>
  </space>
"""
def load_slots_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    rows = []
    for space in root.iter("space"):
        row = {}
        row["id"] = space.get("id")
        row["occupied"] = space.get("occupied")
        points = []
        for p in space.findall("./contour/point"):
            points.append((int(p.get("x")), int(p.get("y"))))
        row["points"] = points
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[df["points"].map(len) == 4].reset_index(drop=True)
    return df


# json읽기
def load_slots_json(json_path):
    print("json이 어떤식으로 생성되는지 몰라 미작성")
    print("작성예정")


# =========================================
# 2. 자르기(모든 함수가 꼭지점 4개인 경우만 받음)
# =========================================

#꼭지점 4개를 [왼위, 오위, 오아래, 왼아래]로 정렬
def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    c = pts.mean(axis = 0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c [1], pts[:, 0]-c[0]))] # arctan2는 요소별 arctan값을 계산, 라디안 단위의 각도를 반환해줌.
                                                                        # -PI ~ PI까지의 값. 
    # 왼위의 좌표값이 합했을 때 가장 작음. 해당 인덱스를 가져와서 밀어버림.
    # np.roll(원본 배열, shift, axis=0)행 방향으로. 하나 또 배웠습니다.
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0) 


#이미지 크롭, 이미지를 pts에 맞추어 돌려 크롭하여 반환.
def crop_minrect(img, pts):
    (cx, cy), (w, h), angle = cv2.minAreaRect(np.array(pts, dtype=np.float32)) # 최소 면적의 회전된 사각형을 찾아 반환해줌. 
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0) # 어느 방향으로 얼마나 돌릴지 계산하는 회전지도 행렬을 반환해줌. 어파인 변환행렬이라고 칭함.
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))  # 어파인 변환 행렬 적용해서 이미지 회전. 
    return cv2.getRectSubPix(rotated, (max(int(w), 1), max(int(h), 1)), (cx, cy)) # 바르게 회전된 이미지 크롭하기. 중심점 기준으로 w,h만큼.


#이미지 크롭, 꼭지점 4개로 원근 보정(사각형이 찌그러져있으면 펴서) 자름
def crop_quad(img, pts):
    tl, tr, br, bl = order_points(pts)
    w = max(int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))), 1) # linalg.norm 피타고라스 정리 비슷한거. 두점 사이 거리를 재줌. 가로 폭 계산.
    h = max(int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))), 1) #세로 폭 계산. 찌그러지지 않게 max값으로 가져옴. 1은 0 방지.
    dst = np.array([[0,0], [w-1, 0],[w-1, h-1], [0,h-1]], dtype=np.float32) #최종적으로 그릴 목표(destination) 좌표 예시. 밑에 cv한테 줘야함.
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst) #dst모양으로 바꾸기위해 잡아 당겨야하는 값들을 만들어줌. 
    return cv2.warpPerspective(img, M,(w,h)) # M대로 이미지의 픽셀들을 잡아당겨 크롭해줌.


CROP_FUNCS = {"minrect": crop_minrect, "quad": crop_quad}
# 이미지와 장소(load_slots로 받은 데이터 프레임), 메소드를 골라서 이미지 크롭.
def crop_slots(img, slots, method=CROP_METHOD):
    crop = CROP_FUNCS[method]
    return [crop(img, pts) for pts in slots["points"]]


#=========================================================
# 3. 판정

# 잘라낸 칸 이미지들 -> (라벨 리스트, 확신도 리스트)
# 리스트를 통째로 넣으면 메모리가 터짐. batch장씩 나눠넣기.
# device가 None, GPU, 없음 CPU알아함.
# model은 모델, crops는 크롭한 사진, imgsz는 이미지 사이즈. 웬만하면바꾸지 말기. 학습을 저 값으로 함.
def classify(model, crops, imgsz=IMGSIZE, device=None, batch=256): 
    labels, confs = [], []
    for i in range(0, len(crops), batch):
        for r in model.predict(crops[i:i +batch], imgsz=imgsz, device=device, verbose=False): #이미지의 결과 튀어나옴, 256개. 
            labels.append(model.names[r.probs.top1]) # 가장 확률 높은 클래스 번호 -> 이름으로 바꿈.
            confs.append(float(r.probs.top1conf)) # 그 확률 값. 
    return labels, confs

def judge(model, img, slots, method=CROP_METHOD, **keyword): 
    """사진 한 장 + 칸 표 → (라벨, 확신도).  자르기 + 판정을 한 번에
        위에서 한 메소드들 모은 거임 그냥. **keyword는 나머지 인자는 그냥 넘길게. 라는 뜻
    """ 
    return classify(model, crop_slots(img, slots, method), **keyword)


     
# =========================================
# 4. 결과 그리기

COLORS = {"empty": (0, 200, 0), "occupied": (0, 0, 255), "uncertain": (0, 220, 255)}   # BGR openCV가 BGR을 씀.... 아쉽게도..

#칸마다 색칠하고 왼쪽 위에 빈 자리수 표시 (원본 터치 X)
def draw(img, slots, labels):
    out = img.copy()
    for pts, lab in zip(slots["points"], labels):
        cv2.polylines(out, [order_points(pts).astype(np.int32)], True, COLORS.get(lab, (255, 255, 255)), 2)
    cv2.putText(out, f"FREE {list(labels).count('empty')}/{len(labels)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return out


# 디버깅 
# xml = list(PKLOT.rglob("*.xml"))[30]
# jpg_path = xml.with_suffix(".jpg")
# img = imread(jpg_path)
# df = load_slots_xml(xml)
# pts = df["points"]

# print(pts[0])

# cropedImg = crop_minrect(img, pts[0]) 
# plt.imshow(cropedImg)
# plt.show()
# cropedImg = crop_quad(img, pts[0]) 
# plt.imshow(cropedImg)
# plt.show()



# model_A = YOLO(MODEL_A_PT)
# img     = imread(jpg_path)
# slots   = load_slots_xml(jpg_path.with_suffix(".xml"))

# labels, confs = judge(model_A, img, slots)       # 자르기 + 판정
# result = draw(img, slots, labels)                # 결과 그림
# plt.imshow(result)
# plt.show()