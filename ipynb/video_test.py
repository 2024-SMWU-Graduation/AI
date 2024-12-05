import cv2
import torch
from torchvision import transforms
from model import ResNet18Model  # 모델 정의 포함 파일
from collections import Counter
import time  # 시간 측정을 위한 모듈

MODEL_PATH = "/Users/mks/Documents/GitHub/AI/model/ResNet18_final_best_updated.pth"
CLASS_NAMES = ["Negative", "Positive"]
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
# GPU 사용 여부 설정
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")  # Mac GPU

# Haar Cascade 로드
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# 모델 로드
model = ResNet18Model(num_classes=len(CLASS_NAMES))
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_faces(frame):
    """
    프레임에서 얼굴 검출
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def predict_face(face):
    """
    검출된 얼굴에서 예측 결과 반환
    """
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = transforms.ToPILImage()(face_rgb)
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    start_time = time.time()  # 예측 시작 시간
    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs, 1)
    inference_time = time.time() - start_time  # 예측 수행 시간

    return CLASS_NAMES[predicted.item()], inference_time


def merge_intervals(timestamps, labels):
    """
    연속된 Negative 판단 구간 병합
    """
    result = []
    start, end = None, None

    for time, label in zip(timestamps, labels):
        if label == "Negative":
            if start is None:
                start = time
            end = time
        else:
            if start is not None:
                result.append((start, end))
                start, end = None, None

    if start is not None:
        result.append((start, end))

    return result


def format_time(seconds):
    """
    초 단위를 MM:SS 형식으로 변환
    """
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02}:{seconds:02}"


def analyze_video(video_path):
    """
    동영상 분석
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("동영상을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 0.5)
    predictions = []
    timestamps = []
    total_inference_time = 0  # 전체 예측 시간 누적

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            faces = detect_faces(frame)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    label, inference_time = predict_face(face)
                    predictions.append(label)
                    timestamps.append(timestamp)
                    total_inference_time += inference_time
        frame_count += 1

    cap.release()

    # 결과 요약
    summary = Counter(predictions)
    if len(predictions) > 0:
        negative_ratio = (summary["Negative"] / sum(summary.values())) * 100
        print(f"Negative 비율: {negative_ratio:.2f}%")

        # 연속된 Negative 구간 병합 및 출력
        negative_intervals = merge_intervals(timestamps, predictions)
        if negative_intervals:
            print("Negative 판단 구간:")
            for start, end in negative_intervals:
                print(f" {format_time(start)} ~ {format_time(end)}")
        else:
            print("Negative 판단 없음.")
    else:
        print("검출된 얼굴이 없어 분석할 수 없습니다.")
    # 모델 예측 시간 출력
    print(f"전체 모델 예측 수행 시간: {total_inference_time:.2f}초")

analyze_video("/Users/mks/Documents/GitHub/AI/archive/test_video.mov")
