import cv2
import torch
from torchvision import transforms
from collections import Counter

# 클래스 이름 정의 및 디바이스 설정
CLASS_NAMES = ["Negative", "Positive"]
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")

# 전처리 파이프라인 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_model(model_path, model_class):
    """
    PyTorch 모델을 로드합니다.

    :param model_path: 모델 가중치 파일 경로
    :param model_class: 모델 클래스 (예: ResNet18Model)
    :return: 로드된 PyTorch 모델
    """
    model = model_class(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def detect_faces(frame):
    """
    프레임에서 얼굴을 검출합니다.

    :param frame: 비디오 프레임 (OpenCV 이미지)
    :return: 검출된 얼굴의 좌표 리스트
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def predict_face(face, model):
    """
    얼굴 이미지를 입력으로 받아 예측 결과를 반환합니다.

    :param face: 얼굴 이미지 (OpenCV 이미지)
    :param model: PyTorch 모델
    :return: 예측된 클래스 이름
    """
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = transforms.ToPILImage()(face_rgb)
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs, 1)

    return CLASS_NAMES[predicted.item()]

def analyze_video(video_path, model):
    """
    비디오를 분석하고 결과를 반환합니다.

    :param video_path: 비디오 파일 경로
    :param model: PyTorch 모델
    :return: 분석 결과 문자열 (Negative 비율 및 구간)
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 0.5)

    predictions = []
    negative_intervals = []

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            faces = detect_faces(frame)
            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]
                label = predict_face(face, model)
                predictions.append(label)
                if label == "Negative":
                    time_in_seconds = (frame_count / fps)
                    negative_intervals.append(time_in_seconds)

        frame_count += 1

    cap.release()

    summary = Counter(predictions)
    negative_ratio = (summary["Negative"] / sum(summary.values())) * 100 if predictions else 0

    # 연속적인 구간 계산 및 포맷팅
    formatted_intervals = []
    if negative_intervals:
        start = negative_intervals[0]
        for i in range(1, len(negative_intervals)):
            if negative_intervals[i] - negative_intervals[i-1] > 0.5:
                end = negative_intervals[i-1]
                duration = end - start
                if duration >= 1.0:  # 1초 이상인 경우
                    # 범위 포맷 추가 (00:00 포함 가능)
                    formatted_intervals.append(
                        f"{int(start//60):02}:{int(start%60):02} - {int(end//60):02}:{int(end%60):02}"
                    )
                elif start > 0:  # 단일 시간 포맷에서 00:00 제외
                    formatted_intervals.append(f"{int(start//60):02}:{int(start%60):02}")
                start = negative_intervals[i]
        # 마지막 구간 처리
        end = negative_intervals[-1]
        duration = end - start
        if duration >= 1.0:
            formatted_intervals.append(
                f"{int(start//60):02}:{int(start%60):02} - {int(end//60):02}:{int(end%60):02}"
            )
        elif start > 0:  # 단일 시간 포맷에서 00:00 제외
            formatted_intervals.append(f"{int(start//60):02}:{int(start%60):02}")

    return f"Negative 비율: {negative_ratio:.2f}%", formatted_intervals
