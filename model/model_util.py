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
    state_dict = torch.load(model_path, map_location=device)
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
    얼굴 이미지를 입력으로 받아 예측 결과와 확률을 반환합니다.

    :param face: 얼굴 이미지 (OpenCV 이미지)
    :param model: PyTorch 모델
    :return: 예측된 클래스 이름과 확률 값
    """
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_pil = transforms.ToPILImage()(face_rgb)
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(face_tensor)
        probabilities = torch.softmax(outputs, dim=1)  # 소프트맥스 확률 계산
        _, predicted = torch.max(outputs, 1)

    return CLASS_NAMES[predicted.item()], probabilities[0][predicted.item()].item()


def analyze_video(video_path, model):
    """
    비디오를 분석하고 부정적인 구간과 비율을 반환합니다.

    :param video_path: 비디오 파일 경로
    :param model: PyTorch 모델
    :return: 분석 결과 딕셔너리 (부정 비율 및 구간)
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
                try:
                    label, probability = predict_face(face, model)  # 확률 값도 반환
                    predictions.append(label)
                    if label == "Negative":
                        time_in_seconds = (frame_count / fps)
                        negative_intervals.append((time_in_seconds, probability))
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    continue

        frame_count += 1

    cap.release()

    # Negative 비율 계산 (시간 기반)
    total_negative_time = sum(
        t2[0] - t1[0] for t1, t2 in zip(negative_intervals[:-1], negative_intervals[1:])
        if t2[0] - t1[0] <= 0.5
    )

    total_video_time = frame_count / fps if fps > 0 else 1  # FPS가 0일 경우 기본값 설정
    negative_ratio = (total_negative_time / total_video_time) * 100

    # 연속 구간 병합 및 강도 추가
    merged_intervals = []

    if negative_intervals:
        start, intensity_sum = negative_intervals[0][0], negative_intervals[0][1]
        count = 1

        for i in range(1, len(negative_intervals)):
            current_time, current_intensity = negative_intervals[i]
            prev_time = negative_intervals[i - 1][0]

            if current_time - prev_time > 1.0:  # 간격이 1초 이상이면 새로운 구간 시작
                avg_intensity = intensity_sum / count
                intensity_label = "매우 부정적" if avg_intensity > 0.8 else "약간 부정적"
                merged_intervals.append((start, prev_time, intensity_label))
                start, intensity_sum, count = current_time, current_intensity, 1
            else:
                intensity_sum += current_intensity
                count += 1

        # 마지막 구간 추가
        avg_intensity = intensity_sum / count
        intensity_label = "매우 부정적" if avg_intensity > 0.8 else "약간 부정적"
        merged_intervals.append((start, negative_intervals[-1][0], intensity_label))

    return {
        "negative_ratio": round(negative_ratio, 2),
        "negative_summary": f"전체 영상 중 {round(negative_ratio, 2)}%가 부정적으로 분류되었습니다.",
        "negative_intervals": [
            {"start": f"{int(start // 60):02}:{int(start % 60):02}",
             "end": f"{int(end // 60):02}:{int(end % 60):02}",
             "intensity": intensity} for start, end, intensity in merged_intervals
        ]
    }
