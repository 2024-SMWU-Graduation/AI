import cv2
import torch
import numpy as np
from torchvision import transforms
from ipynb.model import ResNet18Model

# 설정값
WEIGHTS_PATH = "model/ResNet18_final_best_updated.pth"  # 모델 가중치 경로
CLASS_NAMES = ["Negative", "Positive"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 전처리 변환
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    # 모델 로드
    model = ResNet18Model(num_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 얼굴 검출
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # 얼굴 영역 추출 및 전처리
            face_img = frame[y:y + h, x:x + w]
            face_tensor = transform(face_img).unsqueeze(0).to(DEVICE)

            # 예측 수행
            with torch.no_grad():
                outputs = model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                conf, preds = torch.max(probabilities, 1)

            # 결과 표시
            label = f"{CLASS_NAMES[preds.item()]} ({conf.item():.2f})"
            color = (0, 0, 255) if CLASS_NAMES[preds.item()] == "Negative" else (0, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # 화면 출력
        cv2.imshow('Webcam Emotion Detection', frame)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
