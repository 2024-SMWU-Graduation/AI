import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image

# 모델 정의
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet18Model, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 모델 경로와 클래스 이름
MODEL_PATH = "/Users/mks/Documents/GitHub/AI/model/ResNet18_best.pth"
CLASS_NAMES = ["Negative", "Neutral", "Positive"]

# GPU 설정
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
model = ResNet18Model(num_classes=len(CLASS_NAMES)).to(device)

# state_dict 키 변환 함수
def adapt_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if not key.startswith("model."):
            new_key = f"model.{key}"
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

# 가중치 로드
try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    adapted_state_dict = adapt_state_dict_keys(state_dict)
    model.load_state_dict(adapted_state_dict)
    print("모델 가중치가 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"모델 가중치 로드 중 오류 발생: {e}")


model = model.to(device)
model.eval()

# 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Haar Cascade로 얼굴 검출
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def predict_emotion(face_image):
    """
    주어진 얼굴 이미지를 ResNet18 모델로 감정(긍정, 부정, 중립)을 예측합니다.
    """
    face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(face_tensor)
        _, predicted = torch.max(outputs, 1)
        return CLASS_NAMES[predicted.item()]

# 실시간 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다. 종료합니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(224, 224))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        emotion = predict_emotion(face)
        
        # 얼굴 영역 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
