import torch
import torch.nn as nn
from torchvision import models

class ResNet18Model(nn.Module):
    """
    Pretrained ResNet18 모델을 기반으로 새로운 출력 레이어를 추가한 클래스.
    Args:
        num_classes (int): 출력 클래스 수
    """
    def __init__(self, num_classes):
        super(ResNet18Model, self).__init__()
        # Pretrained ResNet18 모델 로드
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 출력 레이어 교체
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model(weights_path, num_classes, device):
    """
    모델을 생성하고 저장된 가중치를 로드하는 함수.
    Args:
        weights_path (str): 저장된 모델 가중치 경로
        num_classes (int): 출력 클래스 수
        device (torch.device): 모델을 로드할 디바이스 (CPU/GPU)
    Returns:
        nn.Module: 가중치가 로드된 모델
    """
    # 모델 초기화
    model = ResNet18Model(num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device)

def save_model(model, path):
    """
    모델의 가중치를 저장하는 함수.
    Args:
        model (nn.Module): 저장할 모델
        path (str): 저장할 파일 경로
    """
    torch.save(model.state_dict(), path)
