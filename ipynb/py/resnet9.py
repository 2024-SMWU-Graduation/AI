import torch
import torch.nn as nn
import torch.nn.functional as F
from image_classification import ImageClassificationBase


# 기본적인 합성곱 레이어 블록을 구성하는 함수
def conv_block(
    in_channels,       # 입력 채널 수
    out_channels,      # 출력 채널 수
    pool=False,        # 풀링 레이어 포함 여부
    kernel_size=3,     # 합성곱 커널 크기
    padding=1,         # 패딩 값
    stride=1,          # 스트라이드 값
    pool_kernel_size=2 # 풀링 커널 크기
):
    # 레이어들을 순차적으로 담을 리스트
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        ),
        nn.BatchNorm2d(out_channels),  # 배치 정규화를 통해 학습 속도 및 성능 개선
        nn.ReLU(inplace=True),         # 활성화 함수로 ReLU 사용
    ]

    if pool:
        layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size))  # 풀링이 True인 경우 MaxPool 레이어 추가

    return nn.Sequential(*layers)  # 순차적으로 구성된 블록 반환


# 정확도 계산 함수
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)  # 예측된 클래스 인덱스 추출
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))  # 정답과 일치하는 예측의 비율 계산


# 이미지 분류 기본 클래스 정의
class ImageClassificationBase(nn.Module):
    # 학습 시 수행할 단일 스텝
    def training_step(self, batch):
        inputs, labels = batch                # 배치에서 입력 데이터와 레이블을 분리
        outputs = self(inputs)                # 입력을 모델에 전달해 출력 예측
        loss = F.cross_entropy(outputs, labels)  # 손실 함수로 교차 엔트로피 사용
        acc = accuracy(outputs, labels)       # 정확도 계산
        return {"loss": loss, "acc": acc.detach()}  # 손실과 정확도 반환

    # 검증 시 수행할 단일 스텝
    def validation_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs, labels)
        return {"val_loss": loss.detach(), "val_acc": acc.detach()}

    # 에포크의 손실과 정확도 평균 계산
    def get_metrics_epoch_end(self, outputs, validation=True):
        # 배치의 손실 및 정확도 키 설정
        if validation:
            loss_ = "val_loss"
            acc_ = "val_acc"
        else:
            loss_ = "loss"
            acc_ = "acc"

        # 각 배치의 손실과 정확도 추출 후 평균 계산
        batch_losses = [x[f"{loss_}"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()

        batch_accs = [x[f"{acc_}"] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()

        return {
            f"{loss_}": epoch_loss.detach().item(),
            f"{acc_}": epoch_acc.detach().item(),
        }

    # 에포크의 결과를 출력하는 함수
    def epoch_end(self, epoch, result, num_epochs):
        print(
            f"Epoch: {epoch+1}/{num_epochs} -> lr: {result['lrs'][-1]:.5f} "
            f"loss: {result['loss']:.4f}, acc: {result['acc']:.4f}, "
            f"val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}\n"
        )


# ResNet9 네트워크 정의
class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # 첫 번째 합성곱 블록 (풀링 포함)
        self.conv1 = conv_block(in_channels, 48, pool=True)
        # 두 번째 합성곱 블록 (풀링 포함)
        self.conv2 = conv_block(48, 96, pool=True)
        # 첫 번째 잔차(residual) 블록
        self.res1 = nn.Sequential(
            conv_block(96, 96),     # 합성곱 레이어
            conv_block(96, 96),     # 합성곱 레이어
        )

        # 세 번째 합성곱 블록 (풀링 포함)
        self.conv3 = conv_block(96, 192, pool=True)
        # 네 번째 합성곱 블록 (풀링 포함)
        self.conv4 = conv_block(192, 384, pool=True)

        # 두 번째 잔차(residual) 블록
        self.res2 = nn.Sequential(
            conv_block(384, 384),
            conv_block(384, 384)
        )

        # 분류기 블록
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),   # 모든 특성 지도를 1x1로 축소
            nn.Flatten(),                   # 평탄화
            nn.Dropout(0.2),                # 과적합 방지를 위한 드롭아웃
            nn.Linear(384, num_classes),    # 최종 선형 레이어 (출력 크기: 클래스 수)
        )

        # 네트워크 순서 정의
        self.network = nn.Sequential(
            self.conv1,
            self.conv2,
            self.res1,
            self.conv3,
            self.conv4,
            self.res2,
            self.classifier,
        )

    # 순전파 정의
    def forward(self, xb):
        out = self.conv1(xb)         # 첫 번째 합성곱 블록
        out = self.conv2(out)        # 두 번째 합성곱 블록
        out = self.res1(out) + out   # 첫 번째 잔차 블록과 합성
        out = self.conv3(out)        # 세 번째 합성곱 블록
        out = self.conv4(out)        # 네 번째 합성곱 블록
        out = self.res2(out) + out   # 두 번째 잔차 블록과 합성
        out = self.classifier(out)   # 분류기 블록
        return out
