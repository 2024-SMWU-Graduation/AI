# 👩‍💻 이지터뷰 (Easy-terview) : AI 면접 피드백 서비스
## AI repository
- 2024 SMWU IT공학전공 졸업프로젝트 이지터뷰 **AI 레포지토리**입니다.
- 표정 인식 모델 개발 및 OpenAI API를 활용한 프롬프트 엔지니어링 진행했습니다.
## 사용한 기술 스택
### AI
[![stackticon](https://firebasestorage.googleapis.com/v0/b/stackticon-81399.appspot.com/o/images%2F1742062624840?alt=media&token=b87f0212-1182-47b3-bfad-a12561784a31)](https://github.com/msdio/stackticon)
### Backend
[![stackticon](https://firebasestorage.googleapis.com/v0/b/stackticon-81399.appspot.com/o/images%2F1742062870669?alt=media&token=90360d80-1489-4c38-b922-f219624e2209)](https://github.com/msdio/stackticon)
### Infra
[![stackticon](https://firebasestorage.googleapis.com/v0/b/stackticon-81399.appspot.com/o/images%2F1742062839976?alt=media&token=88ed61be-693e-4eb6-a956-46129cff85d1)](https://github.com/msdio/stackticon)

## AI 시스템 아키텍쳐
<img width="1988" alt="Image" src="https://github.com/user-attachments/assets/1c229918-f5a2-4b69-b8a7-3d884238a3b9" />

## 표정 인식 모델 학습 과정
1. Kaggle의 FERData를 이용한 ResNet9 모델 학습 (`ResNet9_epoch-100_score-0.8633.pth`)
2. AI-Hub의 한국인 감정 인식을 위한 복합 영상 데이터를 이용한 ResNet18 모델 학습 (`ResNet18_final_best_updated.pth`)
   - 학습시킨 ResNet9 모델을 이용하여 전이학습 진행
## 포팅 매뉴얼
### 사전 준비 사항
다음 소프트웨어가 설치되어 있어야 합니다.
- Python (>= 3.8)
- Git
- 가상 환경 관리 도구 (`venv` 또는 `conda`)
- 필수 라이브러리 (`requirements.txt` 파일 참고)
- GPU (선택 사항, 모델 학습 시 권장)

### 저장소 클론
로컬에 해당 저장소를 클론합니다:

```sh
git clone https://github.com/2024-SMWU-Graduation/AI.git
cd AI
```

### 가상 환경 설정
의존성 관리를 위해 가상 환경을 사용하는 것을 권장합니다.

-  `venv` 사용 시
```sh
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate    # Windows
```

-  `conda` 사용 시
```sh
conda create --name ai_project python=3.8
conda activate ai_project
```

### 필수 라이브러리 설치
가상 환경 활성화 후, 필요한 라이브러리를 설치합니다:

```sh
pip install -r requirements.txt
```

### 서버 실행
Flask 서버를 실행하여 모델 예측을 수행하려면 다음 명령어를 실행합니다.

```sh
python app.py
```