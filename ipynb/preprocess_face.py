import os
from multiprocessing import Pool
import cv2

# Haar cascade 분류기를 로드하여 얼굴 검출 준비
haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces_and_resize(img_path: str):
    """
    주어진 이미지 경로에서 얼굴 검출 후 얼굴 이미지를 (224, 224)로 리사이즈하여 저장합니다.

    Args:
        img_path (str): 이미지 파일 경로
    """
    if not os.path.exists(img_path):
        return

    try:
        # 이미지 읽기
        img = cv2.imread(img_path)

        # 회색조로 변환하여 얼굴 검출
        faces = haar_cascade.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(224, 224)
        )

        # 검출된 얼굴 처리
        for x, y, w, h in faces:
            m = max(w, h)  # 정사각형을 만들기 위해 큰 쪽 길이를 기준으로 설정
            face_img = img[y:y + m, x:x + m].copy()  # 얼굴 부분 자르기
            face_img_resized = cv2.resize(face_img, (224, 224))  # 리사이즈

            # 새로운 파일명 생성 및 저장
            base, ext = os.path.splitext(img_path)
            output_path = f"{base}_face{ext}"
            cv2.imwrite(output_path, face_img_resized)

        # 원본 이미지를 삭제하여 중복 저장 방지
        os.remove(img_path)

    except Exception as e:
        print(f"Error processing file {img_path}: {e}")

def process_images_in_subdirectories(root_directory: str):
    """
    지정된 루트 디렉토리 내 모든 하위 디렉토리의 이미지를 처리.

    Args:
        root_directory (str): 이미지가 있는 루트 디렉토리 경로
    """
    if not os.path.exists(root_directory):
        print(f"Root directory does not exist: {root_directory}")
        return

    # 모든 하위 디렉토리 및 파일 탐색
    for dirpath, _, filenames in os.walk(root_directory):
        img_paths = [
            os.path.join(dirpath, f) for f in filenames if not f.endswith("_face.jpg")
        ]

        if img_paths:
            print(f"Processing {len(img_paths)} images in {dirpath}...")

            # 병렬 처리
            with Pool(10) as pool:
                pool.map(detect_faces_and_resize, img_paths)

# 실제 실행 코드
if __name__ == "__main__":
    train_dir = "/Users/mks/Documents/GitHub/AI/archive/train"  # 대상 디렉토리
    process_images_in_subdirectories(train_dir)
