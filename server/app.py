import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS 설정을 위한 라이브러리
from werkzeug.utils import secure_filename

from ipynb.model import ResNet18Model  # 모델 클래스 임포트
from model.model_util import load_model, analyze_video  # 모델 관련 함수 임포트

# Flask 설정 및 초기화
app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 요청 허용

# 파일 업로드 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 디렉토리(server) 경로
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')  # uploads 폴더 경로
MODEL_PATH = os.path.join(BASE_DIR, '../model/ResNet18_final_best_updated.pth')  # 모델 경로
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 디렉토리 생성 확인 (없으면 생성)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 모델 로드 (서버 시작 시 한 번만 수행)
model = load_model(MODEL_PATH, ResNet18Model)

def allowed_file(filename):
    """
    허용된 파일 확장자인지 확인합니다.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    비디오 파일 업로드 및 분석 처리 API 엔드포인트.
    """
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # uploads 폴더에 저장

        file.save(filepath)  # 파일 저장

        # 비디오 분석 수행 및 결과 반환
        result = analyze_video(filepath, model)
        return jsonify({'message': '분석 완료', 'result': result}), 200

    return jsonify({'error': '허용되지 않는 파일 형식입니다.'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """
    서버 상태 확인을 위한 헬스 체크 API.
    """
    return jsonify({'status': 'OK'}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # 개발 환경에서 디버그 모드 활성화
