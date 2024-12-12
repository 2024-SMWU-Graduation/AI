import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from ipynb.model import ResNet18Model  # 모델 클래스 임포트
from model.model_util import load_model, analyze_video  # 모델 관련 함수 임포트

# Flask 설정 및 초기화
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 디렉토리(server) 경로 얻기
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')  # uploads 폴더 경로 설정 (server/uploads)
MODEL_PATH = os.path.join(BASE_DIR, '../model/ResNet18_final_best_updated.pth')  # 모델 경로 설정

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

app = Flask(__name__, template_folder='templates')  # templates 폴더 경로 지정 (server/templates)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 디렉토리 생성 확인 (없으면 생성)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 모델 로드 (초기화 시 한 번만 수행)
model = load_model(MODEL_PATH, ResNet18Model)


def allowed_file(filename):
    """
    허용된 파일 확장자인지 확인합니다.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """
    비디오 파일 업로드 및 분석 처리 엔드포인트.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return '파일이 없습니다.', 400

        file = request.files['file']
        if file.filename == '':
            return '파일이 선택되지 않았습니다.', 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # uploads 폴더에 저장

            file.save(filepath)  # 파일 저장

            # 비디오 분석 수행 및 결과 반환
            result = analyze_video(filepath, model)
            return result, 200

    return render_template('upload.html')  # templates/upload.html 렌더링


if __name__ == '__main__':
    app.run(debug=True, port=8080)  # 필요 시 포트를 변경 가능
