import os
import json
import boto3
import re
import requests
from openai import OpenAI
import logging

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 및 OpenAI 설정
s3_client = boto3.client('s3')
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def get_s3_metadata(bucket_name, object_key):
    """
    S3 head_object를 이용하여 사용자 정의 메타데이터(interview-id) 가져오기
    """
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        metadata = response.get('Metadata', {})
        interview_id = metadata.get('interview-id')
        return interview_id
    except Exception as e:
        logger.error(f"S3 메타데이터 조회 오류: {e}")
        return None


def analyze_script(script):
    """
    OpenAI GPT-4를 사용하여 질문과 답변을 분석합니다.
    JSON 형식으로 original_script, positive_feedback, constructive_feedback, improved_answer를 반환합니다.
    """
    # 유니코드 문자를 명시적으로 UTF-8로 처리
    script = script.encode('utf-8').decode('utf-8')

    prompt = f"""
    다음은 대본 텍스트입니다.

    면접자의 1분 자기소개 대본: {script}

    위 내용을 바탕으로 아래 형식의 JSON을 생성해주세요.

    {{
        "original_script": "{script}",
        "positive_feedback": "이 대본에서 잘한 점을 간결하고 구체적으로 작성",
        "constructive_feedback": "이 대본에서 개선이 필요한 점을 명확히 작성",
        "improved_answer": "이 대본을 보완한 예시를 제공"
    }}

    """

    # 최신 OpenAI API 호출 방식
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "너는 경력직 인사담당자로 현재 면접관으로 일하고 있어. 면접자의 1분 자기소개를 읽고 분석해봐."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=1000,
    )

    # GPT의 응답에서 메시지 내용 추출
    # OpenAI 응답에서 JSON 추출 후 파싱
    feedback_json = response.choices[0].message.content  # 응답 내용을 가져옴
    parsed_feedback = json.loads(feedback_json)

    return parsed_feedback


def send_feedback_to_api(interview_id, analyze_link):
    """
    분석 결과를 외부 API로 전송
    """
    api_url = "http://54.180.100.5:8080/api/feedback/introduce"

    payload = {
        "interviewId": interview_id,
        "analyzeLink": analyze_link
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.patch(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API 전송 오류: {e}")
        return None


def lambda_handler(event, context):
    """
    AWS Lambda 핸들러
    """
    for record in event['Records']:
        sns_message = record['Sns']['Message']
        try:
            s3_event = json.loads(sns_message)
            logger.info(f"S3 Event: {s3_event}")
        except json.JSONDecodeError as e:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Invalid SNS message format: {str(e)}"})
            }

        # S3 이벤트에서 파일 정보 추출
        bucket_name = s3_event['Records'][0]['s3']['bucket']['name']
        key = s3_event['Records'][0]['s3']['object']['key']
        logger.info(f"Processing file - Bucket: {bucket_name}, Key: {key}")

        # 파일 경로 검증
        match = re.match(r'^user/(\d+)/introduce/script/.*$', key)
        if not match:
            return {
                "statusCode": 400,
                "body": json.dumps(
                    {"error": "Invalid file path. Must be in the format user/{userid}/introduce/script/."})
            }

        userid = match.group(1)

        # S3에서 JSON 파일 다운로드
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        file_content = response['Body'].read().decode('utf-8')
        logger.info(f"File Content Retrieved: {file_content[:200]}...")

        try:
            script_data = json.loads(file_content)
        except json.JSONDecodeError as e:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Invalid JSON format: {str(e)}"})
            }

        # S3 객체의 메타데이터에서 interview-id 가져오기
        interview_id = get_s3_metadata(bucket_name, key)
        print("Interview ID:", interview_id)
        if not interview_id:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing interview-id in S3 metadata."})
            }

        # 대본 텍스트 가져오기
        script_text = script_data.get("results", {}).get("transcripts", [{}])[0].get("transcript", "")
        print("Script Text:", script_text)
        if not script_text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No transcript content found in the input file."})
            }

        # 대본 분석
        feedback = analyze_script(script_text)
        print(feedback, type(feedback))

        if feedback is None:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "GPT 응답 오류. JSON 데이터가 없습니다."})
            }

        # 결과 저장용 파일 이름 생성
        feedback_key = f'user/{userid}/introduce/feedback/feedback-{key.split("/")[-1].split("-")[-1]}'

        # 결과 저장 (ensure_ascii=False 추가)
        feedback_data = json.dumps(feedback, ensure_ascii=False)

        s3_client.put_object(
            Bucket=bucket_name,
            Key=feedback_key,
            Body=feedback_data,  # UTF-8로 저장
            ContentType='application/json'
        )

        # S3 링크 생성
        analyze_link = f"https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/{feedback_key}"

        # API 전송
        api_response = send_feedback_to_api(interview_id, analyze_link)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Feedback 생성 및 저장 완료.",
            "output_file": feedback_key,
            "api_response": api_response
        }, ensure_ascii=False)
    }
