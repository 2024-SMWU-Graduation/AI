import os
import json
import boto3
import re
import requests
import logging
import uuid
from urllib.parse import unquote
from openai import OpenAI
import traceback

# 로깅 설정
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 및 OpenAI 클라이언트 설정
s3_client = boto3.client('s3')
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def send_feedback_to_api(question_id, analyze_link):  # link
    """
    분석 결과를 외부 API로 전송
    """
    api_url = "http://54.180.100.5:8080/api/feedback/random"

    payload = {
        "questionId": question_id,
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
        error_details = traceback.format_exc()  # 전체 스택 트레이스 가져오기
        logger.error(f"link API 전송 오류: {e}\n상세 오류: {error_details}")

    if response is not None:
        logger.error(f"응답 코드: {response.status_code}, 응답 내용: {response.text}")

    return None


def send_question_to_api(interview_id, question_data):  # question 자체 스트링값
    """
    분석 결과를 외부 API로 전송
    """
    api_url = "http://54.180.100.5:8080/api/interview/random/question/tail"

    payload = {
        "interviewId": interview_id,
        "questionData": question_data
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"꼬리질문 API 전송 오류: {e}")
        return None


def analyze_script(question, answer):
    """
    OpenAI GPT-4를 사용하여 질문과 답변을 분석합니다.
    JSON 형식으로 follow_up_question, positive_feedback, improvements를 반환합니다.
    """
    question = question.encode('utf-8').decode('utf-8')
    answer = answer.encode('utf-8').decode('utf-8')

    prompt = f"""
    아래는 질문과 사용자(면접자)의 답변입니다.

    면접관의 질문: {question}
    면접자의 답변: {answer}

    위 내용을 바탕으로 아래 항목들을 포함한 순수 JSON 형식의 응답을 만들어 주세요.
    1. follow_up_question: 답변에 대한 꼬리 질문.
    2. 잘한 점: 대본에서 긍정적인 부분.
    3. 보완할 점: 대본에서 개선이 필요한 부분.
    4. 보완된 대본 예시: 보완된 대본의 간단한 예시를 제공합니다.

    """

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system",
             "content": "너는 현직자로 현재 면접관을 하고 있는 상태야. 면접자의 직무관련 답변을 읽고, 어떤지 피드백해주고, 답변에 대해 추가 질문하는 역할이야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=1000,
    )

    feedback = response.choices[0].message.content
    return feedback


def lambda_handler(event, context):
    """
    AWS Lambda 핸들러:
    - SNS 메시지로 전달받은 S3 이벤트를 처리합니다.
    - S3에서 JSON 파일을 다운로드하여 `custom_metadata`의 `content`를 질문으로 사용합니다.
    - JSON 내 `transcripts` 값을 답변으로 사용하여 OpenAI API로 분석을 수행합니다.
    - 결과를 `user/{userid}/random/{interviewid}/feedback/` 경로에 저장합니다.
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

        # S3 이벤트에서 버킷 이름과 객체 키 추출
        bucket_name = s3_event['Records'][0]['s3']['bucket']['name']
        key = s3_event['Records'][0]['s3']['object']['key']
        logger.info(f"Processing file - Bucket: {bucket_name}, Key: {key}")

        # 파일 경로 검증
        match = re.match(r'^user/(\d+)/random/(\d+)/script/.*\.json$', key)
        if not match:
            return {
                "statusCode": 400,
                "body": json.dumps({
                                       "error": "Invalid file path. Must be in the format user/{userid}/random/{interviewid}/script/filename.json"})
            }
        userid = match.group(1)
        interviewid = match.group(2)

        # S3에서 JSON 파일 다운로드
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=key)
            file_content = response['Body'].read().decode('utf-8')
            logger.info(f"File Content Retrieved: {file_content[:200]}...")
        except Exception as e:
            logger.error(f"파일 다운로드 오류: {e}")
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "Error retrieving file from S3."})
            }

        try:
            json_data = json.loads(file_content)
        except json.JSONDecodeError as e:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": f"Invalid JSON format: {str(e)}"})
            }

        # JSON 내 `custom_metadata`에서 질문(content) 추출 및 디코딩
        question_encoded = json_data.get("custom_metadata", {}).get("content", "")
        question_id = json_data.get("custom_metadata", {}).get("question-id", "")
        if not question_encoded:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing question in JSON metadata."})
            }
        question = unquote(question_encoded)  # URL 디코딩

        # JSON 내 `transcripts`에서 답변 추출
        answer = json_data.get("results", {}).get("transcripts", [{}])[0].get("transcript", "")
        if not answer:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No transcript content found in the input file."})
            }

        # OpenAI API를 이용한 분석 수행
        feedback_json = analyze_script(question, answer)

        # 결과 저장용 피드백 파일 경로 생성
        original_filename = key.split("/")[-1]
        feedback_key = f"user/{userid}/random/{interviewid}/feedback/feedback-{original_filename}"

        # GPT 응답이 순수 JSON 형식인지 확인 후 파싱
        try:
            feedback_data = json.loads(feedback_json)
        except json.JSONDecodeError:
            feedback_data = {"feedback": feedback_json}

        # 원본 질문에 대한 답변 추가
        feedback_data["question"] = question
        feedback_data["answer"] = answer

        # S3에 피드백 JSON 파일 저장
        s3_client.put_object(
            Bucket=bucket_name,
            Key=feedback_key,
            Body=json.dumps(feedback_data, ensure_ascii=False),
            ContentType='application/json'
        )

        # 저장된 피드백 파일의 URL 생성
        feedback_link = f"https://{bucket_name}.s3.ap-northeast-2.amazonaws.com/{feedback_key}"
        logger.info(f"Feedback saved at: {feedback_link}")

        follow_up_question = feedback_data.get("follow_up_question", "")
        print(follow_up_question)

        # 링크 API 전송
        link_response = send_feedback_to_api(question_id, feedback_link)
        question_response = send_question_to_api(interviewid, follow_up_question)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "Feedback 생성 및 저장 완료.",
            "link_response": link_response,
            "question_response": question_response,
        }, ensure_ascii=False)
    }
