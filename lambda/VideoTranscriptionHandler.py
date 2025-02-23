import boto3
import os
import time
import uuid
import logging
import re
import json
from urllib.parse import unquote

# AWS 클라이언트 초기화
s3_client = boto3.client('s3')
transcribe_client = boto3.client('transcribe')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_s3_metadata(bucket_name, object_key):
    """
    S3 head_object를 이용하여 사용자 정의 메타데이터(interview-id) 가져오기
    """
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        metadata = response.get('Metadata', {})
        interview_id = metadata.get('interview-id', None)  # 메타데이터 키는 소문자로 저장됨
        return interview_id
    except Exception as e:
        logger.error(f"S3 메타데이터 조회 오류: {e}")
        logger.info(f"Bucket Name: {bucket_name}, Object Key: {object_key}")
        return None


def lambda_handler(event, context):
    try:
        # SNS 메시지 파싱 및 S3 객체 정보 추출
        sns_message = event['Records'][0]['Sns']['Message']
        s3_event = json.loads(sns_message)
        bucket_name = s3_event['Records'][0]['s3']['bucket']['name']
        object_key = unquote(s3_event['Records'][0]['s3']['object']['key'])
    except Exception as e:
        logger.error(f"S3 이벤트 파싱 실패: {e}")
        return {"message": "Invalid SNS message"}

    # 파일 경로와 확장자 검증
    if not re.match(r'^user/\w+/introduce/video/.+\.(mp4|mp3|wav|flac)$', object_key):
        logger.warning(f"잘못되었거나 지원되지 않는 파일 경로: {object_key}")
        return {"message": "잘못되었거나 지원되지 않는 파일 경로"}

    # userId 추출
    user_id = object_key.split('/')[1]

    # S3 객체의 사용자 정의 메타데이터에서 interview-id 가져오기
    interview_id = get_s3_metadata(bucket_name, object_key)
    if not interview_id:
        logger.warning("S3 메타데이터에 'interview-id'가 없습니다.")

    # 고유한 Transcription Job 이름 생성
    job_name = f"transcription-{user_id}-{uuid.uuid4()}"
    media_uri = f"s3://{bucket_name}/{object_key}"

    try:
        # Transcription Job 시작
        response = transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': media_uri},
            MediaFormat=object_key.split('.')[-1].lower(),
            LanguageCode='ko-KR',
            OutputBucketName=bucket_name,
            OutputKey=f"user/{user_id}/introduce/script/{job_name}.json",
            Tags=[{'Key': 'interview-id', 'Value': interview_id}]
        )

        logger.info(f"Transcription job 성공적으로 시작됨: {response}")

    except Exception as e:
        logger.error(f"Transcription job 시작 오류: {e}")
        raise e

    return {"message": f"Transcription job {job_name} started"}
