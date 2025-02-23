import boto3
import logging
import json
import re
from urllib.parse import urlparse

# AWS 클라이언트 초기화
s3_client = boto3.client('s3')
transcribe_client = boto3.client('transcribe')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    try:
        logger.info(f"Received Event: {json.dumps(event)}")

        # 1. Transcription Job 이름 추출
        detail = event.get('detail', {})
        job_name = detail.get('TranscriptionJobName')
        if not job_name:
            logger.error("TranscriptionJobName 누락")
            return {"status": "INVALID_EVENT"}

        # 2. Transcription Job 정보 조회
        job_response = transcribe_client.get_transcription_job(
            TranscriptionJobName=job_name
        )
        transcription_job = job_response['TranscriptionJob']

        # 3. 태그 추출
        tags = transcription_job.get('Tags', [])
        # 태그에서 interview-id, question-id, content (조건부 추가)
        metadata_to_add = {}
        for tag in tags:
            if tag['Key'] == 'interview-id':
                metadata_to_add['interview-id'] = tag['Value']
            elif tag['Key'] == 'question-id':
                metadata_to_add['question-id'] = tag['Value']
            elif tag['Key'] == 'content':
                metadata_to_add['content'] = tag['Value']

        # 4. S3 URI 파싱
        transcript_uri = transcription_job['Transcript']['TranscriptFileUri']
        if transcript_uri.startswith("https://"):
            parsed_url = urlparse(transcript_uri)
            path_segments = parsed_url.path.lstrip("/").split("/", 1)
            bucket_name = path_segments[0]
            object_key = path_segments[1] if len(path_segments) > 1 else ""
        elif transcript_uri.startswith("s3://"):
            s3_path = transcript_uri.replace("s3://", "", 1)
            bucket_name, object_key = s3_path.split("/", 1)
        else:
            logger.error(f"유효하지 않은 URI 형식: {transcript_uri}")
            return {"status": "INVALID_URI"}

        # 5. 기존 메타데이터 가져오기 및 병합
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        existing_metadata = response.get('Metadata', {})

        # 새로운 메타데이터 병합
        new_metadata = existing_metadata.copy()
        new_metadata.update(metadata_to_add)  # 조건부로 추가된 메타데이터 병합

        # 5. 메타데이터 추가
        s3_client.copy_object(
            Bucket=bucket_name,
            CopySource={'Bucket': bucket_name, 'Key': object_key},
            Key=object_key,
            MetadataDirective='REPLACE',
            Metadata=new_metadata
        )

        logger.info(f"메타데이터 추가 완료: {new_metadata}")
        return {"status": "SUCCESS"}

    except Exception as e:
        logger.error(f"처리 실패: {e}")
        raise
