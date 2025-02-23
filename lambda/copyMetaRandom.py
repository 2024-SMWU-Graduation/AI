import boto3
import json
import logging
import re

# AWS 클라이언트 초기화
s3_client = boto3.client('s3')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def is_valid_file_path(file_path):
    valid_path_format = r'^user/\w+/random/\w+/script/.+\.json$'
    return re.match(valid_path_format, file_path) is not None

def lambda_handler(event, context):
    try:
        # EventBridge 이벤트에서 데이터 추출
        detail = event['detail']
        job_name = detail['TranscriptionJobName']

        # S3에서 metadata 파일 경로 생성
        # job_name 형식: transcription-{user_id}-{interview_id}-{uuid}
        user_id, interview_id = job_name.split('-')[1], job_name.split('-')[2]
        metadata_file_key = f"user/{user_id}/random/{interview_id}/metadata/{job_name}.json"

        logger.info(f"Attempting to access metadata file: Bucket=easy-terview-smwu, Key={metadata_file_key}")

        # S3에서 metadata 파일 존재 여부 확인
        try:
            s3_client.head_object(Bucket="easy-terview-smwu", Key=metadata_file_key)
            logger.info(f"Metadata file exists: {metadata_file_key}")
        except s3_client.exceptions.NoSuchKey:
            logger.error(f"Metadata file not found: {metadata_file_key}")
            return {"message": "Metadata file not found"}

        # metadata 파일 가져오기
        response = s3_client.get_object(Bucket="easy-terview-smwu", Key=metadata_file_key)
        metadata = json.loads(response['Body'].read().decode('utf-8'))

        bucket_name = metadata['bucket_name']
        output_key = metadata['output_key']
        question_id = metadata['question_id']
        content = metadata['content']

        # 파일 경로 검증
        if not is_valid_file_path(output_key):
            logger.warning(f"Invalid file path: {output_key}")
            return {"message": "Invalid file path"}

        # S3에서 트랜스크립션 결과 JSON (대본 데이터) 가져오기
        response = s3_client.get_object(Bucket=bucket_name, Key=output_key)
        transcript_data = json.loads(response['Body'].read().decode('utf-8'))

        # 대본 데이터에 metadata 값 추가
        transcript_data['custom_metadata'] = {
            "question-id": question_id,
            "content": content
        }

        # 수정된 JSON 파일을 기존 경로에 업로드 (덮어쓰기)
        s3_client.put_object(
            Bucket=bucket_name,
            Key=output_key,
            Body=json.dumps(transcript_data, ensure_ascii=False).encode('utf-8'),
            ContentType='application/json'
        )

        logger.info(f"Transcription result updated successfully at {output_key}")
    except Exception as e:
        logger.error(f"Error processing transcription result: {e}")
        raise e

    return {"message": "Transcription result updated successfully"}
