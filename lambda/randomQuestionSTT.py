import boto3
import uuid
import json
import logging
from urllib.parse import unquote

# AWS 클라이언트 초기화
s3_client = boto3.client('s3')
transcribe_client = boto3.client('transcribe')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        # SNS 메시지 파싱 및 S3 객체 정보 추출
        sns_message = event['Records'][0]['Sns']['Message']
        s3_event = json.loads(sns_message)
        bucket_name = s3_event['Records'][0]['s3']['bucket']['name']
        object_key = unquote(s3_event['Records'][0]['s3']['object']['key'])

        # userId와 interviewId 추출
        user_id = object_key.split('/')[1]
        interview_id = object_key.split('/')[3]

        # S3 객체의 메타데이터 가져오기
        head_response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        metadata = head_response.get('Metadata', {})
        question_id = metadata.get('question-id')
        content = metadata.get('content')

        if not question_id or not content:
            logger.warning("메타데이터에 'question-id' 또는 'content'가 없습니다.")
            return {"message": "Metadata is missing required fields"}

        # 고유한 Transcription Job 이름 생성 (interview_id 포함, uuid 하이픈 제거)
        job_name = f"transcription-{user_id}-{interview_id}-{uuid.uuid4().hex}"
        media_uri = f"s3://{bucket_name}/{object_key}"

        # Transcription Job 시작
        output_key = f"user/{user_id}/random/{interview_id}/script/{job_name}.json"
        transcribe_client.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': media_uri},
            MediaFormat=object_key.split('.')[-1].lower(),
            LanguageCode='ko-KR',
            OutputBucketName=bucket_name,
            OutputKey=output_key
        )

        # 메타데이터 저장 (S3에 JSON 파일로 업로드)
        metadata_file_key = f"user/{user_id}/random/{interview_id}/metadata/{job_name}.json"
        metadata_to_save = {
            "job_name": job_name,
            "bucket_name": bucket_name,
            "output_key": output_key,
            "question_id": question_id,
            "content": content
        }
        s3_client.put_object(
            Bucket=bucket_name,
            Key=metadata_file_key,
            Body=json.dumps(metadata_to_save),
            ContentType='application/json'
        )

        logger.info(f"Metadata saved to {metadata_file_key}")
    except Exception as e:
        logger.error(f"Error starting transcription job: {e}")
        raise e

    return {"message": f"Transcription job {job_name} started"}
