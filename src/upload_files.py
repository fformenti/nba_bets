import os
import sagemaker
from nba_bets.src.constants import LOCAL_TEAMS_CITIES_CONFERENCES_FILE

BUCKET_NAME = "formenti-nba-bets"
INPUT_DATA_PREFIX = "processed"


def upload_data_to_s3():
    print("\n=== Uploading data to S3 ===")
    input_files = ["regular_season.csv", LOCAL_TEAMS_CITIES_CONFERENCES_FILE]
    sagemaker_session = sagemaker.Session()
    for local_file in input_files:
        if os.path.exists(local_file):
            s3_uri = sagemaker_session.upload_data(
                path=local_file, bucket=BUCKET_NAME, key_prefix=INPUT_DATA_PREFIX
            )
            print(f"Uploaded {local_file} to {s3_uri}")
        else:
            print(f"Warning: {local_file} not found. Please ensure the file exists.")

    return f"s3://{BUCKET_NAME}/{INPUT_DATA_PREFIX}"
