import boto3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure with your credentials
aws_access_key_id = os.getenv("AWS_ACCESS_KEY")
aws_secret_access_key = os.getenv("AWS_ACCESS_SECRET")

# Create a session
session = boto3.Session(
    aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key
)

# Create S3 client
s3 = session.client("s3")

# List all buckets
response = s3.list_buckets()
print("S3 Buckets:")
for bucket in response["Buckets"]:
    print(f"- {bucket['Name']}")
