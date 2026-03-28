"""S3 utility helpers for uploading and downloading datasets, models, and reports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError


def get_s3_client(region: str = "us-east-1"):
    return boto3.client("s3", region_name=region)


def upload_file(
    local_path: str | Path,
    bucket: str,
    s3_key: Optional[str] = None,
    region: str = "us-east-1",
) -> str:
    """Upload a local file to S3. Returns the s3:// URI."""
    local_path = Path(local_path)
    s3_key = s3_key or local_path.name
    client = get_s3_client(region)
    client.upload_file(str(local_path), bucket, s3_key)
    uri = f"s3://{bucket}/{s3_key}"
    print(f"Uploaded {local_path} -> {uri}")
    return uri


def download_file(
    bucket: str,
    s3_key: str,
    local_path: str | Path,
    region: str = "us-east-1",
) -> Path:
    """Download a file from S3 to a local path."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_s3_client(region)
    client.download_file(bucket, s3_key, str(local_path))
    print(f"Downloaded s3://{bucket}/{s3_key} -> {local_path}")
    return local_path


def upload_directory(
    local_dir: str | Path,
    bucket: str,
    s3_prefix: str = "",
    region: str = "us-east-1",
) -> list[str]:
    """Recursively upload a local directory to S3. Returns list of URIs."""
    local_dir = Path(local_dir)
    uris = []
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_dir)
            key = f"{s3_prefix}/{relative}".lstrip("/")
            uri = upload_file(file_path, bucket, key, region)
            uris.append(uri)
    return uris


def bucket_exists(bucket: str, region: str = "us-east-1") -> bool:
    client = get_s3_client(region)
    try:
        client.head_bucket(Bucket=bucket)
        return True
    except ClientError:
        return False


def create_bucket_if_missing(bucket: str, region: str = "us-east-1") -> None:
    if bucket_exists(bucket, region):
        print(f"Bucket already exists: {bucket}")
        return
    client = get_s3_client(region)
    if region == "us-east-1":
        client.create_bucket(Bucket=bucket)
    else:
        client.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )
    print(f"Created bucket: {bucket}")


def provision_buckets(config: dict) -> None:
    """Create all project S3 buckets from config if they do not exist."""
    region = config["aws"]["region"]
    for _name, bucket in config["aws"]["buckets"].items():
        create_bucket_if_missing(bucket, region)
