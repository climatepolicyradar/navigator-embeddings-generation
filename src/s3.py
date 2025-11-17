import tempfile
from typing import Any

import boto3
import numpy as np
from aws_error_utils.aws_error_utils import errors
from botocore.exceptions import ClientError

from src.config import S3_PATTERN


def validate_s3_pattern(s3_path: str):
    """Validates that a string is a valid s3 path."""
    s3_match = S3_PATTERN.match(s3_path)
    if s3_match is None:
        raise Exception(f"Key does not represent an s3 path: {s3_path}")
    bucket = s3_match.group("bucket")
    key = s3_match.group("prefix")
    s3client = boto3.client("s3")
    return bucket, key, s3client


# TODO do we want to instantiate one client object and pass that through rather than
#  instantiating each time?
def check_file_exists_in_s3(s3_path: str) -> bool:
    """Checks whether a file exists in an S3 bucket."""
    bucket, key, s3client = validate_s3_pattern(s3_path)
    try:
        s3client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False
    except errors.NoSuchBucket:
        return False
    except errors.NoSuchKey:
        return False
    except Exception as e:
        raise e


def s3_object_read_text(s3_path: str) -> str:
    """
    Read text from an S3 object.

    :param s3_path: path to S3 object, including s3:// prefix
    :return str: text of S3 object
    """
    bucket, key, s3client = validate_s3_pattern(s3_path)

    try:
        response = s3client.get_object(Bucket=bucket, Key=key)
    except errors.NoSuchBucket:
        raise ValueError(f"Bucket {bucket} does not exist")
    except errors.NoSuchKey:
        raise ValueError(f"Key {key} does not exist")
    except Exception as e:
        raise e

    return response["Body"].read().decode("utf-8")


def write_json_to_s3(json_data: str, s3_path: str) -> None:
    """Writes JSON data to an S3 bucket."""
    bucket, key, s3client = validate_s3_pattern(s3_path)

    # Upload the JSON string to S3
    try:
        s3client.put_object(Body=json_data, Bucket=bucket, Key=key)
    except errors.NoSuchBucket:
        raise ValueError(f"Bucket {bucket} does not exist")
    except Exception as e:
        raise e


def save_ndarray_to_s3_as_npy(array: Any, s3_path: str) -> None:
    """Saves a NumPy ndarray to an S3 bucket as a .npy file."""
    bucket, key, s3client = validate_s3_pattern(s3_path)

    with tempfile.TemporaryFile() as f:
        np.save(f, array)
        f.seek(0)

        try:
            s3client.upload_fileobj(f, bucket, key)
        except errors.NoSuchBucket:
            raise ValueError(f"Bucket {bucket} does not exist")
        except Exception as e:
            raise e
