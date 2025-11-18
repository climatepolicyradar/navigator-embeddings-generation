import json

import numpy as np

from src.s3 import (
    validate_s3_pattern,
    s3_object_read_text,
    write_json_to_s3,
    save_ndarray_to_s3_as_npy,
)


def test_validate_s3_pattern(test_file_key):
    """Test that validate_s3_pattern returns the correct bucket, client and key."""
    bucket, key, s3client = validate_s3_pattern(f"s3://{test_file_key}")

    assert bucket == "test-bucket"
    assert key == "test_prefix/test_id.json"
    assert s3client is not None

    try:
        validate_s3_pattern("random_string")
    except Exception as e:
        assert "Key does not represent an s3 path: random_string" in str(e)


def test_s3_object_read_text(pipeline_s3_client, test_file_key, test_file_json):
    """Test that we can read text from an s3 object."""
    assert json.loads(s3_object_read_text(f"s3://{test_file_key}")) == test_file_json


def test_write_json_to_s3(pipeline_s3_client, s3_bucket_and_region, test_file_json):
    """Test that we can write json to an s3 object."""
    write_json_to_s3(
        json.dumps(test_file_json),
        f"s3://{s3_bucket_and_region['bucket']}/prefix/test.json",
    )
    assert (
        json.loads(
            s3_object_read_text(
                f"s3://{s3_bucket_and_region['bucket']}/prefix/test.json"
            )
        )
        == test_file_json
    )


def test_save_ndarray_to_s3_as_npy(pipeline_s3_client, s3_bucket_and_region):
    """Test that we can save an ndarray to s3."""
    save_ndarray_to_s3_as_npy(
        np.array([1, 2, 3]), f"s3://{s3_bucket_and_region['bucket']}/prefix/test.npy"
    )

    response = pipeline_s3_client.list_objects_v2(
        Bucket=s3_bucket_and_region["bucket"], Prefix="prefix/test.npy"
    )
    contents = response.get("Contents", [])
    assert any(obj["Key"] == "prefix/test.npy" for obj in contents)

    try:
        save_ndarray_to_s3_as_npy(
            np.array([1, 2, 3]), "s3://random_bucket/prefix/test.npy"
        )
    except Exception as e:
        assert "Bucket random_bucket does not exist" in str(e)
