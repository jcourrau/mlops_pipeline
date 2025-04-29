import os
import boto3

def download_from_s3(
        bucket_name:str,
        s3_prefix: str,
        local_dir: str

):
    session = boto3.Session()

    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)

    print(f"Downloading data from s3://{bucket_name}/{s3_prefix}...")

    for obj in bucket.objects.filter(Prefix=s3_prefix):
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        if obj.key.endswith('/'):
            continue

        target_path = os.path.join(
            local_dir,
            os.path.relpath(obj.key, s3_prefix)
        )
        target_folder = os.path.dirname(target_path)

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        print(f"Downloading {obj.key} to {local_dir}"
              f"...")
        try:
            # Download to the full file path, not just the directory
            bucket.download_file(obj.key, target_path)
        except Exception as e:
            print(f"Error downloading {obj.key}: {str(e)}")
            continue

    print("Data download complete.")

if __name__ == "__main__":
    # Download the data from S3
    download_from_s3(
        bucket_name="jason-mlops",
        s3_prefix="ETL_data",
        local_dir="./data/"
    )



