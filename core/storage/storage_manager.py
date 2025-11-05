import boto3
import os
from datetime import datetime, timedelta, timezone
class R2Client:
    _instance = None
    
    def __new__(cls
                ,token_value="mNmh_3httCrl529qHNzq8XRwW2v02NY3aVzZnXq0"
                ,endpoint_url="https://b29348b0eb08b6f4b38ef37555861653.r2.cloudflarestorage.com"
                ,aws_access_key_id="d2d4beeca9da169a19da6ccfb5a7a5d5"
                ,ws_secret_access_key="f36a23173e3c1d05a0c019e5ae58758b8e59567d03bad34a97c4aab9f2ed11f8"):
        if cls._instance is None:
            cls._instance = super(R2Client, cls).__new__(cls)
            token_value = token_value
            cls._instance.s3_client = boto3.client(
                's3',
                endpoint_url=endpoint_url,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=ws_secret_access_key
            )
        return cls._instance

    def download_from_r2(self, object_key, local_file_path='', bucket_name='lockness'):
        """
        Download a file from Cloudflare R2 and save it to a local path.
        
        :param object_key: The key of the object in R2 (e.g., 'path/to/file.txt')
        :param local_file_path: The local path to save the downloaded file
        :param bucket_name: The name of the R2 bucket (default: 'lockness')
        """
        local_file_path = os.path.join(local_file_path,object_key)
        if not os.path.exists(os.path.dirname(local_file_path)):
            print(f"Creating directories for path: {local_file_path}")
            os.makedirs(os.path.dirname(local_file_path))
        self.s3_client.download_file(bucket_name, object_key, local_file_path)
        return local_file_path

    def download_from_r2_with_dirs(self, object_key, local_base_dir, bucket_name='lockness'):
        """
        Download a file from Cloudflare R2, creating the local folder structure to match the R2 object key.
        
        :param object_key: The key of the object in R2 (e.g., 'path/to/file.txt')
        :param local_base_dir: The base local directory where the folder structure will be created
        :param bucket_name: The name of the R2 bucket (default: 'lockness')
        """
        local_file_path = os.path.join(local_base_dir, object_key)
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        self.s3_client.download_file(bucket_name, object_key, local_file_path)
        print(f"File downloaded successfully to {local_file_path} (with directories created)")

    def upload_to_r2(self, object_key, local_file_path, bucket_name='lockness'):
        """
        Upload a local file to Cloudflare R2.
        
        :param object_key: The key to assign to the object in R2 (e.g., 'path/to/file.txt')
        :param local_file_path: The local path of the file to upload
        :param bucket_name: The name of the R2 bucket (default: 'lockness')
        """
        self.s3_client.upload_file(local_file_path, bucket_name, object_key)


    def upload_folder_to_r2(self, local_folder_path, r2_base_key='', bucket_name='lockness', skip_file=''):
        """
        Upload an entire local folder (recursively) to Cloudflare R2, preserving the folder structure in object keys.
        """
        print(f"Processing directory: {local_folder_path}")
        for root, dirs, files in os.walk(local_folder_path):
            for file in files:
                print(f"Processing sub directory: {file}")
                local_file = os.path.join(root, file)

                # Skip unwanted file
                if file == skip_file:
                    continue

                # Compute relative path inside the folder
                relative_path = os.path.relpath(local_file, local_folder_path)

                # Combine with base key (if any) and normalize to S3-style
                object_key = os.path.join(r2_base_key, relative_path).replace("\\", "/")

                print(f"Uploading {local_file} -> r2://{bucket_name}/{object_key}")

                # Upload to R2
                self.s3_client.upload_file(local_file, bucket_name, object_key)



    def Removefile(self,day,bucket_name='lockness'):
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=day)
        LOG_FILE = 'deleted_files.txt'
        # Open log file for writing
        with open(LOG_FILE, 'w') as log:
            log.write("Deleted files and their prefixes:\n")

            # Paginate through all objects in the bucket
            paginator = self.s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        last_modified = obj['LastModified']

                        # Check if object is older than 1 day
                        if last_modified < cutoff_time:
                            # Extract prefix (everything before the last '/')
                            prefix = '/'.join(key.split('/')[:-1]) + '/' if '/' in key else ''
                            file_name = key.split('/')[-1]

                            # Log the file and prefix
                            log.write(f"Prefix: {prefix}\nFile: {file_name}\nFull Key: {key}\n\n")

                            # Delete the object
                            self.s3_client.delete_object(Bucket=bucket_name, Key=key)
                            print(f"Deleted: {key}")

        print(f"Deletion complete. Log saved to {LOG_FILE}")