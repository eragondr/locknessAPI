import boto3
import os

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

    def download_from_r2(self, object_key, local_file_path=None, bucket_name='lockness'):
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
        
        print(f"File downloaded successfully to {local_file_path}")
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
        print(f"File uploaded successfully to R2 at {object_key}")

    def upload_folder_to_r2(self, local_folder_path, r2_base_key='', bucket_name='lockness'):
        """
        Upload an entire local folder (recursively) to Cloudflare R2, preserving the folder structure in object keys.
        
        :param local_folder_path: The local path of the folder to upload
        :param r2_base_key: Optional base key prefix in R2 (e.g., 'folder/') where the structure will be uploaded
        :param bucket_name: The name of the R2 bucket (default: 'lockness')
        """
        print(f"Processing directory: {local_folder_path}")
        for root, dirs, files in os.walk(local_folder_path):
            print(f"Processing directory: {root}")
            for file in files:
                local_file = os.path.join(root, file)
                # relative_path = os.path.relpath(local_file, local_folder_path)
                # object_key = os.path.join(local_file,r2_base_key, relative_path).replace('\\', '/')
                # print(f"Uploading {local_file} to R2 at {object_key}")
                self.s3_client.upload_file(local_file, bucket_name, local_file)
            