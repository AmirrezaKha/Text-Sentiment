import os
import glob
import logging
import pandas as pd
import kaggle  
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
from typing import Union


class KaggleDataLoader:
    def __init__(self, data_dir="data", parquet_dir="parquet_data", bucket_name='parquet-bucket'):
        """
        Initialize the KaggleDataLoader class, which is responsible for loading datasets from Kaggle,
        converting CSV files to Parquet format, and uploading them to MinIO storage.

        Args:
            data_dir (str): Directory to store downloaded datasets (default is 'data').
            parquet_dir (str): Directory to store Parquet files (default is 'parquet_data').
            bucket_name (str): The name of the MinIO bucket where Parquet files will be uploaded (default is 'parquet-bucket').

        Returns:
            None
        """
        self.data_dir = data_dir
        self.parquet_dir = parquet_dir

        # Load environment variables
        load_dotenv()

        # Ensure the data directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.parquet_dir, exist_ok=True)

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # MinIO connection details from environment variables
        self.minio_url = os.getenv("MINIO_HOST")
        self.minio_access_key = os.getenv("MINIO_ROOT_USER")
        self.minio_secret_key = os.getenv("MINIO_ROOT_PASSWORD")
        self.bucket_name = bucket_name

        if self.minio_url and self.minio_access_key and self.minio_secret_key:
            try:
                self.minio_client = Minio(
                    self.minio_url,
                    access_key=self.minio_access_key,
                    secret_key=self.minio_secret_key,
                    secure=False
                )
                self.logger.info(f"Successfully connected to MinIO at {self.minio_url}")
            except Exception as e:
                self.logger.error(f"Failed to connect to MinIO at {self.minio_url}: {e}")
                self.minio_client = None
        else:
            self.logger.info("MinIO connection details not provided in environment variables. MinIO will not be used.")
            self.minio_client = None

        # Check Kaggle authentication
        self._authenticate_kaggle()

    def _authenticate_kaggle(self):
        """
        Verifies Kaggle API authentication by looking for kaggle.json in /root/.kaggle.

        Returns:
            None
        """
        try:
            # This will automatically look for the kaggle.json in the correct directory
            kaggle.api.authenticate()
            self.logger.info("Kaggle API authenticated successfully.")
        except Exception as e:
            self.logger.error("Kaggle API authentication failed: %s", e)

    def process_csv_to_parquet(self, dataset_name: str, encoding='ISO-8859-1') -> None:
        """
        Converts all CSV files in the data directory to Parquet files with the same name
        and uploads them to MinIO if available.

        Args:
            dataset_name (str): The name of the Kaggle dataset to download and process.
            encoding (str): The encoding to use when reading CSV files (default is 'ISO-8859-1').

        Returns:
            None
        """
        try:
            # Download and unzip dataset files
            kaggle.api.dataset_download_files(dataset_name, path=self.data_dir, unzip=True)
            self.logger.info(f"Downloaded dataset '{dataset_name}' to '{self.data_dir}' directory")
            csv_files = glob.glob(f"{self.data_dir}/*.csv")
            if not csv_files:
                self.logger.error("No CSV files found in the data directory.")
                return

            for csv_file in csv_files:
                # Create a corresponding Parquet file name
                base_name = os.path.basename(csv_file).replace('.csv', '.parquet')
                parquet_file = os.path.join(self.parquet_dir, base_name)

                # Load CSV into DataFrame
                data = pd.read_csv(csv_file, encoding=encoding)
                self.logger.info(f"Loaded data from '{csv_file}'")

                # Save the DataFrame as a Parquet file
                data.to_parquet(parquet_file, index=False)
                self.logger.info(f"Data saved to '{parquet_file}' as Parquet")

                # Upload to MinIO if available
                if self.minio_client:
                    self._upload_to_minio(parquet_file)
        except Exception as e:
            self.logger.error(f"Error while processing CSV to Parquet: {e}")

    def _upload_to_minio(self, file_path: str):
        """
        Uploads a file to MinIO storage.
        
        Args:
            file_path (str): Path to the file to upload.

        Returns:
            None
        """
        try:
            object_name = os.path.basename(file_path)

            # Ensure the bucket exists
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)

            # Upload the file to MinIO
            self.minio_client.fput_object(self.bucket_name, object_name, file_path)
            self.logger.info(f"File '{file_path}' uploaded to MinIO bucket '{self.bucket_name}'")
        except S3Error as e:
            self.logger.error(f"Failed to upload file to MinIO: {e}")
