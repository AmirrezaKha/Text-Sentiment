import unittest
from unittest.mock import patch, MagicMock
import os
from external.scripts.data_loading import KaggleDataLoader


class TestKaggleDataLoader(unittest.TestCase):

    @patch("kaggle.api.authenticate")
    def test_authenticate_kaggle_success(self, mock_kaggle_authenticate):
        """
        Test successful Kaggle authentication.
        """
        mock_kaggle_authenticate.return_value = None
        loader = KaggleDataLoader()
        loader._authenticate_kaggle()  # Test the method
        mock_kaggle_authenticate.assert_called_once()

    @patch("kaggle.api.dataset_download_files")
    @patch("glob.glob", return_value=["data/test.csv"])
    @patch("pandas.read_csv", return_value=MagicMock())
    @patch("pandas.DataFrame.to_parquet")
    @patch.object(KaggleDataLoader, "_upload_to_minio")
    def test_process_csv_to_parquet(self, mock_upload, mock_to_parquet, mock_read_csv, mock_glob, mock_download):
        """
        Test the process of converting CSV to Parquet and uploading it to MinIO.
        """
        loader = KaggleDataLoader(data_dir="data", parquet_dir="parquet_data")
        dataset_name = "test-dataset"

        # Simulate processing of CSV files
        loader.process_csv_to_parquet(dataset_name=dataset_name)
        
        # Assertions
        mock_download.assert_called_once_with(dataset_name, path="data", unzip=True)
        mock_glob.assert_called_once_with("data/*.csv")
        mock_read_csv.assert_called_once_with("data/test.csv", encoding='ISO-8859-1')
        mock_to_parquet.assert_called_once_with("parquet_data/test.parquet", index=False)
        mock_upload.assert_called_once_with("parquet_data/test.parquet")

    @patch("minio.Minio")
    @patch("minio.Minio.fput_object")
    def test_upload_to_minio_success(self, mock_fput, MockMinio):
        """
        Test uploading a file to MinIO.
        """
        mock_minio_client = MockMinio.return_value
        mock_fput.return_value = None
        loader = KaggleDataLoader(data_dir="data", parquet_dir="parquet_data", bucket_name="parquet-bucket")
        loader.minio_client = mock_minio_client

        # Simulate uploading a file
        file_path = "parquet_data/test.parquet"
        loader._upload_to_minio(file_path)

        # Assertions
        mock_minio_client.fput_object.assert_called_once_with("parquet-bucket", "test.parquet", file_path)

    @patch("minio.Minio")
    def test_upload_to_minio_failure(self, MockMinio):
        """
        Test failure in uploading to MinIO (e.g., bucket error).
        """
        mock_minio_client = MockMinio.return_value
        mock_minio_client.fput_object.side_effect = Exception("MinIO upload error")
        loader = KaggleDataLoader(data_dir="data", parquet_dir="parquet_data", bucket_name="parquet-bucket")
        loader.minio_client = mock_minio_client

        # Simulate uploading a file and catching the exception
        file_path = "parquet_data/test.parquet"
        with self.assertRaises(Exception):
            loader._upload_to_minio(file_path)

if __name__ == "__main__":
    unittest.main()
