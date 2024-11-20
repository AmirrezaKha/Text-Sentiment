import unittest
from unittest.mock import patch, MagicMock
from external.scripts.main_pipeline import DataPipelineFlow
from data_loading import KaggleDataLoader
import subprocess

class TestDataPipelineFlow(unittest.TestCase):
    @patch("data_loading.KaggleDataLoader")
    def test_load_data_success(self, MockKaggleDataLoader):
        # Arrange
        mock_loader = MockKaggleDataLoader.return_value
        mock_loader.process_csv_to_parquet.return_value = None

        # Act and Assert
        try:
            flow = DataPipelineFlow()
            flow.load_data()
        except Exception:
            self.fail("load_data raised an exception unexpectedly.")

    @patch("subprocess.run")
    def test_run_spark_normal_script_success(self, mock_subprocess_run):
        # Arrange
        mock_subprocess_run.return_value = MagicMock(stdout="Job executed successfully", stderr="")

        # Act and Assert
        try:
            flow = DataPipelineFlow()
            flow.run_spark_normal_script()
        except Exception:
            self.fail("run_spark_normal_script raised an exception unexpectedly.")

    @patch("subprocess.run")
    def test_run_spark_nlp_script_success(self, mock_subprocess_run):
        # Arrange
        mock_subprocess_run.return_value = MagicMock(stdout="Job executed successfully", stderr="")

        # Act and Assert
        try:
            flow = DataPipelineFlow()
            flow.run_spark_nlp_script()
        except Exception:
            self.fail("run_spark_nlp_script raised an exception unexpectedly.")

    @patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, 'cmd', output="Error"))
    def test_run_spark_nlp_script_failure(self, mock_subprocess_run):
        # Arrange and Act
        flow = DataPipelineFlow()

        # Assert
        with self.assertRaises(subprocess.CalledProcessError):
            flow.run_spark_nlp_script()

    def test_end_step(self):
        # Arrange
        flow = DataPipelineFlow()

        # Act and Assert
        try:
            flow.end()
        except Exception:
            self.fail("end raised an exception unexpectedly.")

if __name__ == "__main__":
    unittest.main()
