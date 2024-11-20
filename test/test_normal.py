import unittest
from unittest.mock import patch, MagicMock
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from external.scripts.mlops.normal_sentiment_pipeline import *

class TestSentimentModelingPipeline(unittest.TestCase):

    def setUp(self):
        """Set up a pipeline instance and Spark session for testing."""
        self.pipeline = SentimentModelingPipeline()
        self.pipeline.spark = SparkSession.builder.master("local[*]").appName("UnitTest").getOrCreate()

    def tearDown(self):
        """Stop the Spark session after tests."""
        if self.pipeline.spark:
            self.pipeline.spark.stop()

    def test_spark_initialization(self):
        """Test that the Spark session initializes correctly."""
        self.assertIsNotNone(self.pipeline.spark, "Spark session should not be None.")
        self.assertEqual(self.pipeline.spark.sparkContext.appName, "s3_to_iceberg", "Spark session should have the correct app name.")

    def test_load_data(self):
        """Test that the Iceberg table loads data successfully (mocked)."""
        with patch.object(self.pipeline.spark.read, 'format', return_value=MagicMock(load=MagicMock(return_value=True))) as mock_read:
            data = self.pipeline.load_data()
            self.assertTrue(data, "Data loading should return a DataFrame.")
            mock_read.assert_called_with("iceberg")

    def test_preprocess_data(self):
        """Test the data preprocessing step."""
        # Create mock data
        schema = StructType([
            StructField("text", StringType(), True),
            StructField("label", StringType(), True),
            StructField("polarity", StringType(), True),
            StructField("subjectivity", StringType(), True)
        ])
        mock_data = self.pipeline.spark.createDataFrame([
            ("I love Spark NLP!", "positive", "1", "0.8"),
            ("Not a fan of this.", "negative", "0", "0.4")
        ], schema)

        processed_data = self.pipeline.preprocess_data(mock_data)

        self.assertTrue("features" in processed_data.columns, "Processed data should include 'features' column.")
        self.assertTrue("label_indexed" in processed_data.columns, "Processed data should include 'label_indexed' column.")

    def test_create_model(self):
        """Test model creation for Logistic Regression."""
        model_name = "LogisticRegression"
        model = self.pipeline.create_model(model_name)
        self.assertEqual(model.__class__.__name__, "LogisticRegression", "Should create a LogisticRegression model.")

    @patch('mlflow.log_metric')
    def test_train_and_evaluate(self, mock_log_metric):
        """Test the training and evaluation process (mocked)."""
        # Create mock data
        schema = StructType([
            StructField("features", StructType([]), True),
            StructField("label_indexed", DoubleType(), True)
        ])
        mock_data = self.pipeline.spark.createDataFrame([
            (Vectors.dense([0.1, 0.2, 0.3]), 1.0),
            (Vectors.dense([0.4, 0.5, 0.6]), 0.0)
        ], schema)

        with patch.object(self.pipeline, 'create_model', return_value=LogisticRegression(featuresCol="features", labelCol="label_indexed")) as mock_model:
            self.pipeline.train_and_evaluate(mock_data, "LogisticRegression")
            mock_log_metric.assert_called()

    def test_run_pipeline(self):
        """Test running the full pipeline (mocked)."""
        with patch.object(self.pipeline, 'load_data', return_value=True), \
             patch.object(self.pipeline, 'preprocess_data', return_value=True), \
             patch.object(self.pipeline, 'train_and_evaluate', return_value=True) as mock_train_evaluate:
            self.pipeline.run_pipeline()
            mock_train_evaluate.assert_called()

if __name__ == "__main__":
    unittest.main()
