import unittest
from unittest.mock import patch
from pyspark.sql import SparkSession
from external.scripts.mlops.llm_sentiment_pipeline import SentimentAnalysisPipeline
class TestSentimentAnalysisPipeline(unittest.TestCase):
    def test_spark_initialization(self):
        """Test if Spark session is initialized with the correct app name."""
        pipeline = SentimentAnalysisPipeline()
        spark_app_name = pipeline.spark.sparkContext.appName
        self.assertEqual(spark_app_name, "s3_to_iceberg", "Spark session should be initialized with the correct app name.")
        pipeline.spark.stop()

    @patch('sparknlp.pretrained.PretrainedPipeline')
    def test_pretrained_pipeline_loading(self, MockPretrainedPipeline):
        """Test loading of the pre-trained pipeline."""
        MockPretrainedPipeline.return_value = True
        pipeline = SentimentAnalysisPipeline()
        self.assertIsNotNone(pipeline.pipeline_1, "Pre-trained pipeline should be loaded successfully.")

if __name__ == "__main__":
    unittest.main()
