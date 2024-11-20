from metaflow import FlowSpec, step
from data_loading import KaggleDataLoader
from dotenv import load_dotenv
import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()

class DataPipelineFlow(FlowSpec):
    @step
    def start(self):
        """
        Initialize the flow by defining the datasets to be processed.

        Returns:
            None
        """
        self.datasets = [
            'ranja7/nlp-sentiment-scoring-noheaderlabel',
            'ankurzing/sentiment-analysis-for-financial-news',
            'charunisa/chatgpt-sentiment-analysis',
            'clovisdalmolinvieira/news-sentiment-analysis',
            'kazanova/sentiment140'
        ]
        logging.info("Starting data pipeline.")
        self.next(self.load_data)
        return None

    @step
    def load_data(self):
        """
        Load datasets using KaggleDataLoader, process them into Parquet format, 
        and upload to a storage bucket.

        Returns:
            None
        """
        try:
            data_loader = KaggleDataLoader(
                data_dir="data",
                parquet_dir="parquet_data",
                bucket_name='parquet-bucket'
            )
            logging.info("Loading datasets.")
            for dataset in self.datasets:
                data_loader.process_csv_to_parquet(dataset_name=dataset)
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            self.next(self.end)
            return None
        self.next(self.run_spark_normal_script)
        return None

    @step
    def run_spark_script(self):
        """
        Execute a Spark job for normal sentiment analysis tasks.

        Returns:
            None
        """
        try:
            command = [
                "sudo", "docker", "exec", "-it", "spark-master",
                "python", "/opt/bitnami/spark/spark_main.py"
            ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logging.info("Spark job normal NLP tasks submitted successfully.")
            logging.debug(f"Output: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            logging.error("Error during Spark job execution.")
            logging.debug(f"Error Output: {e.stderr.strip()}")
            self.next(self.end)
            raise e
        self.next(self.run_spark_nlp_script)
        return None

    @step
    def run_spark_nlp_script(self):
        """
        Execute a Spark job for sentiment analysis tasks using Spark NLP.

        Returns:
            None
        """
        try:
            command = [
                "sudo", "docker", "exec", "-it", "spark-master",
                "python", "/opt/bitnami/spark/normal_sentiment_pipeline.py"
            ]
            # For Clustering system, run following:
            # command = [
            #     "sudo", "docker", "exec", "-it", "spark-master",
            #     "spark-submit", "/opt/bitnami/spark/normal_sentiment_pipeline.py"
            # ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logging.info("Spark job Spark normal tasks submitted successfully.")
            logging.debug(f"Output: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            logging.error("Error during Spark job execution.")
            logging.debug(f"Error Output: {e.stderr.strip()}")
            raise e
        self.next(self.end)
        try:
            command = [
                "sudo", "docker", "exec", "-it", "spark-master",
                "python", "/opt/bitnami/spark/llm_sentiment_pipeline.py"
            ]
            # For Clustering system, run following:
            # command = [
            #     "sudo", "docker", "exec", "-it", "spark-master",
            #     "spark-submit", "/opt/bitnami/spark/llm_sentiment_pipeline.py"
            # ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logging.info("Spark job Spark NLP tasks submitted successfully.")
            logging.debug(f"Output: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            logging.error("Error during Spark job execution.")
            logging.debug(f"Error Output: {e.stderr.strip()}")
            raise e
        self.next(self.end)
        return None

    @step
    def end(self):
        """
        End the flow. Log a message indicating completion.

        Returns:
            None
        """
        logging.info("Pipeline has completed successfully.")
        return None


if __name__ == "__main__":
    DataPipelineFlow()
