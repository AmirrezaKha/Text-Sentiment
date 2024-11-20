import os
import logging
from pyspark.sql import SparkSession
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Tokenizer
from sparknlp.annotator import RoBertaForSequenceClassification
from pyspark.ml import Pipeline
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql.functions import col, expr, lower, upper
import mlflow


class SentimentAnalysisPipeline:
    def __init__(self):
        """Initialize logging, Spark session, and pre-trained pipeline."""
        self.logger = self.setup_logger()
        self.logger.info("Starting Sentiment Analysis Pipeline...")

        self.spark = self.initialize_spark()
        self.logger.info("Spark session initialized.")

        self.pipeline_1 = self.load_pretrained_pipeline()
        self.pipeline_2 = self.load_custom_pipeline()

    @staticmethod
    def setup_logger():
        """Set up logger for the pipeline."""
        logger = logging.getLogger("SentimentAnalysisPipeline")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def initialize_spark(self):
        """Initialize Spark session with Iceberg and MinIO configurations."""
        self.logger.info("Initializing Spark session with Iceberg and MinIO configuration...")

        minio_user = os.getenv('MINIO_ROOT_USER')
        minio_password = os.getenv('MINIO_ROOT_PASSWORD')
        minio_host = os.getenv('MINIO_HOST')

        self.logger.info(f"MinIO Host: {minio_host}")

        spark = SparkSession.builder \
            .appName("s3_to_iceberg") \
            .config("spark.sql.catalog.sentiment_db", "org.apache.iceberg.spark.SparkCatalog") \
            .config("spark.sql.catalog.sentiment_db.type", "hadoop") \
            .config("spark.sql.catalog.sentiment_db.warehouse", "s3a://parquet-bucket/iceberg") \
            .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1") \
            .config("spark.sql.defaultCatalog", "sentiment_db") \
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.access.key", minio_user) \
            .config("spark.hadoop.fs.s3a.secret.key", minio_password) \
            .config("spark.hadoop.fs.s3a.endpoint", minio_host) \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .config("spark.driver.memory", "16g") \
            .config("spark.executor.memory", "16g") \
            .getOrCreate()

        return spark

    def load_pretrained_pipeline(self):
        """Load the first pre-trained Sentiment RoBERTa model pipeline."""
        try:
            pipeline = PretrainedPipeline("sentiment_roberta_large_english_3_classes_pipeline", lang="en")
            self.logger.info("Pre-trained pipeline (pipeline_1) loaded successfully.")
            return pipeline
        except Exception as e:
            self.logger.error(f"Error loading first pre-trained pipeline: {e}")
            return None

    def load_custom_pipeline(self):
        """Load the second custom RoBERTa pipeline."""
        try:
            document_assembler = DocumentAssembler() \
                .setInputCol("text") \
                .setOutputCol("document")

            tokenizer = Tokenizer() \
                .setInputCols("document") \
                .setOutputCol("token")

            seq_classifier = RoBertaForSequenceClassification.pretrained(
                "roberta_classifier_base_indonesian_1.5g_sentiment_analysis_smsa",
                lang="en"
            ).setInputCols(["document", "token"]) \
                .setOutputCol("class")

            pipeline = Pipeline(stages=[document_assembler, tokenizer, seq_classifier])
            self.logger.info("Custom RoBERTa pipeline (pipeline_2) created successfully.")
            return pipeline
        except Exception as e:
            self.logger.error(f"Error creating custom pipeline: {e}")
            return None

    def load_data(self):
        """Load data from Iceberg table."""
        self.logger.info("Reading Iceberg table...")
        try:
            iceberg_table_location = "sentiment_db.sentiment_catalog.news_sentiment_analysis"
            df = self.spark.read.format("iceberg").load(iceberg_table_location)
            self.logger.info("Data loaded from Iceberg table successfully.")
            return df
        except Exception as e:
            self.logger.error(f"Error reading Iceberg table: {e}")
            return None

    def run_analysis(self, data, pipeline, pipeline_name):
        """Run sentiment analysis using the specified pipeline."""
        if pipeline:
            self.logger.info(f"Running sentiment analysis with {pipeline_name}...")
            try:
                if isinstance(pipeline, PretrainedPipeline):
                    annotations = pipeline.transform(data)
                else:
                    pipeline_model = pipeline.fit(data)
                    annotations = pipeline_model.transform(data)

                self.logger.info(f"Sentiment analysis with {pipeline_name} completed.")
                return annotations
            except Exception as e:
                self.logger.error(f"Error during sentiment analysis with {pipeline_name}: {e}")
                return None
        else:
            self.logger.error(f"{pipeline_name} is not available. Cannot run analysis.")
            return None


    def evaluate_accuracy(self, results, pipeline_name):
        """Evaluate the accuracy of predictions by comparing with the label column."""
        self.logger.info(f"Evaluating accuracy of predictions for {pipeline_name}...")
        try:
            if pipeline_name == "pipeline_2":
                predictions = results.withColumn("predicted_label", expr("CASE " 
                    "WHEN array_contains(class.result, 'POSITIVE') THEN 'POSITIVE' "
                    "WHEN array_contains(class.result, 'NEGATIVE') THEN 'NEGATIVE' "
                    "WHEN array_contains(class.result, 'NEUTRAL') THEN 'NEUTRAL' END"))
                predictions = predictions.withColumn("predicted_label", upper(col("predicted_label")))
                predictions = predictions.withColumn("label", upper(col("label")))
            else:
                predictions = results.withColumn("predicted_label", expr("CASE " 
                    "WHEN array_contains(class.result, 'positive') THEN 'positive' "
                    "WHEN array_contains(class.result, 'negative') THEN 'negative' "
                    "WHEN array_contains(class.result, 'neutral') THEN 'neutral' END"))

            correct_predictions = predictions.filter(col("label") == col("predicted_label")).count()
            total_predictions = predictions.count()
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            self.logger.info(f"Accuracy for {pipeline_name}: {accuracy:.2%} ({correct_predictions}/{total_predictions} correct predictions)")

            self.log_to_mlflow(pipeline_name, accuracy)
        except Exception as e:
            self.logger.error(f"Error during accuracy evaluation for {pipeline_name}: {e}")


    def show_results(self, annotations):
        """Display results of the sentiment analysis."""
        if annotations:
            self.logger.info("Displaying results...")
            annotations.select("text", "label", "class.result").show(truncate=False)
        else:
            self.logger.error("No annotations to display.")


    def log_to_mlflow(self, pipeline_name, accuracy):
        """Log accuracy to MLflow."""
        try:
            mlflow.set_experiment("Sentiment Analysis Pipelines")
            
            with mlflow.start_run(run_name=pipeline_name):
                mlflow.log_param("pipeline_name", pipeline_name)
                mlflow.log_metric("accuracy", accuracy)
                self.logger.info(f"Accuracy for {pipeline_name} logged to MLflow: {accuracy:.2%}")
        except Exception as e:
            self.logger.error(f"Error logging to MLflow for {pipeline_name}: {e}")

# Main Execution
if __name__ == "__main__":
    sentiment_analysis = SentimentAnalysisPipeline()

    data = sentiment_analysis.load_data()

    if data:
        results_1 = sentiment_analysis.run_analysis(data, sentiment_analysis.pipeline_1, "pipeline_1")

        if results_1:
            sentiment_analysis.evaluate_accuracy(results_1, "pipeline_1")
            sentiment_analysis.show_results(results_1)

        results_2 = sentiment_analysis.run_analysis(data, sentiment_analysis.pipeline_2, "pipeline_2")

        if results_2:
            sentiment_analysis.evaluate_accuracy(results_2, "pipeline_2")
            sentiment_analysis.show_results(results_2)

    sentiment_analysis.spark.stop()

