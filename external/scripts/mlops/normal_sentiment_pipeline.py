from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, VectorAssembler, StringIndexer
from pyspark.ml.classification import MultilayerPerceptronClassifier, LogisticRegression, RandomForestClassifier, GBTClassifier, NaiveBayes, LinearSVC
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
import os
import mlflow
import mlflow.spark
import logging

class SentimentModelingPipeline:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.spark = None
        self.initialize_spark()
        self.paramGrids = {
            "LogisticRegression": ParamGridBuilder()
                .addGrid(LogisticRegression().regParam, [0.01, 0.1, 1.0])
                .addGrid(LogisticRegression().elasticNetParam, [0.0, 0.5, 1.0])
                .build(),
            "NaiveBayes": ParamGridBuilder()
                .addGrid(NaiveBayes().smoothing, [0.5, 1.0, 1.5])
                .build(),
        }

    def initialize_spark(self):
        """Initialize Spark session."""
        self.logger.info("Initializing Spark session with Iceberg and MinIO configuration...")

        minio_user = os.getenv('MINIO_ROOT_USER')
        minio_password = os.getenv('MINIO_ROOT_PASSWORD')
        minio_host = os.getenv('MINIO_HOST')

        self.spark = SparkSession.builder \
            .appName("s3_to_iceberg") \
            .config("spark.sql.catalog.sentiment_db", "org.apache.iceberg.spark.SparkCatalog") \
            .config("spark.sql.catalog.sentiment_db.type", "hadoop") \
            .config("spark.sql.catalog.sentiment_db.warehouse", "s3a://parquet-bucket/iceberg") \
            .config("spark.sql.defaultCatalog", "sentiment_db") \
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.access.key", minio_user) \
            .config("spark.hadoop.fs.s3a.secret.key", minio_password) \
            .config("spark.hadoop.fs.s3a.endpoint", minio_host) \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .getOrCreate()

        if self.spark is not None:
            self.logger.info("Spark session initialized successfully.")
        else:
            self.logger.error("Failed to initialize Spark session.")
            raise Exception("Failed to initialize Spark session.")

    def load_data(self):
        """Load data from Iceberg table."""
        self.logger.info("Reading Iceberg table...")
        iceberg_table_location = "sentiment_db.sentiment_catalog.news_sentiment_analysis"
        df = self.spark.read.format("iceberg").load(iceberg_table_location)
        self.logger.info("Loaded data from Iceberg table.")
        return df

    def preprocess_data(self, df):
        """Preprocess the data."""
        self.logger.info("Applying Tokenizer...")
        tokenizer = Tokenizer(inputCol="text", outputCol="text_tokens")
        df = tokenizer.transform(df)

        self.logger.info("Applying HashingTF...")
        hashing_tf = HashingTF(inputCol="text_tokens", outputCol="text_features", numFeatures=1000)
        df = hashing_tf.transform(df)

        self.logger.info("Indexing labels for multi-class classification...")

        label_indexer = StringIndexer(inputCol="label", outputCol="label_indexed")
        df = label_indexer.fit(df).transform(df)

        self.logger.info("label_indexer...")
        self.logger.info(df.select("label_indexed").distinct().show())


        self.logger.info("Indexing polarity and subjectivity...")
        polarity_indexer = StringIndexer(inputCol="polarity", outputCol="polarity_indexed")
        subjectivity_indexer = StringIndexer(inputCol="subjectivity", outputCol="subjectivity_indexed")
        df = polarity_indexer.fit(df).transform(df)
        df = subjectivity_indexer.fit(df).transform(df)

        self.logger.info("Assembling features...")
        feature_columns = [col for col in df.columns if col not in ['label', 'text', 'text_tokens', 'polarity', 'subjectivity', 'label_indexed']]
        feature_columns += ['text_features', 'polarity_indexed', 'subjectivity_indexed']
        vec_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        df = vec_assembler.transform(df)

        return df

    def create_model(self, model_name):
        """Create model based on selected model name."""
        if self.spark is None:
            self.initialize_spark()

        if model_name == "LogisticRegression":
            model = LogisticRegression(featuresCol="features", labelCol="label_indexed")
        elif model_name == "RandomForest":
            model = RandomForestClassifier(featuresCol="features", labelCol="label_indexed")
        elif model_name == "GradientBoostedTrees":
            model = GBTClassifier(featuresCol="features", labelCol="label_indexed")
        elif model_name == "NaiveBayes":
            model = NaiveBayes(featuresCol="features", labelCol="label_indexed")
        elif model_name == "LinearSVC":
            model = LinearSVC(featuresCol="features", labelCol="label_indexed")
        elif model_name == "MLP_Neural_Network":
            model = MultilayerPerceptronClassifier(featuresCol="features", labelCol="label_indexed")
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        return model

    def train_and_evaluate(self, df, model_name, paramGrid=None):
        """Train and evaluate the model."""
        self.logger.info(f"Training and evaluating model: {model_name}")

        self.logger.info("Splitting data into training and testing sets...")
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=123)

        model = self.create_model(model_name)

        evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="f1")
        cv = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

        self.logger.info(f"Training the {model_name} model with CrossValidator...")
        cv_model = cv.fit(train_data)
        self.logger.info(f"{model_name} model training completed.")

        self.logger.info("Making predictions on the test set...")
        predictions = cv_model.transform(test_data)

        self.logger.info(f"Evaluating the {model_name} model...")
        f1_score = evaluator.evaluate(predictions)
        self.logger.info(f"{model_name} F1 Score: {f1_score}")

        accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label_indexed", predictionCol="prediction", metricName="accuracy")
        accuracy = accuracy_evaluator.evaluate(predictions)
        self.logger.info(f"{model_name} Accuracy: {accuracy}")

        mlflow.log_metric(f"{model_name}_f1_score", f1_score)
        mlflow.log_metric(f"{model_name}_accuracy", accuracy)
        mlflow.end_run()


    def run_pipeline(self):
            """Run the entire pipeline for all models."""
            mlflow.start_run()
            self.initialize_spark()
            df = self.load_data()
            df = self.preprocess_data(df)

            for model_name, paramGrid in self.paramGrids.items():
                self.train_and_evaluate(df, model_name, paramGrid)
            

    def stop_spark(self):
        """Stop the Spark session."""
        if self.spark is not None:
            self.logger.info("Stopping Spark session...")
            self.spark.stop()
            self.logger.info("Spark session stopped.")

pipeline = SentimentModelingPipeline()
pipeline.run_pipeline()
