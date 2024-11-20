import os
import logging
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from textblob import TextBlob
import spacy
from pyspark.sql.types import StringType

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

nlp = spacy.load("en_core_web_sm")

class DataIngestion:
    def __init__(self, iceberg_catalog, minio_config):
        self.iceberg_catalog = iceberg_catalog
        self.minio_config = minio_config
        self.spark = self._initialize_spark_session()
        
    def _initialize_spark_session(self):
        spark = SparkSession.builder \
            .appName("s3_to_iceberg") \
            .config("spark.sql.catalog.sentiment_db", "org.apache.iceberg.spark.SparkCatalog") \
            .config("spark.ui.showConsoleProgress", "true") \
            .config("spark.sql.catalog.sentiment_db.type", "hadoop") \
            .config("spark.sql.catalog.sentiment_db.warehouse", self.iceberg_catalog) \
            .config("spark.sql.defaultCatalog", "sentiment_db") \
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
            .config("spark.hadoop.fs.s3a.access.key", self.minio_config['access_key']) \
            .config("spark.hadoop.fs.s3a.secret.key", self.minio_config['secret_key']) \
            .config("spark.hadoop.fs.s3a.endpoint", self.minio_config['endpoint']) \
            .config("spark.hadoop.fs.s3a.path.style.access", "true") \
            .getOrCreate()
        
        logging.info("Spark session created successfully.")
        return spark
    
    def create_iceberg_table(self, table_name):
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS sentiment_db.sentiment_catalog.{table_name} (
                text STRING,
                label STRING,
                word_count INT,
                char_count INT,
                avg_word_length FLOAT,
                polarity FLOAT,
                subjectivity FLOAT,
                noun_count INT,
                verb_count INT,
                adj_count INT,
                entity_count INT
            ) USING iceberg;
        """
        try:
            self.spark.sql(create_table_sql)
            logging.info(f"Table {table_name} created successfully in Iceberg.")
        except Exception as e:
            logging.error(f"Error creating Iceberg table {table_name}: {e}")
    
    def load_parquet_from_s3(self, s3_path: str):
        logging.info(f"Loading Parquet file from S3: {s3_path}")
        try:
            df = self.spark.read.parquet(s3_path)
            logging.info(f"Successfully loaded Parquet file from S3: {s3_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading Parquet file from S3: {s3_path}. Error: {e}")
            return None
    
    def save_to_iceberg(self, df, table_name, mode="overwrite"):
        try:
            if df is None or df.count() == 0:
                logging.warning("The DataFrame is null or empty. No data to save.")
            else:
                logging.info("The DataFrame is not null and has data. Proceeding to save.")

                df.write \
                    .format("iceberg") \
                    .mode(mode) \
                    .option("mergeSchema", "true") \
                    .saveAsTable(table_name)

                logging.info(f"Data saved to Iceberg table: {table_name}")
        except Exception as e:
            logging.error(f"Error saving to Iceberg: {e}")
        
    def add_extra_features(self, df):
        df = df.withColumn("word_count", F.size(F.split(F.col("text"), " ")))
        df = df.withColumn("char_count", F.length(F.col("text")))
        df = df.withColumn("avg_word_length", F.col("char_count") / F.col("word_count"))
        
        df = df.withColumn("polarity", F.udf(lambda text: TextBlob(text).sentiment.polarity)(F.col("text")))
        df = df.withColumn("subjectivity", F.udf(lambda text: TextBlob(text).sentiment.subjectivity)(F.col("text")))

        def pos_counts(text):
            doc = nlp(text)
            return {"noun_count": sum(1 for token in doc if token.pos_ == "NOUN"),
                    "verb_count": sum(1 for token in doc if token.pos_ == "VERB"),
                    "adj_count": sum(1 for token in doc if token.pos_ == "ADJ"),
                    "entity_count": len(doc.ents)}
        
        pos_count_udf = F.udf(lambda text: pos_counts(text), "map<string,int>")
        pos_df = df.withColumn("pos_counts", pos_count_udf(F.col("text"))).select(
            "*", F.col("pos_counts")["noun_count"].alias("noun_count"),
            F.col("pos_counts")["verb_count"].alias("verb_count"),
            F.col("pos_counts")["adj_count"].alias("adj_count"),
            F.col("pos_counts")["entity_count"].alias("entity_count")
        ).drop("pos_counts")
        
        return pos_df

    def transform_nlp_sentiment_scoring(self, df):
        logging.info("Columns in DataFrame before renaming: %s", df.columns)

        split_col = F.split(F.col(df.columns[0]), ",")       
        remove_mentions_udf = F.udf(lambda text: " ".join([word for word in text.split() if not word.startswith("@")]), StringType())
        df = df.withColumn("text", remove_mentions_udf(split_col.getItem(0)))
        logging.info("Columns in DataFrame after adding 'text': %s", df.columns)
        
        df = df.withColumn("label", split_col.getItem(2))
        
        df = df.withColumn("label", F.when(F.col("label").contains("Quite_positive"), "positive")
                                            .when(F.col("label").contains("Quite_negative"), "negative")
                                            .when(F.col("label").contains("Positive"), "positive")
                                            .when(F.col("label").contains("Negative"), "negative")
                                            .otherwise("neutral"))
        
        logging.info("Columns in DataFrame before adding extra features: %s", df.columns)
        df = self.add_extra_features(df)
        
        logging.info("Columns in DataFrame after adding extra features: %s", df.columns)
        
        df = df.select("text", "label", "word_count", "char_count", "avg_word_length",
                    "polarity", "subjectivity", "noun_count", "verb_count", "adj_count", "entity_count")
        
        logging.info("Columns in DataFrame after renaming: %s", df.columns)

        return df

    def transform_sentiment_analysis_financial_news_v1(self, df):
        logging.info("Columns in DataFrame before renaming: %s", df.columns)
        split_col = F.split(F.col(df.columns[0]), ",", 2)
        df = df.withColumn("label", split_col.getItem(0))
        
        df = df.withColumn("text", F.trim(split_col.getItem(1)))
        df = df.filter(F.col("text").isNotNull())
        remove_quotes_udf = F.udf(lambda text: text.strip('"'), StringType())
        df = df.withColumn("text", remove_quotes_udf(F.col("text")))

        logging.info("Columns in DataFrame after extracting 'label' and 'text': %s", df.columns)
        
        df = df.withColumn("label", F.when(F.col("label").contains("positive"), "positive")
                                            .when(F.col("label").contains("negative"), "negative")
                                            .otherwise("neutral"))
        
        logging.info("Columns in DataFrame before adding extra features: %s", df.columns)
        df = self.add_extra_features(df)
        
        logging.info("Columns in DataFrame after adding extra features: %s", df.columns)
        df = df.select("text", "label", "word_count", "char_count", "avg_word_length",
                    "polarity", "subjectivity", "noun_count", "verb_count", "adj_count", "entity_count")
        
        logging.info("Columns in DataFrame after renaming: %s", df.columns)

        return df
    
    def transform_chatgpt_sentiment_analysis(self, df):
        logging.info("Columns in DataFrame before renaming: %s", df.columns)
        df = df.withColumnRenamed("tweets", "text") \
            .withColumnRenamed("labels", "label") \
            .withColumn("label", F.when(F.col("label") == "bad", "negative")
                                    .when(F.col("label") == "good", "positive")
                                    .otherwise("neutral"))
        logging.info("Columns in DataFrame after renaming and label mapping: %s", df.columns)
        logging.info("Columns in DataFrame before adding extra features: %s", df.columns)
        
        df = self.add_extra_features(df)
        logging.info("Columns in DataFrame after adding extra features: %s", df.columns)
        df = df.select("text", "label", "word_count", "char_count", "avg_word_length",
                    "polarity", "subjectivity", "noun_count", "verb_count", "adj_count", "entity_count")
        logging.info("Columns in DataFrame after selecting final columns: %s", df.columns)

        return df
    
    def transform_news_sentiment_analysis(self, df):
        logging.info("Columns in DataFrame before processing: %s", df.columns)
        
        df = df.withColumn("text", F.col("Description")) \
            .withColumn("label", F.col("Sentiment"))
        
        df.show(5)
        
        df = df.filter(F.col("text").isNotNull() & (F.col("text") != ""))       
        null_text_count = df.filter(F.col("text").isNull()).count()
        logging.info(f"Rows with null 'text': {null_text_count}")
        logging.info("Columns in DataFrame after extracting 'text', 'label': %s", df.columns)

        def remove_quotes(text):
            if text is not None:
                return text.strip('"')
            return text

        remove_quotes_udf = F.udf(remove_quotes, StringType())
        df = df.withColumn("text", remove_quotes_udf(F.col("text")))
        df.show(5)

        logging.info("Columns in DataFrame after adding extra features: %s", df.columns)

        df = self.add_extra_features(df)

        logging.info("Columns in DataFrame after adding extra features: %s", df.columns)
        df = df.select("text", "label", "word_count", "char_count", 
                    "avg_word_length", "polarity", "subjectivity", 
                    "noun_count", "verb_count", "adj_count", "entity_count")
        
        logging.info("Columns in DataFrame after final selection: %s", df.columns)
        logging.info(f"Row count after transformation: {df.count()}")
        
        return df
    
    def transform_sentiment_analysis_data(self, df):
        df = df.withColumnRenamed("_c0", "label").withColumnRenamed("_c4", "text") \
               .withColumn("label", F.when(F.col("label") == "0", "negative")
                                       .when(F.col("label") == "2", "neutral")
                                       .when(F.col("label") == "4", "positive"))
        df = self.add_extra_features(df)
        return df.select("text", "label", "word_count", "char_count", "avg_word_length",
                         "polarity", "subjectivity", "noun_count", "verb_count", "adj_count", "entity_count")
    
    def transform_sentiment140(self, df):
        logging.info("Columns in DataFrame before transformation: %s", df.columns)
        
        if len(df.columns) > 0:
            df = df.withColumnRenamed(df.columns[0], "label") \
                .withColumnRenamed(df.columns[1], "timestamp") \
                .withColumnRenamed(df.columns[2], "query") \
                .withColumnRenamed(df.columns[3], "user") \
                .withColumnRenamed(df.columns[4], "tweet_text")
        
        df = df.filter(
            F.concat_ws("", *[F.coalesce(F.col(c), "").alias(c) for c in df.columns]).isNotNull() &
            (F.concat_ws("", *[F.coalesce(F.col(c), "") for c in df.columns]).cast("string") != "")
        )
        
        split_col = F.split(F.col("tweet_text"), ",")
        remove_mentions_udf = F.udf(lambda text: " ".join([word for word in text.split() if not word.startswith("@")]), StringType())
        df = df.withColumn("text", remove_mentions_udf(split_col.getItem(5)))
        
        logging.info("Columns in DataFrame after adding 'text': %s", df.columns)

        df = df.withColumn("label", split_col.getItem(0))
        df = df.withColumn("label", F.when(F.col("label") == "0", "negative")
                                        .when(F.col("label") == "2", "neutral")
                                        .when(F.col("label") == "4", "positive")
                                        .otherwise(F.col("label")))
        
        logging.info("Columns in DataFrame after adding and mapping 'label': %s", df.columns)
        logging.info("Columns in DataFrame before adding extra features: %s", df.columns)
        
        df = self.add_extra_features(df)
        
        logging.info("Columns in DataFrame after adding extra features: %s", df.columns)

        df = df.select("text", "label", "word_count", "char_count", "avg_word_length",
                    "polarity", "subjectivity", "noun_count", "verb_count", "adj_count", "entity_count")

        logging.info("Columns in DataFrame after selecting final columns: %s", df.columns)

        return df

    def sanitize_table_name(self, table_name):
            import re
            return re.sub(r'[^a-zA-Z0-9_]', '_', table_name)
        
    def process_all_datasets(self):
        datasets = {
            "s3a://parquet-bucket/csv/sentiment_scoring.parquet": self.transform_nlp_sentiment_scoring,
            "s3a://parquet-bucket/csv/file.parquet": self.transform_chatgpt_sentiment_analysis,
            "s3a://parquet-bucket/csv/news_sentiment_analysis.parquet": self.transform_news_sentiment_analysis,
        }

        logging.info("Starting table creation process.")
        for file_path in datasets.keys():
            try:
                table_name = os.path.basename(file_path).replace('.parquet', '')
                sanitized_table_name = self.sanitize_table_name(table_name)
                logging.info(f"Creating Iceberg table for {sanitized_table_name} from {file_path}")
                self.create_iceberg_table(sanitized_table_name)
                logging.info(f"Iceberg table {sanitized_table_name} created successfully.")
            except Exception as e:
                logging.error(f"Error creating Iceberg table for {file_path}: {e}")
        
        logging.info("Starting dataset transformation and processing.")
        for s3_path, transform_func in datasets.items():
            try:
                logging.info(f"Loading data from {s3_path}")
                df = self.load_parquet_from_s3(s3_path)
                logging.info(f"Data loaded successfully from {s3_path}")

                logging.info(f"Transforming dataset with function {transform_func.__name__}")
                transformed_df = transform_func(df)
                logging.info(f"Dataset transformed successfully using {transform_func.__name__}")

                table_name = os.path.basename(s3_path).replace(".parquet", "")
                sanitized_table_name = self.sanitize_table_name(table_name)
                iceberg_table_name = f"sentiment_db.sentiment_catalog.{sanitized_table_name}"
                logging.info(f"Saving transformed data to Iceberg table {iceberg_table_name}")
                
                self.save_to_iceberg(transformed_df, iceberg_table_name)
                logging.info(f"Transformed data saved to Iceberg table {iceberg_table_name} successfully.")
            except Exception as e:
                logging.error(f"Error processing dataset from {s3_path}: {e}")
        
        logging.info("All datasets processed successfully.")

if __name__ == "__main__":
    minio_config = {
        "access_key": os.getenv('MINIO_ROOT_USER'),
        "secret_key": os.getenv('MINIO_ROOT_PASSWORD'),
        "endpoint": os.getenv('MINIO_HOST')
    }
    data_ingestion = DataIngestion(
        iceberg_catalog="s3a://parquet-bucket/iceberg",
        minio_config=minio_config
    )
    data_ingestion.process_all_datasets()
