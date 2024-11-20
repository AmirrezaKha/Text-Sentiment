
# MLOps Sentiment Text Analysis 🚀

This project combines **Data Engineering** and **Data Science** workflows to build an end-to-end pipeline for sentiment analysis using **modern MLOps tools**. By leveraging **Dockerized microservices**, this system integrates **MLflow**, **MinIO**, **Apache Iceberg**, and **Apache Spark**, creating a scalable and efficient platform for processing, analyzing, and modeling sentiment datasets.

## 🌟 Project Purpose

The main goal of this project is to process and analyze **sentiment datasets from Kaggle** through a robust pipeline:

1. **Data Ingestion**:
   - Download datasets via the Kaggle API.
   - Store raw data as **Parquet files** in **MinIO** (an S3-compatible object storage).

2. **Data Transformation**:
   - Add **new features** (e.g., word embeddings, text metadata) to the text data.
   - Save the transformed datasets into **Apache Iceberg tables** for efficient querying, updates, and machine learning workflows.

3. **Machine Learning**:
   - Train models to:
     - **Classify sentiment labels**.
     - Use **pretrained Spark NLP models from John Snow Labs** for advanced sentiment prediction tasks.
   - Track model performance and metadata using **MLflow**.

---

## 🛠️ Tools and Technologies

| **Component**        | **Purpose**                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| **MinIO**            | S3-like storage for raw and processed data.                                |
| **Apache Iceberg**   | Manage large datasets with schema evolution and incremental updates.       |
| **Apache Spark**     | Perform distributed data transformations and train ML models.              |
| **Spark NLP**        | Pretrained NLP models by John Snow Labs for advanced text processing.      |
| **MLflow**           | Monitor and compare machine learning experiments.                         |
| **Metaflow**         | Simplify and orchestrate the entire pipeline.                             |
| **Docker**           | Containerized deployment of all services for scalability and ease of use. |

---

## ⚙️ Workflow Explanation

### **1️⃣ Data Ingestion**
- Download the sentiment dataset from Kaggle using the **Kaggle API**.
- Save the raw dataset as **Parquet files** in **MinIO**.

### **2️⃣ Data Transformation**
- Use **Apache Spark** to process raw data and add extra features:
  - Generate **text embeddings**.
  - Create additional metadata for textual analysis.
- Store the enhanced datasets in **Apache Iceberg tables**, enabling efficient querying and schema evolution.

### **3️⃣ Machine Learning**
- Train models using **Spark MLlib** for sentiment classification.
- Leverage **Spark NLP** with **John Snow Labs pretrained models** for English to predict sentiment as an independent task.
- Log all training runs, hyperparameters, and metrics in **MLflow**.

### **4️⃣ Experiment Tracking**
- Use the **MLflow UI** to:
  - Track and visualize experiments.
  - Compare multiple models' performance.
  - Manage model deployment stages.

## 🚀 Deployment Instructions

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- Kaggle API credentials (`kaggle.json` file)

### Steps to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AmirrezaKha/Text-Sentiment.git
   cd mlops_sentiment_text
   ```

2. **Set Environment Variables**:
   - Update the `.env` file with your environment settings:
     ```plaintext
     MINIO_ROOT_USER=minio_admin
     MINIO_ROOT_PASSWORD=minio_password
     MINIO_BUCKET=my_bucket
     MINIO_HOST=minio:9000
     
     MLFLOW_TRACKING_URI=http://mlflow:5000
     ```

3. **Build and Start Services**:
   ```bash
   docker-compose up --build
   ```

4. **Download Kaggle Dataset**:
   - Place your Kaggle API credentials (`kaggle.json`) in the appropriate directory (`~/.kaggle`).


## 📊 Key Features

- **Flexible Data Storage**: 
  - Store raw and processed data efficiently in **MinIO** and **Iceberg tables**.
  
- **Advanced NLP**:
  - Use **John Snow Labs pretrained models** in **Spark NLP** for sentiment prediction.

- **Scalable Machine Learning**:
  - Distributed training on **Apache Spark**.
  - ML experiment tracking and lifecycle management with **MLflow**.

- **End-to-End Automation**:
  - Pipelines orchestrated using **Metaflow** for seamless execution.

## 📈 Future Enhancements
- Integrate **LLM-based sentiment analysis models**.
- Enable real-time processing with streaming frameworks.
- Add CI/CD pipelines for automated deployment and monitoring.


## 📝 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.
