# MediPredict AI – Medical E-Commerce & Insurance Intelligence Platform

## 1. Project Overview

MediPredict AI is a Python-based Machine Learning and Retrieval-Augmented Generation (RAG) demonstration platform designed for medical e-commerce analytics and insurance claim review support.

The project combines multiple Machine Learning techniques with a locally hosted Large Language Model (LLM) to demonstrate how AI can support business intelligence, customer analytics, product recommendations and project knowledge assistance.

**Important:** All datasets used in this project are synthetic. The platform is an educational and technical demonstration, not a real-world medical diagnosis or insurance decision system.

## 2. Project Objectives

- Predict medical e-commerce order quantities using Machine Learning.
- Identify insurance claims that may require additional human review.
- Segment customers based on purchasing behaviour.
- Recommend products using collaborative filtering.
- Build a local AI assistant using Ollama and Llama 3.2 1B.
- Implement RAG using TF-IDF retrieval and cosine similarity.

## 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Information Retrieval | TF-IDF, Cosine Similarity |
| LLM | Llama 3.2 1B |
| Local LLM Runtime | Ollama |
| API Integration | Python Requests |
| Development | Jupyter Notebook, VS Code / CMD |

## 4. Project Architecture

The platform consists of the following modules:

1. Synthetic Data Generation
2. Data Preprocessing and Exploratory Data Analysis
3. Order Quantity Prediction
4. Insurance Claim Review Classification
5. Customer Segmentation
6. Customer Segment Prediction
7. Product Recommendation
8. Local LLM Assistant
9. RAG-Based Project Knowledge Assistant

## 5. Machine Learning Modules

### Module 1 – Order Quantity Prediction

**Objective:** Predict the quantity of products customers may order.

- Algorithm: Random Forest Regressor
- Dataset: 1,000 synthetic medical e-commerce orders
- Target: Quantity
- Features include product ID, category, unit price, discount, customer age, city, payment method, inventory and date-related features.

**Model Evaluation:**

| Metric | Random Forest |
|---|---:|
| MAE | 0.8648 |
| RMSE | 1.0662 |
| R² Score | 0.7373 |

Random Forest performed better than the Linear Regression and mean-baseline models in the synthetic test dataset.

### Module 2 – Insurance Claim Review Classification

**Objective:** Identify insurance claims that may require human review.

- Algorithm: Random Forest Classifier
- Dataset: 1,000 synthetic insurance claims
- Target: Review Flag
- Class 0: Routine
- Class 1: Review required

**Input Features:**
- Customer age
- Policy type
- Policy tenure
- Claim amount
- Document completeness
- Previous claims

**Model Evaluation:**

| Metric | Result |
|---|---:|
| Accuracy | 85% |
| Review Precision | 85.71% |
| Review Recall | 54.55% |
| Review F1 Score | 66.67% |

The model is intended only to demonstrate claim prioritisation for human review. It does not automatically approve or reject insurance claims.

### Module 3 – Customer Segmentation

**Objective:** Group customers based on their purchasing behaviour.

- Algorithm: K-Means Clustering
- Number of clusters: 4
- Feature scaling: StandardScaler

**Features:**
- Order count
- Total spend
- Average order value
- Total quantity
- Average discount
- Unique products
- Cancelled orders
- Returned orders

The resulting clusters represent different customer purchasing patterns.

Cluster IDs are numerical labels, not customer rankings.

### Module 4 – Customer Segment Prediction

A saved StandardScaler and K-Means model are used to predict the segment of a new customer profile.

The prediction uses the same feature structure as the customer segmentation training data.

### Module 5 – Product Recommendation

**Objective:** Recommend products based on customer purchase history.

- Approach: Collaborative Filtering
- Similarity algorithm: Cosine Similarity
- Dataset: Completed purchase transactions
- Customer-product matrix: 194 customers × 10 products

The recommendation engine identifies customers with similar purchase patterns and recommends products that the target customer has not previously purchased.

**Example:**

For customer 1002, who previously purchased P003 and P005, the demonstration recommended:

| Product ID | Product | Price |
|---|---|---:|
| P002 | Glucose Test Strips | ₹650 |
| P010 | First Aid Kit | ₹750 |
| P004 | Digital Thermometer | ₹350 |

**Evaluation:**

| Metric | Result |
|---|---:|
| HitRate@3 | 0.3687 |
| Precision@3 | 0.1229 |

These results are from a synthetic-data leave-one-out evaluation. They should not be interpreted as real-world recommendation performance.

## 6. Local LLM Integration

The project integrates a locally hosted LLM using Ollama.

**Model:** Llama 3.2 1B

The Python application communicates with the local Ollama API:

`http://localhost:11434/api/generate`

This enables local text generation without requiring a paid cloud LLM API key.

## 7. RAG Implementation

The project implements a lightweight Retrieval-Augmented Generation pipeline.

**Retrieval components:**
- Knowledge document: `data/rag/medipredict_knowledge.txt`
- Text processing: TF-IDF Vectorizer
- Similarity calculation: Cosine Similarity
- Retrieval: Top relevant document sections
- Answer generation: Llama 3.2 1B through Ollama

**RAG workflow:**

1. The user submits a question.
2. The retriever loads the project knowledge document.
3. The document is divided into sections.
4. TF-IDF converts the sections into numerical vectors.
5. Cosine similarity ranks relevant sections.
6. Retrieved context is passed to the local LLM.
7. The LLM generates an answer using the supplied project context.

This is a TF-IDF-based retrieval approach, not a dense embedding-based retrieval system.

## 8. Project Structure

```text
MediPredictAI/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── rag/
│       └── medipredict_knowledge.txt
│
├── models/
│   ├── insurance_random_forest.pkl
│   ├── customer_segmentation_scaler.pkl
│   ├── customer_kmeans.pkl
│   ├── customer_product_matrix.pkl
│   └── customer_similarity.pkl
│
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── prepare_ml_data.py
│   ├── train_regression.py
│   ├── compare_regression_models.py
│   ├── insurance_data_generation.py
│   ├── prepare_insurance_data.py
│   ├── train_insurance_models.py
│   ├── analyze_insurance_model.py
│   ├── predict_insurance_claim.py
│   ├── prepare_customer_segments.py
│   ├── scale_customer_features.py
│   ├── train_customer_clusters.py
│   ├── predict_customer_segment.py
│   ├── prepare_recommendation_data.py
│   ├── calculate_customer_similarity.py
│   ├── recommend_products.py
│   ├── evaluate_recommendations_v2.py
│   ├── ollama_test.py
│   ├── ollama_chat.py
│   ├── rag_retriever.py
│   └── rag_chat.py
│
├── README.md
└── requirements.txt
```

*Note: The structure shows the main project files. Additional analysis and helper scripts may also be present.*

## 9. Installation and Setup

### Prerequisites

- Python 3.10
- Ollama
- Llama 3.2 1B model

### Install Python Dependencies

Open CMD and navigate to the project directory:

```cmd
cd /d F:\Python310\MediPredictAI
```

Install dependencies:

```cmd
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib requests
```

### Install and Prepare Ollama

Install Ollama and download the model:

```cmd
ollama pull llama3.2:1b
```

Start the Ollama server:

```cmd
ollama serve
```

Verify the model:

```cmd
ollama list
```

Verify the API:

```cmd
curl http://127.0.0.1:11434/api/tags
```

## 10. Running the RAG Assistant

Keep the Ollama server running in a separate CMD window.

Open another CMD window:

```cmd
cd /d F:\Python310\MediPredictAI
```

Run:

```cmd
python src\rag_chat.py
```

Example questions:

- Which algorithm is used for insurance claim classification?
- What is the accuracy of the insurance model?
- Which algorithm is used for customer segmentation?
- How does the product recommendation system work?
- Does this project predict cancer using medical scan images?

The assistant retrieves relevant project knowledge and generates answers using the local LLM.

## 11. Project Limitations

1. All datasets are synthetic and do not represent actual patient or insurance records.
2. Model evaluation results do not establish real-world performance.
3. Customer segmentation quality is limited by the synthetic dataset and selected features.
4. Product recommendations are demonstration outputs.
5. Insurance predictions are not claim decisions.
6. The platform does not diagnose patients or recommend medical treatment.
7. TF-IDF retrieval may fail to identify relevant information when questions use unfamiliar wording.
8. LLM responses can still contain unsupported information despite context-based prompting.
9. Local LLM performance depends on available RAM and CPU resources.

## 12. Future Enhancements

- Integrate larger and more representative datasets where legally permitted.
- Improve retrieval with semantic embeddings.
- Add a web-based user interface.
- Develop API endpoints for ML predictions.
- Improve recommendation evaluation using time-based testing.
- Add model monitoring and experiment tracking.
- Explore cloud deployment and scalable inference.

## 13. Conclusion

MediPredict AI demonstrates the practical integration of data processing, supervised Machine Learning, unsupervised learning, collaborative filtering, local LLM inference and Retrieval-Augmented Generation.

The project provides a foundation for exploring how AI technologies can support medical e-commerce business intelligence and insurance claim review workflows while keeping human decision-making central.
---
**Project Name:** MediPredict AI
**Programming Language:** Python 3.10
**LLM Runtime:** Ollama
**LLM Model:** Llama 3.2 1B
**Data:** Synthetic demonstration datasets