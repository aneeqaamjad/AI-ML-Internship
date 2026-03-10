# AI & Machine Learning Internship – Task Submissions
## Phase 1

This repository contains the completed tasks for the **AI & Machine Learning Internship**.  
Each task focuses on a core concept of Machine Learning, Deep Learning, or AI application development.

---

##  Task 1: Data Preprocessing & Exploration
**Objective:**  
Clean, explore, and understand a dataset before applying ML models.

**Key Work:**
- Handled missing values
- Removed irrelevant columns
- Performed exploratory data analysis (EDA)
- Used Pandas, NumPy, Matplotlib, and Seaborn

**Skills Gained:**
- Data cleaning
- Feature understanding
- Exploratory analysis

---

##  Task 2: Classification Model
**Objective:**  
Build a classification model and evaluate its performance.

**Key Work:**
- Data splitting (train/test)
- Logistic Regression / Decision Tree
- Accuracy, confusion matrix, classification report

**Skills Gained:**
- Supervised learning
- Model evaluation
- Feature-label relationship

---

##  Task 3: Regression Model
**Objective:**  
Predict continuous values using regression.

**Key Work:**
- Linear Regression
- Feature selection
- Model training and prediction
- Mean Squared Error (MSE)

**Skills Gained:**
- Regression concepts
- Performance metrics
- Real-world prediction modeling

---

##  Task 4: Deep Learning Model
**Objective:**  
Build a basic neural network for prediction/classification.

**Key Work:**
- Neural network using TensorFlow / Keras or PyTorch
- Hidden layers & activation functions
- Model training and evaluation

**Skills Gained:**
- Neural networks
- Deep learning fundamentals
- Loss functions & optimization

---

##  Task 5: Mental Health Support Chatbot (Fine-Tuned LLM)
**Objective:**  
Build an empathetic chatbot for emotional and mental health support.

**Model Used:**  
- DistilGPT2 (fine-tuned)

**Dataset:**  
- Emotion-based conversational dataset (Kaggle)

**Key Work:**
- Data preprocessing for conversational format
- Fine-tuning LLM using Hugging Face Trainer
- Empathetic response generation
- Command-line chatbot interface

**Skills Gained:**
- Large Language Model fine-tuning
- NLP conversation modeling
- Ethical & empathetic AI design

---

##  Task 6: Model Evaluation & Error Handling
**Objective:**  
Analyze errors and improve model robustness.

**Key Work:**
- Identified common errors (NameError, missing imports)
- Debugged ML pipeline issues
- Improved code structure and reliability

**Skills Gained:**
- Debugging ML workflows
- Error analysis
- Model stability improvement

---

##  Tools & Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / PyTorch
- Hugging Face Transformers
- Jupyter Notebook

---


## Phase 2

# Task 1: News Topic Classifier Using BERT

## Objective
Fine-tune a transformer model (BERT) to classify news headlines into topic categories.

## Model Used
- **BERT-base-uncased** (fine-tuned)

## Dataset
- AG News Dataset (Hugging Face)

## Key Work
- Tokenized and preprocessed the dataset
- Fine-tuned bert-base-uncased using Hugging Face Transformers
- Evaluated using accuracy and F1-score
- Deployed using Streamlit for live interaction

## Skills Gained
- NLP using Transformers
- Transfer learning & fine-tuning
- Evaluation metrics for text classification
- Lightweight model deployment

## Results
- Accuracy: 94.2%
- F1-Score: 0.942

## Tools Used
- Python, PyTorch
- Hugging Face Transformers
- Streamlit
- Scikit-learn

---

# Task 2: End-to-End ML Pipeline with Scikit-learn

## Objective
Build a reusable and production-ready machine learning pipeline for predicting customer churn.

## Models Used
- Logistic Regression
- Random Forest (with GridSearchCV)

## Dataset
- Telco Churn Dataset

## Key Work
- Implemented preprocessing (scaling, encoding) using Pipeline
- Trained Logistic Regression and Random Forest
- Used GridSearchCV for hyperparameter tuning
- Exported complete pipeline using joblib

## Skills Gained
- ML pipeline construction
- Hyperparameter tuning with GridSearch
- Model export and reusability
- Production-readiness practices

## Results
- Best Model: Tuned Random Forest
- Accuracy: 80.5%
- ROC-AUC: 0.86

## Tools Used
- Python
- Scikit-learn (Pipeline, GridSearchCV)
- Pandas, NumPy
- Joblib
- Matplotlib, Seaborn

---

# Task 5: Auto Tagging Support Tickets Using LLM

## Objective
Automatically tag support tickets into categories using a large language model (LLM).

## Model Used
- Facebook BART-large-MNLI (zero-shot classification)

## Dataset
- Free-text Support Ticket Dataset (custom)

## Key Work
- Used prompt engineering with LLM
- Compared zero-shot vs few-shot performance
- Applied few-shot learning with examples
- Output top 3 most probable tags per ticket

## Skills Gained
- Prompt engineering
- LLM-based text classification
- Zero-shot and few-shot learning
- Multi-class prediction and ranking

## Results
| Method | Accuracy | Top-3 Accuracy |
|--------|----------|----------------|
| Zero-Shot | 75.0% | 87.5% |
| Few-Shot | 87.5% | 95.0% |

## Tools Used
- Python
- Hugging Face Transformers
- PyTorch
- Scikit-learn
- Pandas, NumPy

---
