# Anomaly Detection in Reviews (NLP)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Key Learnings](#key-learnings)
- [Future Work](#future-work)
- [Business Impact](#business-impact)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Contact](#contact)

---

## Project Overview

This project leverages Natural Language Processing (NLP) and classical machine learning to detect anomalies in Amazon product reviews. The goal is to identify inconsistencies between review text sentiment and the associated star rating, thereby improving the reliability of product feedback for businesses and customers.

### Objective:
The objective here is to build a model that is able to clearly distinguish the sentiments and thereby predict Anomaly records appropriately, so it is equally important to keep both Precision and Recall scores very high. Hence, it is very much significant to achieve a higher F1 score. 

-- 

## Dataset

- **Amazon Fine Food Reviews**: Reviews from 2012, 10 columns.
- **Amazon Grocery & Gourmet**: Reviews from 1996–2014, 9 columns.
- Combined dataset: ~720,000 records.
- Focus: Only 1-star and 5-star reviews for binary sentiment classification.

**Key Attributes:**
- `reviewerID`: Unique customer ID
- `asin`: Product ID
- `reviewText`: Review content
- `overall`: Star rating (1–5)
- ...and others

---

## Approach

1. **Data Cleaning & Preprocessing**
   - Remove duplicates and irrelevant columns
   - Handle missing values
   - Focus on 1-star and 5-star reviews
   - Balance the dataset (40k samples per class)

2. **Text Preprocessing**
   - Lowercasing, accent removal, contraction expansion
   - Remove digits, HTML tags, punctuation, stopwords (customized), repeated characters, URLs, extra spaces

3. **Tokenization & Lemmatization**
   - NLTK word tokenization
   - Lemmatization with POS tagging

4. **Feature Engineering**
   - TF-IDF vectorization (uni-, bi-, tri-grams)

5. **Modeling**
   - Train/test split (80/20)
   - Models: Logistic Regression, Multinomial Naive Bayes, Random Forest, XGBoost
   - Hyperparameter tuning with GridSearchCV (F1-score focus)
   - Cross-validation (7-fold, stratified)

6. **Evaluation**
   - Metrics: Precision, Recall, F1-score, Accuracy, ROC-AUC
   - Confusion matrices and ROC curves
   - Visualizations: Word clouds, metric comparison plots

7. **Anomaly Detection**
   - Manual sentiment labeling on a test sample (Cochran's formula for sample size)
   - Compare model predictions to manual sentiment to flag anomalies

---

## Results

- **Best Sentiment Classifier:** Multinomial Naive Bayes (F1-score ≈ 0.926)
- **Best Anomaly Detector:** Logistic Regression (Weighted F1-score up to 0.98)
- **Key Metrics:**
  - F1-score (Sentiment): 0.92+
  - F1-score (Anomaly): 0.92–0.98 (Logistic Regression)
- **Visualizations:**
  - Word clouds for positive/negative reviews
  - Confusion matrices and ROC curves for each model

For a given dataset, the best sentiment classifier may not necessarily turn out to perform the same in the case of Anomaly classification. Here, a high F1 score for Sentiment Classification was achieved by the Multinomial Naïve Bayes model. However, when it comes to Anomaly classification, Logistic Regression model has given the highest F1 score.

---

## Key Learnings

- Built an end-to-end NLP pipeline for text-based anomaly detection
- Applied advanced text preprocessing and custom stopword handling
- Compared multiple ML models and performed hyperparameter tuning
- Used cross-validation and custom metrics for robust evaluation
- Demonstrated the difference between sentiment classification and anomaly detection performance

---

## Future Work

- Expand sentiment classes (e.g., group 1–3 stars as negative, 4–5 as positive)
- Experiment with deep learning models (BERT, LSTM)
- Use advanced embeddings (Word2Vec, GloVe, transformers)
- Automate anomaly feedback loop for real-time systems

---

## Business Impact

The anomaly detection system can help businesses identify and address inconsistent customer feedback, improving product credibility and customer trust. Automated alerts can prompt customers to re-examine their feedback, enhancing the quality of product ratings and user experience. This way businesses can draw data-driven insights to analyse the gaps and take necessary steps to boost their sales. 

---

## How to Run

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/Anomaly_Detection_NLP.git
   cd Anomaly_Detection_NLP
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```sh
   jupyter notebook anomaly_classifier.ipynb
   ```

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- nltk
- xgboost
- matplotlib
- seaborn
- wordcloud
- beautifulsoup4

---

## Contact

For questions or collaboration, reach out via [LinkedIn](https://www.linkedin.com/in/g-kamatchi/)

---

## License

This project is licensed under the MIT License.
