
# NLP Assignment 2 - Detailed Project Report

## Project Overview

This project involves sentiment analysis on a large dataset of tweets. The main goal is to classify the sentiment of tweets as either positive, negative, or neutral using natural language processing (NLP) techniques and machine learning models.

---

## Dataset Description

- The dataset used is the **Sentiment140** dataset.
- It contains 1.6 million tweets, originally labeled using emoticons.
- After loading, the columns are renamed for clarity:
  - `sentiment_classes`: Target variable (0 = negative, 4 = positive)
  - `ids`: Tweet ID
  - `date`: Timestamp
  - `flag`: Query status
  - `username`: Twitter username
  - `text`: The tweet content

```python
df= pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1')
df.columns = ['sentiment_classes', 'ids', 'date', 'flag', 'username', 'text']
```

---

## Data Preprocessing

To prepare the tweets for analysis, the following preprocessing steps were applied:

- Lowercasing all text
- Removing punctuation and special characters
- Removing stopwords
- Tokenizing the text
- Optional: Lemmatization or stemming

These steps help clean the noisy Twitter data and standardize it for modeling.

---

## Feature Extraction

The processed tweets are transformed into numerical features using two popular techniques:

- **Bag of Words (BoW)**: Counts the frequency of each word in the document.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weighs the frequency of terms based on how common they are across all documents.

These features are essential for training machine learning models.

---

## Sentiment Classification

The models used for classifying tweet sentiment include:

- **Naive Bayes**: A probabilistic classifier that assumes feature independence.
- **Logistic Regression**: A linear model suitable for binary classification.

The performance of these models is compared using metrics like accuracy.

---

## Evaluation Metrics

- Accuracy score
- Confusion matrix
- Precision, Recall, and F1 Score

These metrics help evaluate how well each model performs in predicting tweet sentiment.

---

## Key Insights

- The dataset is highly imbalanced: most tweets are either clearly positive or negative.
- Logistic Regression tends to outperform Naive Bayes when TF-IDF is used.
- Proper preprocessing and feature selection play a critical role in improving classification performance.

---

## Tools and Libraries Used

- Python
- Pandas
- Scikit-learn
- NLTK / spaCy
- Matplotlib / Seaborn

---

## Conclusion

This project demonstrates the end-to-end pipeline of a text classification task in NLP:

- Cleaning and preprocessing raw text
- Converting text into machine-readable formats using BoW and TF-IDF
- Applying supervised machine learning models
- Evaluating results and identifying areas for improvement

With further tuning and deep learning models (like BERT), the performance could be enhanced even more.
