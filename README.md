# AI vs Human Text Classification

This notebook demonstrates a text classification task to distinguish between AI-generated and human-written texts. It preprocesses the dataset, visualizes the distribution of the classes, and applies machine learning models to predict the category of each text.

## 1. Importing Libraries

```python
import numpy as np
import pandas as pd
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import words
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
```

### 2. CSV Data

You can obtain the CSV data required for this project from the following link:

[Download CSV data](https://drive.google.com/file/d/1J_3_HaYdSEg_LjlGjL4_L7WybYg0GARC/view?usp=sharing)

Once you have downloaded the CSV file, make sure to place it in the appropriate directory to be used by the application.

## 3. Loading and Exploring the Dataset

The dataset is loaded from a CSV file named `AI_Human.csv`, which contains two columns: `text` (the text to be classified) and `generated` (the label indicating whether the text is AI-generated or human-written).

```python
df = pd.read_csv("./AI_Human.csv")
```

We explore the data and get basic information:

```python
df.head()  # Preview the first few rows of the dataset
df.info()  # Display dataset info
df.describe()  # Get descriptive statistics for the dataset
```

### Distribution of Classes

The dataset consists of 487,235 samples, with 305,797 human-written texts (labeled as 0) and 181,438 AI-generated texts (labeled as 1). A count plot shows the distribution of the labels.

```python
sns.countplot(data=df, x="generated", hue="generated", palette="viridis", legend=False)
plt.show()
```

## 4. Preprocessing the Text

The text is preprocessed by:

1. Removing newlines and apostrophes.
2. Removing punctuation.
3. Removing irrelevant words and ensuring all words are correctly spelled.

Functions used for preprocessing:

```python
def remove_tags(text: str) -> str:
    text = text.replace("\n", "").replace("'", "")
    return text

def remove_punctuation(text: str) -> str:
    regex = f"[{re.escape(string.punctuation)}]"
    return re.sub(regex, "", text)

def is_spelled_correctly(word: str) -> bool:
    return word in english_words
```

These functions are applied to the dataset to clean the text data before analysis.

## 5. Model Training and Evaluation

A machine learning pipeline is set up using the `CountVectorizer` and `TfidfTransformer` for text feature extraction, followed by classification models such as Multinomial Naive Bayes and Support Vector Classifier (SVC).

```python
# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['generated'], test_size=0.3, random_state=42)

# Define pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
```

### Metrics

The classification report includes metrics such as precision, recall, F1-score, and support for both classes (AI-generated and human-written texts).

## 6. Results and Discussion

The performance of the models is evaluated using classification metrics. The goal is to build a reliable classifier that can distinguish between human-written and AI-generated content.

---

### Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `seaborn`, `nltk`, `scikit-learn`, `matplotlib`

### Installation

To install the required libraries, run:

```bash
pip install numpy pandas seaborn nltk scikit-learn matplotlib
```

### Usage

1. Load your dataset in CSV format (ensure it has `text` and `generated` columns).
2. Run the notebook to preprocess the data, train the model, and evaluate the results.
