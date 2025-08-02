# Movie Review Sentiment Analysis Using Real Scraped Data

This project performs sentiment analysis on movie reviews collected from real-world sources through web scraping. The goal is to classify user reviews as positive or negative by leveraging both classical machine learning methods and deep learning.

---

## Project Overview

### 1. Data Collection (Web Scraping)

- Movie reviews were **extracted from online review platforms** using automated web scraping techniques.
- The scraping process involved sending HTTP requests, parsing HTML content with tools like BeautifulSoup, and extracting user comments.
- This approach ensures that the dataset reflects **authentic user opinions** and language patterns.

### 2. Data Cleaning & Labeling

- Raw scraped data often contains noise such as HTML tags, special characters, and irrelevant text.
- Text preprocessing steps were applied to clean the reviews, including:
  - Removing HTML artifacts and punctuation
  - Lowercasing all text
  - Removing stopwords and extra whitespace
- Each review was manually or heuristically **labeled with a binary sentiment label**:
  - `0` for negative reviews
  - `1` for positive reviews
- The final cleaned and labeled dataset was saved as `labeled_sentiment_reviews.csv`.

### 3. Model Training & Evaluation

- Two classical machine learning models were trained using TF-IDF vectorized features:
  - Logistic Regression
  - Multinomial Naive Bayes
- A deep learning model was also developed:
  - An LSTM (Long Short-Term Memory) network with embedding layers to capture sequential dependencies in text.
- The dataset was split into training and testing subsets for unbiased evaluation.
- Performance metrics calculated include:
  - Accuracy
  - Precision, Recall, F1-Score (detailed in classification reports)
- The LSTM model leverages sequence padding and tokenization to handle variable length reviews.

### 4. Visualization & Analysis

- Confusion matrices were generated for each model to visualize true/false positives and negatives.
- Word clouds highlight the most frequent words in positive and negative reviews, providing intuitive insight into the dataset.
- A bar plot compares overall accuracy scores of the three models.
- Training history plots show LSTM’s accuracy progression over epochs for both training and validation sets.

