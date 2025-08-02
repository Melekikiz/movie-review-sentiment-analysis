import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from wordcloud import WordCloud

# --- Load Data ---
df = pd.read_csv("labeled_sentiment_reviews.csv")
X = df["clean_review"].values
y = df["label"].values

# === Machine Learning Models with TF-IDF ===

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42)

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

print("\n🔹 Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

print("\n🔹 Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# === Deep Learning (LSTM) Text Preparation ===

# Raw text train-test split (LSTM needs raw text)
X_train_raw, X_test_raw, y_train_lstm, y_test_lstm = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Tokenizer
vocab_size = 5000
max_len = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_raw)

# Convert text to sequences and pad
X_train_seq = tokenizer.texts_to_sequences(X_train_raw)
X_test_seq = tokenizer.texts_to_sequences(X_test_raw)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# === LSTM Model Definition ===
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    X_train_pad, y_train_lstm,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# Predictions and evaluation
y_pred_prob = model.predict(X_test_pad)
y_pred_lstm = (y_pred_prob > 0.5).astype(int)

print("\n🔹 LSTM")
print("Accuracy:", accuracy_score(y_test_lstm, y_pred_lstm))
print(classification_report(y_test_lstm, y_pred_lstm))

# === Visualization: Combined 2x3 Grid Figure ===

# Prepare WordCloud texts
neg_text = " ".join(df[df["label"] == 0]['clean_review'])
pos_text = " ".join(df[df["label"] == 1]['clean_review'])

# Prepare model accuracy scores
models = ['Logistic Regression', 'Naive Bayes', 'LSTM']
scores = [
    accuracy_score(y_test, y_pred_lr),
    accuracy_score(y_test, y_pred_nb),
    accuracy_score(y_test_lstm, y_pred_lstm)
]

# Confusion matrix plotting function
def plot_confusion_matrix(ax, y_true, y_pred, title, cmap):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

# Create subplots (2 rows x 3 cols)
fig, axs = plt.subplots(2, 3, figsize=(18, 12))

# Plot confusion matrices
plot_confusion_matrix(axs[0, 0], y_test, y_pred_lr, "Logistic Regression Confusion Matrix", cmap="Greens")
plot_confusion_matrix(axs[0, 1], y_test, y_pred_nb, "Naive Bayes Confusion Matrix", cmap="Oranges")
plot_confusion_matrix(axs[0, 2], y_test_lstm, y_pred_lstm, "LSTM Confusion Matrix", cmap="Blues")

# Plot WordClouds
wc_neg = WordCloud(width=400, height=300, background_color='white').generate(neg_text)
axs[1, 0].imshow(wc_neg)
axs[1, 0].set_title("Common Words in Negative Reviews")
axs[1, 0].axis('off')

wc_pos = WordCloud(width=400, height=300, background_color='white').generate(pos_text)
axs[1, 1].imshow(wc_pos)
axs[1, 1].set_title("Common Words in Positive Reviews")
axs[1, 1].axis('off')

# Plot model accuracy barplot + LSTM training accuracy on twin axis
sns.barplot(x=models, y=scores, ax=axs[1, 2])
axs[1, 2].set_ylim(0, 1)
axs[1, 2].set_ylabel("Accuracy")
axs[1, 2].set_title("Model Accuracy Comparison")

ax2 = axs[1, 2].twinx()
ax2.plot(history.history['accuracy'], label='Train Accuracy', color='red', marker='o')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue', marker='x')
ax2.set_ylim(0, 1)
ax2.set_ylabel("LSTM Accuracy")
ax2.legend(loc='lower right')

plt.tight_layout()
plt.show()
