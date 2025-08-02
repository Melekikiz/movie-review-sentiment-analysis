import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

df=pd.read_csv("imdb_shawshank_reviews.csv")

df.drop_duplicates(subset="review", inplace=True)
df=df[df["review"].str.len()>15]
print("Temizlenmiş veri şekli:", df.shape)
# Temiz veriyi kaydet
df_cleaned = df.dropna()
df_cleaned.to_csv("cleaned_sentiment_reviews.csv", index=False)
print("✅ Temizlenmiş veriler 'cleaned_sentiment_reviews.csv' dosyasına kaydedildi.")


df = pd.read_csv("cleaned_sentiment_reviews.csv")
stop_words=set(stopwords.words("english"))

def clean_text(text):
    text=text.lower()
    text=re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)  # Mention ve hashtag temizle
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Noktalama ve rakamları sil
    text = re.sub(r'\s+', ' ', text).strip()  # Fazla boşlukları temizle
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]  # Stopword temizle
    return " ".join(filtered)

df["clean_review"]=df["review"].apply(clean_text)

print("Temizlenmiş ilk 5 yorum:")
print(df[["review", "clean_review"]].head())


#Etiketleme

positive_words=["great", "excellent", "amazing", "wonderful", "fantastic", "brilliant","best"]
negative_words=["bad", "terrible", "boring", "worst", "waste", "awful", "hate", "poor"]

def label_review(text):
    if any(word in text for word in positive_words):
        return 1
    elif any(word in text for word in negative_words):
        return 0
    else:
        return None

df["label"]=df["clean_review"].apply(label_review)
df_labeled=df.dropna(subset=["label"])

print(f"Toplam yorum: {len(df)}")
print(f"Etiketlenebilen Yorum sayısı: {len(df_labeled)}")
print(df_labeled["label"].value_counts())

df_labeled.to_csv("labeled_sentiment_reviews.csv", index=False)
print("✅ Etiketlenmiş veriler 'labeled_sentiment_reviews.csv' dosyasına kaydedildi.")