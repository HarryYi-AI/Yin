# download datasets
# pip install kaggle
# chmod 600 ~/.kaggle/kaggle.json
# kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# !pip install kaggle
import os
os.makedirs("/root/.kaggle", exist_ok=True)
!mv kaggle.json /root/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
import zipfile
with zipfile.ZipFile("imdb-dataset-of-50k-movie-reviews.zip", 'r') as zip_ref:
    zip_ref.extractall("./imdb_data")
import pandas as pd

# data
data = pd.read_csv('./imdb_data/IMDB Dataset.csv')

# data.head
print(data.head())

# information
print(data.info())
print(data['sentiment'].value_counts())

import re

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # del HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # del something
    text = text.lower()  # lower
    return text

data['cleaned_review'] = data['review'].apply(clean_text)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]  # del or remove  stop words
    return ' '.join(tokens)

data['processed_review'] = data['cleaned_review'].apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)  # 5000 features
X = tfidf.fit_transform(data['processed_review']).toarray()
y = data['sentiment'].map({'positive': 1, 'negative': 0})  # transit num

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# test the model
y_pred = model.predict(X_test)

# some indexs
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(data['processed_review']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

import seaborn as sns

sns.countplot(data['sentiment'])
plt.title('Sentiment Distribution')
plt.show()

#another methods
#pip install transformers
# pip install torch torchvision torchaudio
# pip install scikit-learn
# pip install datasets


import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import get_scheduler
from tqdm import tqdm

#  IMDB 
data = pd.read_csv('./imdb_data/IMDB Dataset.csv')

# 1 and 0
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# split 
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['review'], data['sentiment'], test_size=0.2, random_state=42
)

#  BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# def encode data
def encode_data(texts, labels, tokenizer, max_length=512):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
    labels = torch.tensor(labels.values)  # PyTorch tensor 
    return encodings, labels

# encoding
train_encodings, train_labels = encode_data(train_texts, train_labels, tokenizer)
val_encodings, val_labels = encode_data(val_texts, val_labels, tokenizer)

class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# train val
train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)

#  BERT model 
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# GPU 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# loss fun
loss_fn = torch.nn.CrossEntropyLoss()

# scheduler
num_training_steps = len(train_loader) * 3  #  3  epoches
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# train
model.train()
for epoch in range(3):  
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in progress_bar:
        #  GPU
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)

        # forward
        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, labels)

        # back
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # loss
        progress_bar.set_postfix(loss=loss.item())
# eval
model.eval()

# test
all_preds, all_labels = [], []
for batch in val_loader:
    inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
    labels = batch['labels'].to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    preds = torch.argmax(outputs.logits, dim=1)
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# accuracy
accuracy = accuracy_score(all_labels, all_preds)
print("Accuracy:", accuracy)
print(classification_report(all_labels, all_preds, target_names=['Negative', 'Positive']))
# save model and tokenizer
model.save_pretrained('./bert-imdb-model')
tokenizer.save_pretrained('./bert-imdb-model')

# Hardware Requirements: Training BERT models requires high hardware configurations, preferably a GPU. 
# If a GPU is unavailable, consider running the code on Google Colab.

# Hyperparameter Tuning: Experiment with adjusting batch_size, learning_rate, and epochs to optimize the model's performance.

# Model Improvements: Consider using larger pre-trained models (e.g., bert-large-uncased) or more modern architectures 
# such as RoBERTa or DistilBERT to achieve better results.
