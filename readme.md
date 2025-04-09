# 🎬 IMDB Sentiment Showdown: BERT vs. TF-IDF 🚀

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with Love](https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F%20&%20Python-red.svg)](https://python.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-IMDB%2050k-orange.svg)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

**Witness the ultimate clash of titans in Natural Language Processing!** This project pits a classic machine learning approach (TF-IDF + Logistic Regression) against a modern deep learning powerhouse (BERT) in the arena of IMDB movie review sentiment analysis. 🍿💥

Which technique reigns supreme in understanding whether a review is glowing positive ✨ or scathing negative 🍅? Dive in, run the code, and see the results for yourself!

---

## 🔥 Highlights 🔥

*   **Direct Head-to-Head Comparison:** Benchmark the performance of traditional ML vs. state-of-the-art Transformers on the same dataset.
*   **Classic ML Implementation:** Clean, efficient implementation using Scikit-learn's TF-IDF and Logistic Regression.
*   **BERT Fine-Tuning Power:** Leverage the pre-trained knowledge of BERT (`bert-base-uncased`) and fine-tune it specifically for movie review sentiment using Hugging Face Transformers and PyTorch.
*   **Comprehensive Preprocessing:** Includes text cleaning (HTML removal, punctuation stripping) and standard NLP techniques (stopwords removal).
*   **Stunning Visualizations:** Generate insightful word clouds and sentiment distribution plots using Matplotlib and Seaborn.
*   **Reproducible & Easy to Run:** Clear instructions to get you up and running in minutes.
*   **Ready-to-Use Trained Model:** The fine-tuned BERT model is saved, ready for inference or further experimentation!

---

## 🛠️ Tech Stack 🛠️

*   **Core:** Python 🐍
*   **Data Handling:** Pandas 🐼, NumPy 🔢
*   **Classic ML:** Scikit-learn 🧪 (TF-IDF, Logistic Regression, Metrics)
*   **Text Processing:** NLTK 📚, Regex
*   **Deep Learning:** PyTorch 🔥, Hugging Face Transformers 🤗 (BERT, Tokenizer, Trainer Utilities)
*   **Visualization:** Matplotlib 📊, Seaborn 📈, WordCloud ☁️
*   **Utilities:** Kaggle API, TQDM (Progress Bars)

---

## 🚀 Getting Started 🚀

Follow these steps to unleash the sentiment analysis showdown on your machine:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/IMDB-Sentiment-Showdown.git
    cd IMDB-Sentiment-Showdown
    ```

2.  **Set Up a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Kaggle API Setup:**
    *   Go to your Kaggle account, navigate to "Settings", and click "Create New API Token". This will download `kaggle.json`.
    *   **Crucially:** Place the downloaded `kaggle.json` file directly into the root of this project directory (`IMDB-Sentiment-Showdown/`).
    *   The script will automatically move it to the correct location (`~/.kaggle/`) and set the required permissions.

5.  **Download NLTK Data (Handled by Script):**
    The Python script automatically downloads the necessary NLTK 'stopwords' and 'punkt' resources on the first run if they are not found.

---

## ⚡ Running the Analysis ⚡

Execute the main Python script to perform both analyses:

```bash
python src/imdb_sentiment_analyzer.py
```

<img width="507" alt="image" src="https://github.com/user-attachments/assets/239e674f-2cc3-4bbb-b636-753fce661907" />
<img width="421" alt="image" src="https://github.com/user-attachments/assets/c00f08c0-df22-4294-a73b-708e1117aa62" />

![image](https://github.com/user-attachments/assets/5082ebd4-165e-4a98-b97d-1e7298646365)

