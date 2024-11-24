# Sentiment Analysis Using Python

## Introduction
This project performs sentiment analysis on a dataset of reviews using various methods, including Vader from NLTK and a pretrained model from Hugging Face's `transformers` library.

## Setup

Before you begin, make sure you have all the necessary libraries installed:

pip install pandas numpy matplotlib seaborn nltk kagglehub transformers torch

## Data

The data used for this project is a CSV file named TestReviews.csv. The dataset consists of text reviews and their corresponding sentiment labels.

**Steps**

**1. Data Loading**  

First, we load the data and display its shape:

import pandas as pd

df = pd.read_csv('D:/Perarivalan/TestReviews.csv')
print(df.shape)
df = df.head(500)
print(df.shape)
df.head()

**2. Data Visualization**

We visualize the distribution of the sentiment classes:

import matplotlib.pyplot as plt
import seaborn as sns

countgraph = df['class'].value_counts().sort_index().plot(kind='bar', title='Count of review', figsize=(10,5))
countgraph.set_xlabel('Count')
plt.show()

**3. Text Tokenization and Named Entity Recognition**

We tokenize the text and perform POS tagging and named entity recognition:

import nltk

example = df['review'][50]
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

**4. Sentiment Scoring with Vader**

We use the Vader sentiment analyzer to score the reviews:

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['review']
    res = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')

**5. Visualization of Sentiment Scores**
   
We visualize the sentiment scores:

ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound Score by Amazon Star Review')
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

**6. Using a Pretrained Model for Sentiment Analysis**

We use the cardiffnlp/twitter-roberta-base-sentiment model from Hugging Face for sentiment analysis:

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

**7. Comparison of Sentiment Models**
   
We compare the results from Vader and Roberta:

sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()

**8. Example Outputs**

Example outputs from the sentiment analysis:

results_df.query('Score == 1').sort_values('roberta_pos', ascending=False)['Text'].values[0]
results_df.query('Score == 1').sort_values('vader_pos', ascending=False)['Text'].values[0]
results_df.query('Score == 5').sort_values('roberta_neg', ascending=False)['Text'].values[0]
results_df.query('Score == 5').sort_values('vader_neg', ascending=False)['Text'].values[0]

**Conclusion**

This project demonstrates how to use different sentiment analysis techniques and models to analyze and visualize sentiment in text data.

**Authors**
Your Name

**Acknowledgments**
Thanks to the developers of the libraries and models used in this project!
