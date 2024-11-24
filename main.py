import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk

import kagglehub


#read in data
df = pd.read_csv('D:\Perarivalan\Sentiment-Analysis-Project\TestReviews.csv')
print(df.shape)
df = df.head(500)
print(df.shape)

df.head()



countgraph = df['review'].value_counts().sort_index().plot(kind='bar',title='Count of review',figsize=(10,5))
countgraph.set_xlabel('count')
plt.show()