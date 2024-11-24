import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk

import kagglehub


#read in data
df = pd.read_csv('D:\Perarivalan\TestReviews.csv')
print(df.shape)
df = df.head(500)
print(df.shape)

df.head()



countgraph = df['class'].value_counts().sort_index().plot(kind='bar',title='Count of review',figsize=(10,5))
countgraph.set_xlabel('count')
plt.show()

example = df['review'][50]
print(example)

tokens = nltk.word_tokenize(example)
tokens[:10]

tagged = nltk.pos_tag(tokens)
tagged[:10]

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()