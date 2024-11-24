import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# compare


sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()


results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]

results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]

results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]

results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]

from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")

sent_pipeline('I love you!')

sent_pipeline('I hate you!')