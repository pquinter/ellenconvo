from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

text_df = pd.read_csv('../output/convo_06102019.csv')
# plot words by date
colors = {'Porfi':'#0085ad','Ellen':'#f55b70'}
fig, ax = plt.subplots()
for author, group in text_df.groupby('author'):
    ax.plot(group.date, group.words, '.', color=colors[author], ms=20, alpha=0.5)
plt.xticks(rotation=60)
plt.legend(['Ellen','Porfi'])
sns.despine()
ax.set(ylabel='number of words', xlabel='date')
plt.tight_layout()
plt.savefig('./wordsbydate.pdf', bbox_inches='tight')
for (author, date), group in text_df.groupby(['author','date']):
    ax.plot(date, group.words.sum(), '.', color=colors[author], ms=20, alpha=0.5)

fig, ax = plt.subplots()
sns.stripplot(x='date', y='words',  data=text_df, hue='author', palette=colors,
        alpha=0.5, s=10, ax=ax)
