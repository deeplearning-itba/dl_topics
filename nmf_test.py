# %%
import pandas as pd
from utils import plot_top_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, MiniBatchNMF
# %%
df = pd.read_parquet('data/df_joined_2024-04-01 00:00:00.paquet')
# %%
tf_vectorizer = CountVectorizer()
tf = tf_vectorizer.fit_transform(list(df['in__title']))
# %%
nmf = NMF(10)
# %%
doc_topics = nmf.fit_transform(tf)
# %%
doc_topics.shape
# %%
doc_topics.shape
# %%
doc_topics.sum(axis=1)
# %%
nmf.components_.shape
# %%
nmf.components_[:, 0]
# %%

# %%
plot_top_words(
    nmf,
    tf_vectorizer.get_feature_names_out(),
    10,
    'LDA Plot'
)
# %%