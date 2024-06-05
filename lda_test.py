# %%
import pandas as pd
from utils import plot_top_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# %%
df = pd.read_parquet('data/df_joined_2024-04-01 00:00:00.paquet')
# %%
tf_vectorizer = CountVectorizer(
    
)
tf = tf_vectorizer.fit_transform(list(df['in__title']))
# %%
n_topics=10
lda = LatentDirichletAllocation(n_topics)
# %%
doc_probs = lda.fit_transform(tf)
# %%
doc_probs.shape
# %%
doc_probs.sum(axis=1)
# %%
lda.components_.shape
# %%
lda.components_[:, 0]
# %%
plot_top_words(
    lda,
    tf_vectorizer.get_feature_names(),
    10,
    'LDA Plot'
)
# %%
