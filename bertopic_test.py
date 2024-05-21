# %%
from bertopic import BERTopic
import pandas as pd
# %%
topic_model = BERTopic(
    # calculate_probabilities=True
)
# %%
df = pd.read_parquet('data/df_joined_2024-04-01 00:00:00.paquet')
data = list(df['in__title'])
topics, probs = topic_model.fit_transform(data)
# %%
from collections import Counter
Counter(topics)
# %%
topic_model.topic_embeddings_.shape
# %%
total_topics = len(topic_model.get_topics())
# %%
probs.shape
# %%
for idx in range(total_topics):
    max_idx = topic_model.probabilities_[idx].argmax()
    prob = topic_model.probabilities_[idx][max_idx]
    print(max_idx, prob, topics[idx], probs[idx][max_idx])
    
# %%        
# %%
topics = topic_model.get_topics()
print(topics)
print(len(topics))
# %%
hierarchical_topics = topic_model.hierarchical_topics(data)
hierarchical_topics
# %%
topic_model.get_topic(1)
# %%
topic_model.get_representative_docs(1)
# %%
topic_model.embedding_model.embed('Hola que tal').shape
# %%
topic_model.c_tf_idf_
# %%
topic_model.language
# %%
# topic_model.merge_models()
# %%
topic_model.topic_embeddings_.shape
# %%
topic_model.get_topic_freq(2)
# %%
topic_model.get_topic_info(0)
# %%
topic_model.topic_sizes_
# %%
topic_model.get_topics()
# %%
new_docs_topics, new_docs_probs = topic_model.transform(
    [
        'Novedades en la guerra de Ucrania y Rusia',
        'Echaron a participante de Gran Hermano'
    ]
)
# %%
sorted_probs = new_docs_probs.argsort(axis=1)
sorted_probs
# %%
idx = 1
new_docs_probs[idx][sorted_probs[idx]][::-1]
# %%
topic_model.get_topic(34)

# %% Interesante
topic_model.partial_fit()
# %%
len(topic_model.vectorizer_model.get_feature_names_out())
# %%
topic_model.visualize_topics()
# %%
topic_model.visualize_hierarchy()
# %%
topic_model.visualize_heatmap()
# %%
topic_model.visualize_distribution(probs[0], min_probability=0.001)
# %%
# topic_model.visualize_documents(list(df['in__title']))
# %%
topic_model.visualize_term_rank([0, 1, 2, 3])
# %%

# %% Calculate probabilities
df_top_probs = pd.DataFrame([{'prob': p, 'topic': t} for t, p in zip(topics, probs)])
# %%
topic_0_indexes = df_top_probs[
    (df_top_probs['prob'] >= 0.99) &
    (df_top_probs['topic'] == 0)
].index
# %%
topic_0_docs = list(df.iloc[topic_0_indexes]['in__title'])
topic_0_docs
# %%
topic_0_embeddings = topic_model.embedding_model.embed(topic_0_docs)
# %%
topic_0_embeddings.shape
# %%
from sklearn.metrics.pairwise import cosine_similarity
# %%
cosine_similarity(topic_0_embeddings[:10])
# %%
cosine_similarity(topic_0_embeddings[:10], topic_model.topic_embeddings_)
# %%
