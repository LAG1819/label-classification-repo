from tmtoolkit import topicmod
import numpy as np
from gensim.models import CoherenceModel
from itertools import combinations

def get_coherence_lda_gensim(lda_model,bow_corpus,docs):
    # instantiate topic coherence model
    cm = CoherenceModel(model=lda_model, corpus=bow_corpus, texts=docs, coherence='c_v')

    # get topic coherence score
    coherence_lda_value = cm.get_coherence() 
    return coherence_lda_value

def get_coherence_topics_gensim(topics,common_corpus,common_dictionary):
    cm = CoherenceModel(topics=topics, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
    coherence_topic_value = cm.get_coherence()

def get_coherence(lda_components,tf_matrix ,texts,tfiIdf_vectorizer_vocab):
    topicmod.evaluate.metric_coherence_gensim(measure='c_v', 
                        top_n=25, 
                        topic_word_distrib=lda_components,
                        dtm=tf_matrix,#dtm_tf, 
                        vocab=tfiIdf_vectorizer_vocab,
                        texts=texts, #train['cleaned_NOUN'].values)
                        return_mean= True
                        )
def get_coherence_manual(model, term_rankings):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations( term_rankings[topic_index], 2 ):
            pair_scores.append( model.similarity(pair[0], pair[1]) )
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)