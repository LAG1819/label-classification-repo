from tmtoolkit import topicmod
import numpy as np
from gensim.models import CoherenceModel
from itertools import combinations

def get_coherence_lda_gensim(lda_model,bow_corpus,docs):
    """Calculate coherence value of generated 
    Args:
        lda_model (_type_): Generated gensim LDA model 
        bow_corpus (_type_): Text corpus of type bow. Specific type of gensim.
        docs (_type_): Cleaned texts.

    Returns:
        float: Returns the calculated coherence value based on given lda_model.
    """
    # instantiate topic coherence model
    cm = CoherenceModel(model=lda_model, corpus=bow_corpus, texts=docs, coherence='c_v')

    # get topic coherence score
    coherence_lda_value = cm.get_coherence() 
    return coherence_lda_value

def get_coherence_topics_gensim(topics,common_corpus,common_dictionary):
    cm = CoherenceModel(topics=topics, corpus=common_corpus, dictionary=common_dictionary, coherence='u_mass')
    coherence_topic_value = cm.get_coherence()

def get_coherence(lda_components,tf_matrix ,texts,tfiIdf_vectorizer_vocab):
    """Calculation of coherence value based on a given lda model.

    Args:
        lda_components (_type_): _description_
        tf_matrix (_type_): _description_
        texts (_type_): _description_
        tfiIdf_vectorizer_vocab (_type_): _description_
    """
    topicmod.evaluate.metric_coherence_gensim(measure='c_v', 
                        top_n=25, 
                        topic_word_distrib=lda_components,
                        dtm=tf_matrix,#dtm_tf, 
                        vocab=tfiIdf_vectorizer_vocab,
                        texts=texts, #train['cleaned_NOUN'].values)
                        return_mean= True
                        )
def get_coherence_manual(model, term_rankings):
    """Manual calculation of coherence value based on given lda model and term rankings.

    Args:
        model (_type_): Fitted LDA Model.
        term_rankings (_type_): Term Frequencies of fitted lda model.

    Returns:
        float: Returns the calculated coherence value based on given lda_model.
    """
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