import numpy as np

import mallet


def rel_ent_key_list(topic_model: TopicModel, n_top_keywords: int, relevant_topics: List[int]):
    """Returns a list of the top n keywords according to relative entropy [better]
     Arguments:
       topic_model_matrix (ndarray): a topic by vocabulary word matrix where each entry
       is the total word count for that word in that topic
       n_top_words (int): the number of keywords the method will return        #is this how you say it?
       relevant_topics (iterable of int)
     Returns:
       keyword_list (iterable of str): list of the top n keywords, sorted in order of greatest to least (what?)
     """
    topic_word_matrix = topic_model.topic_wordcounts
    # Log of probabilities of vocab words
    vocab_logs = np.log(topic_model.topic_word_matrix.sum(
        axis=0) / topic_word_matrix.sum())

    # Log of probabilities of vocab words given they were in each relevant topic (?)
    topic_logs = np.log(topic_word_matrix[relevant_topics, :].sum(
        axis=0) / topic_word_matrix[relevant_topics, :].sum())  # keeps giving me divide by 0 warning

    # relative entropy proportions, unsorted (?)
    unsorted_props = (topic_word_matrix.sum(axis=0) /
                      topic_word_matrix.sum()) * (topic_logs - vocab_logs)

    sorted_props_and_voc = sorted([(unsorted_props[i], topic_model.vocab[i]) for i in list(
        np.argpartition(unsorted_props, topic_model.n_voc_words - n_top_keywords))[-n_top_keywords:]], reverse=True)
    ordered_vocab = []
    for (_, voc) in sorted_props_and_voc:
        ordered_vocab.append(voc)
    return ordered_vocab

# TODO (faunam|6/19/19): implement tfidf and logtf
