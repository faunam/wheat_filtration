import numpy as np


def rel_ent_key_list(topic_model, n_top_keywords, relevant_topics):
    """Returns a list of the top n keywords based on relative entropy score
     Arguments:
       topic_model (TopicModel): a topic by vocabulary word matrix where each entry
       is the total word count for that word in that topic
       n_top_words (int): the number of keywords the method will return
       relevant_topics (iterable of int)
     Returns:
       keyword_list (iterable of str): list of the top n keywords, sorted
     """
    topic_word_matrix = topic_model.topic_wordcounts.tocsr()
    # Log of probabilities of vocab words

    vocab_logs = np.log(topic_word_matrix.sum(
        axis=0) / topic_word_matrix.sum())

    # Log of probabilities of vocab words given they were in each relevant topic
    topic_logs = np.log(topic_word_matrix[relevant_topics, :].sum(
        axis=0) / topic_word_matrix[relevant_topics, :].sum())

    # relative entropy proportions, unsorted
    unsorted_props = np.asarray(topic_word_matrix.sum(axis=0) /
                                topic_word_matrix.sum()) * np.asarray(topic_logs - vocab_logs)

    unsorted_props = np.matrix.flatten(unsorted_props)

    sorted_props_and_voc = sorted([(unsorted_props[i], topic_model.vocabulary[i]) for i in list(
        np.argpartition(unsorted_props, topic_model.n_voc_words - n_top_keywords))[-n_top_keywords:]], reverse=True)
    ordered_vocab = []
    for (_, voc) in sorted_props_and_voc:
        ordered_vocab.append(voc)
    return ordered_vocab

# TODO (faunam|6/19/19): implement tfidf and logtf
