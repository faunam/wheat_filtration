import numpy as np


# is topic word matrix an appropriately descriptive name for it?
def make_topic_word_matrix(n_topics, n_words, topic_model=None, counts_filepath=None):
    """Returns a matrix of word counts for each topic for the entire vocabulary.
    define exactly one of the keyword arguments
    Arguments:
      topic_model (LdaMallet): a gensim LdaMallet object
      filepath (string): filepath to Mallet word-topic-counts output file
    Raises:
      TypeError: if both arguments are set (only pick one!)
    Returns:
      (ndarray(n_topics, n_docs)): each entry is the total word count for that word in that topic
    """
    # if input is topic_model
    if topic_model is not None and counts_filepath is None:
        topic_word_matrix = topic_model.load_word_topics()
    # if input is filepath to Mallet output word-topic-counts
    if topic_model is None and counts_filepath is not None:
        # is it appropriate to ask for ndocs and ntopics as a fun param or should i be inferring them from the file?
        topic_word_matrix = np.zeros(n_topics, n_words)
        with open(counts_filepath, "r") as counts_file:
            for i, line in enumerate(counts_file):  # one line per word
                for pair in line.split()[2:]:  # topic-wordcount pairs
                    # i dont understand this; is it not a list of 2 things? so how is it assigned to 2 vars?
                    topic, count = [int(num) for num in pair.split(':')]
                    topic_word_matrix[topic, i] += count
    else:
        raise TypeError(
            "pass argument for either topic_model or counts_filepath")

    return topic_word_matrix


def rel_ent_key_list(topic_word_matrix: ndarray, n_top_keywords: int, n_words: int, relevant_topics: List[int], vocab: List[str]):
    """Returns a list of the top n keywords according to relative entropy [better]
     Arguments:
       topic_word_matrix (ndarray): a topic by vocabulary word matrix where each entry
       is the total word count for that word in that topic
       n_top_words (int): the number of keywords the method will return        #is this how you say it?
       n_words (int): size of corpus vocabulary
       relevant_topics (iterable of int)
       vocab (iterable of str)
     Returns:
       keyword_list (iterable of str): list of the top n keywords, sorted in order of greatest to least (what?)
     """
    # Log of probabilities of vocab words
    vocab_logs = np.log(topic_word_matrix.sum(
        axis=0) / topic_word_matrix.sum())

    # Log of probabilities of vocab words given they were in each relevant topic (?)
    topic_logs = np.log(topic_word_matrix[relevant_topics, :].sum(
        axis=0) / topic_word_matrix[relevant_topics, :].sum())  # keeps giving me divide by 0 warning

    # relative entropy proportions, unsorted (?)
    unsorted_props = (topic_word_matrix.sum(axis=0) /
                      topic_word_matrix.sum()) * (topic_logs - vocab_logs)

    sorted_props_and_voc = sorted([(unsorted_props[i], vocab[i]) for i in list(
        np.argpartition(unsorted_props, n_words - m))[-m:]], reverse=True)
    ordered_vocab = []
    for (_, voc) in sorted_props_and_voc:
        ordered_vocab.append(voc)
    return ordered_vocab

# TODO (faunam|6/19/19): implement tfidf and logtf
