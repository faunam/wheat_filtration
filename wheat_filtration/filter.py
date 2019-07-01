import numpy as np

import keywords


def total_topic_proportion(relevant_topics, document_topics):
    """helper - return sum of relevant topic proportions for one document
    Arguments: document_topics (ndarray?): topic proportions for one document"""
    return sum(document_topics[i] for i in relevant_topics)


def superkeyword_presence(text, superkeyword_list):
    """helper - return 1 if text contains any superkeyword, 0 if not"""
    for word in superkeyword_list:
        if word in text:
            return True
    return False


def keyword_proportion(keyword_list, document):
    """Return an array with word list proportions for each document. Word list
    proportion: percentage of words in the given doc that are present in keyword_list"""
    num_keywords = 0.
    doc_tokens = document.split()
    for word in doc_tokens:
        if word in keyword_list:
            num_keywords += 1
    return num_keywords/len(doc_tokens)
    # there may be a more graceful way to execute this but i cant think of it right now


def filter_corp(model, total_topic_prop_threshold, keyword_prop_threshold, relevant_topics: list, keyword_list: list, superkeyword_list: list):
    # filters corpus by percentage of rel_topics, key_words, and super_key_words
    # returns subcorpus in the form of a dictionary? where keys are unique ids? and values are a string of the doc text
    subcorpus = {}
    for i, doc in enumerate(model.docs):
        if superkeyword_presence(doc, superkeyword_list) or \
                total_topic_proportion(relevant_topics, model.doc_topics[i]) > total_topic_prop_threshold or \
                keyword_proportion(keyword_list, doc) > keyword_prop_threshold:
            subcorpus[model.doc_ids[i]] = doc  # add to subcorpus
    return subcorpus


######### under this line are things it would be nice to add later #############

def proportion_lists():
    # TODO
  # makes a matrix or list of ttp, superkeyword, and keyword proportion for the docs in corpus
  # and sets the respective topic model attributes
    pass


def subset_quality(threshs, labeled_subset):  # also had args word_list_gen and scorefun
    """Calculate F1 score for the array of thresholds threshs
    (max topic prop, total topic prop, vocab prop, and number of words
    in vocabulary list) on labeled subset"""
    # TODO (faunam|6/20/19): implement
    pass


def subset_info(threshs):  # seems like a cool feature to include
    """Return set of false positives, true positives, false negatives, and true negatives, as
    well as the sizes of the false neg and false pos sets, as well as the size of set 
    predicted as relevant, about the subset created by the given set of thresholds 
    (mtp, ttp, voc prop, and voc list length, in that order).
    This function can be edited to output any kind of info about the subset, eg the filenames."""
    # TODO (faunam|6/20/19): implement
    pass
