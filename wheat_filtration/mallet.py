from collections import OrderedDict
import os
import subprocess
import warnings

import gensim.corpora as corpora
from gensim.models.wrappers import LdaMallet
import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import coo_matrix

import munge

# not sure what to do with this... vs code keeps moving it
# do i even need to keep it in? can we just assume theyve done this?
# import nltk
# nltk.download('stopwords')


MALLET_PATH = "/Users/fauma/Mallet-master/bin/mallet"  # None  #


def _call_command_line(string, **kwargs):
    """Executes string as a command line prompt. stdout and stderr are keyword args."""
    return subprocess.run(string, **kwargs)


def _make_doc_dictionary(path_to_mallet, mallet_instance_filepath):
    """Returns an Ordered Dictionary containing document unique IDs as keys and
    preprocessed document text as values, for all documents in the corpus.
    Creates and deletes a temporary file in the current directory called "temp_docs_will_be_deleted.txt". """
    command = "{} info --input {} --print-instances".format(path_to_mallet,
                                                            mallet_instance_filepath)
    outfile = open("temp_docs_will_be_deleted.txt", "w")
    _call_command_line(command.split(" "), stdout=outfile)
    outfile.close()

    docs_dictionary = OrderedDict()
    with open("temp_docs_will_be_deleted.txt", "r") as in_file:
        prev_line = ""
        current_doc_id = ""
        current_doc = []
        for line in in_file:
            if line.strip() == "":
                docs_dictionary[current_doc_id] = " ".join(current_doc)
                current_doc = []
            elif prev_line.strip() == "":
                current_doc_id = line.split()[0]
            else:
                current_doc.append(line.split()[1])
            prev_line = line
    os.remove("temp_docs_will_be_deleted.txt")
    return docs_dictionary


def _make_wordcount_and_vocab(mallet_topic_wordcount_filepath, n_topics):
    """Returns a tuple containing a CSC sparse matrix of topic wordcounts, the corpus vocabulary,
    and the size of the vocabulary. The matrix has topics as columns and words as rows,
    so each entry is the total word count for that word in that topic.
    The column index of the worcount matches the index of that word in the vocabulary array."""
    with open(mallet_topic_wordcount_filepath, "r") as in_file:
        lines = in_file.readlines()
        n_voc_words = len(lines)
        topic_wordcounts_matrix = np.zeros(
            (n_topics, n_voc_words))
        vocab = []
        for word, line in enumerate(lines):
            term_and_counts = line.split()
            # the vocab term
            vocab += term_and_counts[1:2]
            topic_wordcount_pairs = term_and_counts[2:]
            for pair in topic_wordcount_pairs:
                topic, word_count = [int(num)
                                     for num in pair.split(':')]
                topic_wordcounts_matrix[topic, word] += word_count
    # csc_matrix for storage simplicity
    return (coo_matrix(topic_wordcounts_matrix).tocsc(), vocab, n_voc_words)


def _make_doctopic_matrix(mallet_doctopic_filepath):
    """Returns a tuple containing a matrix of document topic proportions, number of documents in the corpus,
    and number of topics. The matrix has topics as columns and documents as rows,
    so each entry is the proportion of the given document that that topic comprises."""
    with open(mallet_doctopic_filepath, "r") as in_file:
        corpus_doc_topics = in_file.readlines()
        n_docs = len(corpus_doc_topics)
        # the number of topics of the first line of the file
        n_topics = len(corpus_doc_topics[0].split()[2:])
        doc_topic_matrix = np.zeros((n_docs, n_topics))
        for i, line in enumerate(corpus_doc_topics):
            topics = line.split()[2:]
            doc_topic_matrix[i] = [
                float(topic_prop) for topic_prop in topics]
    return (doc_topic_matrix, n_docs, n_topics)


def _make_mallet_model(corpus_filepath, path_to_mallet, remove_stopwords, corpus_language, num_topics, **kwargs):
    """Returns a tuple containing a gensim-created topic model (class LdaMallet),
    the preprocessed corpus documents (OrderedDict), and the corpus vocabulary
    (iter of str). The keys for the document dictionary are unique document ids of
    the format "doc<i>" where i is the number of the document in the corpus."""
    prepped_corpus = munge.corpus_to_doc_tokens(
        munge.import_corpus(corpus_filepath))
    if remove_stopwords:
        stop_words = stopwords.words(corpus_language)
        prepped_corpus = [
            [word.lower() for word in doc if word.lower() not in stop_words] for doc in prepped_corpus]
    id_to_word = corpora.Dictionary(prepped_corpus)
    term_document_frequency = [
        id_to_word.doc2bow(doc) for doc in prepped_corpus]
    topic_model = LdaMallet(path_to_mallet, corpus=term_document_frequency,
                            id2word=id_to_word, num_topics=num_topics, **kwargs)

    docs = OrderedDict(("doc" + str(i), " ".join(doc))
                       for i, doc in enumerate(prepped_corpus))

    return (topic_model, docs, [word for word in id_to_word.values()])


class TopicModel():
    """Creates an object with attributes of an LDA topic model based on corpus.
    If Mallet output files are not provided, topic model will be created with gensim wrapper,
    according to any optional keyword arguments.
    If you are inputting Mallet output files, you must provide input for all
    three arguments prefaced by "mallet", and you can ignore all
    other optional arguments. If you are not inputting Mallet files, then don't provide
    input for the three arguments prefaced by "mallet".
    Arguments:
        corpus_filepath (str): filepath to where corpus is stored (directory
            containing documents or single file).
        mallet_doctopic_filepath (str, optional): the filepath of the Mallet output
            file for document topic proportions. Default is None.
        mallet_topic_wordcount_filepath (str, optional): the filepath of the Mallet
            output file for topic word counts. Default is None.
        mallet_instance_filepath (str, optional): the filepath of the mallet instance
            file used to make the topic model (typically ends in ".mallet"). Default is None. #appropriate description?
        remove_stopwords (bool, optional): Whether to remove stopwords. Default is True.
        corpus_language (str, optional): Language of corpus, to use for stopword removal. Must be in list of
            supported languages (https://pypi.org/project/stop-words/), and in lowercase letters, e.g. "english".
            Default is "english".
        n_topics (int, optional): Number of topics. Default is 20.
        alpha (int, optional): Alpha parameter of LDA. Default is 50.
        workers (int, optional): Number of threads that will be used for training. Default is 4.
        prefix (str, optional): Prefix for produced temporary files. Defaul is None.
        optimize_interval (int, optional): Optimize hyperparameters every optimize_interval
            iterations (sometimes leads to Java exception 0 to switch off hyperparameter optimization).
            Default is 0.
        iterations (int, optional): Number of training iterations. Default is 1000.
        topic_threshold (float, optional): Threshold of the probability above which we consider a topic. Default is 0.0
        random_seed (int, optional): Random seed to ensure consistent results, if 0 - use system clock. Default is 0.

    Attributes:
        docs (OrderedDict): an ordered dictionary containing the (preprocessed) documents of the corpus
            as values and corresponding document unique IDs as keys.
        topic_wordcounts (numpy.ndarray): a CSC sparse matrix containing the counts of each
            vocabulary word in each topic. Shape: (number of topics, number of vocab words)
        doc_topic_proportions (numpy.ndarray): a matrix containing the topic
            proportions of each document. Shape: (number of topics, number of documents)
        vocabulary (iterable of str): a list containing all vocabulary words. Indeces match column
            indeces of topic_wordcounts.
        n_docs (int): Number of documents in corpus.
        n_topics (int): Number of topics used to make LDA topic model.
        n_voc_words (int): Number of vocabulary words in corpus.
        total_topic_prop (None): Settable. A vector containing the proportion of
            relevant topics in each document. Required shape: (number of documents)
        keyword_prop (None): Settable. A vector containing the proportion of
            relevant keywords in each document. Required shape: (number of documents)
        superkeyword_bin (None): Settable. A vector containing binary (1,0) for
        the presence of a superkeyword in the document. Required shape: (number of documents)

    Raises:
        RuntimeError: If Mallet is not on path and MALLET_PATH isn't set to Mallet location.
        RuntimeError: If only one of mallet_doctopic_filepath, mallet_topic_wordcount_filepath, #appropriate error type?
        and mallet_instance_filepath is passed an argument. Must pass all an argument, or none.
        UserWarning: If corpus is unusually small (less than 100 documents).
    """

    def __init__(self, corpus_filepath, mallet_doctopic_filepath=None,
                 mallet_topic_wordcount_filepath=None, mallet_instance_filepath=None,
                 remove_stopwords=True, corpus_language="english", num_topics=20, **kwargs):

        path_to_mallet = MALLET_PATH
        try:
            if path_to_mallet is None:
                path_to_mallet = "mallet"
            # tests that mallet is there, doesnt run it
            _call_command_line(path_to_mallet)
        except:
            raise RuntimeError("Unable to locate mallet command {}. Please \
            make sure Mallet is added to your PATH variable, or add the path to \
            your Mallet installation to the MALLET_PATH variable at the top \
            of mallet.py.".format(path_to_mallet))

        # topic model outputs using model created with gensim wrapper
        if mallet_doctopic_filepath is None and mallet_topic_wordcount_filepath is None \
                and mallet_instance_filepath is None:

            topic_model, self._docs, self._vocabulary = _make_mallet_model(
                corpus_filepath, path_to_mallet, remove_stopwords, corpus_language, num_topics, **kwargs)
            self._n_docs = len(self.docs)
            self._n_voc_words = len(self.vocabulary)
            self._n_topics = num_topics

            doc_topic_prop_matrix = np.zeros((self.n_docs, self.n_topics))
            for i, line in enumerate(topic_model.load_document_topics()):
                # extracting proportion from label tuple
                doc_topics = [topic_prop[1] for topic_prop in line]
                doc_topic_prop_matrix[i] = doc_topics

            self._doc_topic_proportions = doc_topic_prop_matrix
            # csc_matrix for storage simplicity
            self._topic_wordcounts = coo_matrix(
                topic_model.load_word_topics()).tocsc()
            self._total_topic_prop = None
            self._keyword_prop = None
            self._superkeyword_bin = None

        # topic model outputs using MALLET output files
        elif mallet_doctopic_filepath is not None and mallet_topic_wordcount_filepath is not None \
                and mallet_instance_filepath is not None:
            self._docs = _make_doc_dictionary(
                path_to_mallet, mallet_instance_filepath)
            self._doc_topic_proportions, self._n_docs, self._n_topics = _make_doctopic_matrix(
                mallet_doctopic_filepath)
            self._topic_wordcounts, self._vocabulary, self._n_voc_words = _make_wordcount_and_vocab(
                mallet_topic_wordcount_filepath, self.n_topics)

            self._total_topic_prop = None
            self._keyword_prop = None
            self._superkeyword_bin = None

        else:
            raise RuntimeError(
                "Missing a Mallet file input. please make sure you provide arguments\
                for all three parameters starting with \"mallet\"! If you don't have Mallet output files, don't \
                input any arguments for these parameters.")

        if self.n_docs < 100:  # an abnormally low corpus size
            warnings.warn(
                "Corpus is abnormally small (below 100 documents).")

    @property
    def docs(self):
        """Get corpus documents"""
        return self._docs

    @property
    def topic_wordcounts(self):
        """Get the topic wordcounts matrix"""
        return self._topic_wordcounts

    @property
    def doc_topic_proportions(self):
        """Get the document topic proportions matrix"""
        return self._doc_topic_proportions

    @property
    def vocabulary(self):
        """Get the corpus vocabulary"""
        return self._vocabulary

    @property
    def n_docs(self):
        """Get the number of documents in the corpus"""
        return self._n_docs

    @property
    def n_topics(self):
        """Get the number of topics"""
        return self._n_topics

    @property
    def n_voc_words(self):
        """Get the number of corpus vocabulary words"""
        return self._n_voc_words

    @property
    def total_topic_prop(self):
        """Get or set an array containing the total proportions of relevant topics in each document.
        Input should be a one dimensional numpy array that is n_docs long.
        Raises:
            AssertionError: input is the wrong shape"""
        return self._total_topic_prop

    @total_topic_prop.setter
    def total_topic_prop(self, total_topic_prop):
        assert total_topic_prop.shape == (
            self.n_docs,), "total_topic_prop is the wrong shape! It should be one dimensional and n_docs long"
        self._total_topic_prop = total_topic_prop

    @property
    def keyword_prop(self):
        """Get or set an array containing the total proportions of relevant keywords in each document.
        Input should be a one dimensional numpy array that is n_docs long.
        Raises:
            AssertionError: input is the wrong shape"""
        return self._keyword_prop

    @keyword_prop.setter
    def keyword_prop(self, keyword_prop):
        assert keyword_prop.shape == (
            self.n_docs,), "keyword_prop is the wrong shape! It should be one dimensional and n_docs long"
        self._keyword_prop = keyword_prop

    @property
    def superkeyword_bin(self):
        """Get or set an array containing binary signifiers of the presence or absence of
        super keywords in each document. Input should be a one dimensional numpy array that is n_docs long.
        Raises:
            AssertionError: input is the wrong shape"""
        return self._superkeyword_bin

    @superkeyword_bin.setter
    def superkeyword_bin(self, superkeyword_bin):
        assert superkeyword_bin.shape == (
            self.n_docs,), "superkeyword_bin is the wrong shape! It should be one dimensional and n_docs long"
        self._superkeyword_bin = superkeyword_bin
