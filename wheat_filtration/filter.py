import keywords


def total_topic_proportion(document_topics, relevant_topics):
    """Return sum of relevant topic proportions for a document.
    Arguments:
        document_topics (iterable of float): topic proportions for one document.
        relevant topics (iterable of int): a list of the numbers corresponding
            with the topics considered relevant by the user."""
    assert (len(relevant_topics) <= len(document_topics)
            )  # TODO make this the right kind of error
    return sum([document_topics[i] for i in relevant_topics])


def keyword_proportion(document, keyword_list):
    """Return percentage of words in the given doc that are present in keyword_list."""
    doc_tokens = document.split()
    num_keywords = sum(
        [1 if word in keyword_list else 0 for word in doc_tokens])
    return float(num_keywords)/len(doc_tokens)


def superkeyword_presence(document, superkeywords):
    """Return 1 if document contains any superkeywords, 0 if not."""
    for word in superkeywords:
        if word in document.split():
            return True
    return False


class FilterHelper():
    """Creates a filter object containing filter criteria such as keyword list,
    superkeyword list, total topic proportion threshold, and keyword proportion
    threshold.

    Arguments:
        topic_model (TopicModel): a TopicModel object instantiated with a corpus or
            files from a Mallet topic model.
        relevant_topics (iterable of int): a list of the numbers corresponding
            with the topics considered relevant by the user. Note that the number
            corresponding with the first topic is '0', the second topic is '1', etc.
        n_keywords: number of keywords to include in keyword list. Default is 20.
        superkeywords (iterable of str): a list of keywords which signify immediate relevance
            of the document that contains them (better wording). Default is an empty list.
        keyword_list: A list of keywords ordered by [the relevance they signify]. Default is
            a keyword list generated using the relative entropy method.
        total_topic_prop_threshold (float): the threshold of relevance for the total proportion
            of relevant topics in a document. If a document surpases the threshold, it is considered relevant.
        keyword_prop_threshold (float): the threshold of relevance for the proportion of words
            on the keyword list that appear in a document. If a document surpases the threshold,
            it is considered relevant.

    Attributes:
        topic_model (TopicModel): a TopicModel object instantiated with a corpus or
            files from a Mallet topic model.
        relevant_topics (iterable of int): a list of the numbers corresponding
            with the topics considered relevant by the user.
        superkeywords (iterable of str): a list of keywords which signify immediate relevance
            of the document that contains them (better wording). Default is an empty list.
        keyword_list: A list of keywords ordered by [the relevance they signify]. Default is
            a keyword list generated using the relative entropy method.
        total_topic_prop_threshold (float): the threshold of relevance for the total proportion
            of relevant topics in a document. If a document surpases the threshold, 
            it is considered relevant. Default is 0.25.
        keyword_prop_threshold (float): the threshold of relevance for the proportion of words
            on the keyword list that appear in a document. If a document surpases the threshold,
            it is considered relevant. Default is 0.15.

    Raises:
        RuntimeError: if user enters both keyword list and n_keywords when using the
        keyword_list setter method.
        """

    def __init__(self, topic_model, relevant_topics, keyword_list=None, n_keywords=100, superkeywords=[],
                 total_topic_prop_threshold=0.25, keyword_prop_threshold=0.15):
        self._relevant_topics = relevant_topics
        if keyword_list is None:
            keyword_list = keywords.rel_ent_key_list(
                topic_model, n_keywords, relevant_topics)
        self._keyword_list = keyword_list

        lower_superkeys = [word.lower() for word in superkeywords]
        # TODO: deal with this appropriately when making lowercasing optional
        extended_superkeys = [
            word for word in topic_model.vocabulary if
            word in lower_superkeys or
            any([(chunk in lower_superkeys) for chunk in word.split('_')])
        ]
        self._superkeywords = extended_superkeys

        self._total_topic_prop_threshold = total_topic_prop_threshold
        self._keyword_prop_threshold = keyword_prop_threshold
        self._topic_model = topic_model

    @property
    def topic_model(self):
        """Get topic_model used to create filter"""
        return self._topic_model

    @property
    def relevant_topics(self):
        """Get list of relevant topics"""
        return self._relevant_topics

    @property
    def keyword_list(self):
        """Get or set keyword list. Input either a list of keywords, or input an integer n
        to generate a keyword list containing n words."""
        return self._keyword_list

    @keyword_list.setter
    def keyword_list(self, keyword_list=None, n_keywords=None):
        if keyword_list is not None:
            self._keyword_list = keyword_list
        elif n_keywords is not None:
            self._keyword_list = keywords.rel_ent_key_list(
                self.topic_model, n_keywords, self.relevant_topics)
        else:
            raise RuntimeError(
                "Enter either a keyword list or an integer for number of keywords")

    @property
    def superkeywords(self):
        return self._superkeywords

    @superkeywords.setter
    def superkeywords(self, superkeywords):
        self._superkeywords = superkeywords

    @property
    def total_topic_prop_threshold(self):
        return self._total_topic_prop_threshold

    @total_topic_prop_threshold.setter
    def total_topic_prop_threshold(self, total_topic_prop_threshold):
        self._total_topic_prop_threshold = total_topic_prop_threshold

    @property
    def keyword_prop_threshold(self):
        return self._keyword_prop_threshold

    @keyword_prop_threshold.setter
    def keyword_prop_threshold(self, keyword_prop_threshold):
        self._keyword_prop_threshold = keyword_prop_threshold


def is_relevant(doc, doc_topics, filter_helper):
    """Returns a boolean for relevance of given document. A document is considered
    relevant if: it contains any superkeywords(filter_helper.superkeywords), passes
    the total topic proportion threshold(filter_helper.total_topic_prop_threshold),
    or passes the keyword proportion threshold(filter_helper.keyword_prop_threshold).
    Arguments:
        doc (string): preprocessed document from the corpus
        doc_topics (iterable of float): proportion of each topic present in the given document
        filter_helper (FilterHelper): an object containing the necessary information
            to label the relevance of the given document
    Returns:
        (bool): Representing whether or not the given document is relevant according
        to the information in filter_helper"""

    has_superkeyword = superkeyword_presence(
        doc, filter_helper.superkeywords)
    passes_total_topic_thresh = total_topic_proportion(
        doc_topics, filter_helper.relevant_topics) > filter_helper.total_topic_prop_threshold
    passes_keyword_thresh = keyword_proportion(
        doc, filter_helper.keyword_list) > filter_helper.keyword_prop_threshold

    return has_superkeyword or passes_total_topic_thresh or passes_keyword_thresh


def filter_corpus(topic_model, filter_helper):
    """Filters corpus used to make topic_model according to criteria entered in filter_helper.
    Arguments:
        topic_model (TopicModel): a TopicModel object instantiated with a corpus or
        files from a Mallet topic model.
        filter_helper (FilterHelper): a FilterHelper object instantiated with filter
        properties.
    Returns:
        subcorpus (dict): a dictionary containing the subset of the corpus that passed
        the relevance filter. keys are the unique document ids and values are the (unprocessed)
        document text"""
    subcorpus = {}
    for i, doc_id in enumerate(topic_model.docs):
        doc = topic_model.docs[doc_id]
        doc_topics = topic_model.doc_topic_proportions[i, :]
        if is_relevant(doc, doc_topics, filter_helper):
            # add full document to subcorpus as <doc_id>: <doc_body>
            subcorpus[doc_id] = topic_model.full_docs[doc_id]
    return subcorpus

#####################################################
######### under this line are things it would be nice to add later #############
# TODO (faunam|6/20/19): implement


def proportion_lists():
    """makes a matrix or list of ttp, superkeyword, and keyword proportion for the docs in corpus
    and sets the respective topic model attributes"""
    pass


def subset_quality(threshs, labeled_subset):  # also had args word_list_gen and scorefun
    """Calculate F1 score for the array of thresholds threshs
    (max topic prop, total topic prop, vocab prop, and number of words
    in vocabulary list) on labeled subset"""
    pass


def subset_info(threshs):  # seems like a cool feature to include
    """Return set of false positives, true positives, false negatives, and true negatives, as
    well as the sizes of the false neg and false pos sets, as well as the size of set
    predicted as relevant, about the subset created by the given set of thresholds
    (mtp, ttp, voc prop, and voc list length, in that order).
    This function can be edited to output any kind of info about the subset, eg the filenames."""
    pass
