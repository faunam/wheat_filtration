import shlex
import subprocess

import gensim.corpora as corpora
from gensim.models.wrappers import LdaMallet


MALLET_PATH = None  # "/Users/fauma/Mallet-master/bin/mallet"


def _call_command_line(string):
    """Executes string as a command line prompt."""
    return subprocess.call(shlex.split(string))


def make_topic_model(tokenized_corpus: str, n_topics: int, **kwargs):  # **kwargs?
    """Creates a mallet topic model of corpus with n_topics number of topics and
    any other attributes specified.
    Arguments:
        tokenized_corpus (str):
        n_topics (int):
        alpha (int, optional) – Alpha parameter of LDA.
        id2word (Dictionary, optional) – Mapping between tokens ids and words from corpus, 
            if not specified - will be inferred from corpus.
        workers (int, optional) – Number of threads that will be used for training.
        prefix (str, optional) – Prefix for produced temporary files.
        optimize_interval (int, optional) – Optimize hyperparameters every optimize_interval 
            iterations (sometimes leads to Java exception 0 to switch off hyperparameter optimization).
        iterations (int, optional) – Number of training iterations.
        topic_threshold (float, optional) – Threshold of the probability above which we consider a topic.
        random_seed (int, optional) – Random seed to ensure consistent results, if 0 - use system clock.
    Returns:
        topic_model (LdaMallet): a topic model class object created using the Gensim LDAMallet wrapper. See Gensim documentation
        (https://radimrehurek.com/gensim/models/basemodel.html#gensim.models.basemodel.BaseTopicModel.get_topics) 
        for available class functions. [sufficient?]
    Raises:
        RuntimeError: If Mallet is not on path and MALLET_PATH isn't set to Mallet location.
    """
    path_to_mallet = MALLET_PATH
    try:
        if path_to_mallet is None:
            # will this print something for user? is that an issue?
            path_to_mallet = "mallet"
        # tests that mallet is there, doesnt run it
        _call_command_line(path_to_mallet)
    except:
        raise RuntimeError("Unable to locate mallet command {}. Please make sure Mallet is added to your PATH variable, or add the path to "
                           "your Mallet installation to the MALLET_PATH variable at the top of mallet.py.".format(path_to_mallet))

    id_to_word = corpora.Dictionary(tokenized_corpus)
    term_document_frequency = [
        id_to_word.doc2bow(doc) for doc in tokenized_corpus]
    topic_model = LdaMallet(path_to_mallet, corpus=term_document_frequency,
                            num_topics=n_topics, id2word=id_to_word, **kwargs)

    return topic_model
