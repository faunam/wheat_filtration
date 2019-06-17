from typing import List
import os
import string

import nltk.data


def _next_punc_index(i, tokens: List[str]):
    """Returns the index of the element in an list of tokens following the next
    punctuation after index i. If no punctuation is found within 250 indeces after i,
    returns the index i + 250."""
    curr_index = i
    while curr_index < (i + 250):
        possible_punc = tokens[curr_index][-1]
        if possible_punc in (".", "!", "?"):
            return curr_index + 1
        curr_index += 1
    return curr_index


def import_corpus(source: str):
    """Load corpus from source into one string.
    Arguments:
        source (str): the path to the txt file or directory (containing txt files)
            where the corpus is located. if source is a directory, it must end in /
            or \\ (whatever is appropriate to your system)!
    Returns:
        full_corp (str): a string containing the corpus
        """
    full_corp = ""
    try:
        with open(source, "r") as in_file:
            full_corp += in_file.read()
    except IsADirectoryError:
        directory = os.fsencode(source)
        for thisfile in os.listdir(directory):
            filename = os.fsdecode(thisfile)
            # skip system files
            if filename[0] = ".":
                continue
            try:
                with open(source + filename, "r") as in_file:
                    full_corp += in_file.read()
            except IsADirectoryError:
                continue
    return full_corp


def corpus_to_documents(corpus: str):
    """Splits corpus into appropriately sized documents (250+ words).
    Arguments:
        corpus (str): a string containing the corpus
    Returns:
        corpus_documents (string list): a list containing strings representing documents in the corpus.
        Documents are at least 250 words long and end on the next sentence after 250 words are reached, [better phrasing...]
        unless the following is the last in the corpus, in which case it is also included in the document."""

    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_detector.tokenize(
        corpus.strip())  # tokens have punc attached

    corpus_documents = []
    document_in_progress = []
    for sentence in sentences:
        if len(document_in_progress) < 250:
            document_in_progress.extend(sentence.split())
        else:
            # add document to corpus once it is over 250 words
            corpus_documents.append(" ".join(document_in_progress))
            document_in_progress = []
    # adds leftover at end of corpus to last document
    corpus_documents[-1] = corpus_documents[-1] + \
        " ".join(document_in_progress)

    return corpus_documents


# better name... maybe this function isnt useful to user, just helpful for gensim
def corpus_to_doc_tokens(corpus: str):
    """Splits corpus into tokenized documents of appropriate size (250+ words).
    Arguments:
        corpus (str): a string containing the corpus
    Returns:
        corpus_documents (iterable of iterable of str): an array of tokenized documents. 
        Documents are at least 250 words long and end on the next sentence after 250 words are reached, [better phrasing...]
        unless the following is the last in the corpus, in which case it is also included in the document. """

    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_detector.tokenize(
        corpus.strip())  # tokens have punc attached

    def clean_punc(token):
        # remove punctuation
        return token.translate(
            str.maketrans('', '', string.punctuation + "—"))

    corpus_documents = []
    document_in_progress = []
    for sentence in sentences:
        if len(document_in_progress) < 250:
            document_in_progress.extend(clean_punc(sentence).split())
        else:
            # add document to corpus once it is over 250 words
            corpus_documents.append(document_in_progress)
            document_in_progress = []
    # adds leftover at end of corpus to last document
    corpus_documents[-1].extend(document_in_progress)

    return corpus_documents


def write_clean_corpus(split_corpus_list: List[str], doc_uniq_ids: list,
                       doc_names: List[str], new_file_name: str):
    """Formats split corpus for topic modeling in Mallet, and removes punctuation.
    writes formatted corpus to file, where each document is formatted thusly:
    <unique_id>\t<orig_doc_id>\t<text>
    new_file_name includes the filepath or will be created locally
    Arguments:
        split_corpus_list (str list): list of all documents in corpus, 250-500
            words each (use split_corpus).
        doc_uniq_ids (list): list of document unique IDs. Must be the same length
            as split_corpus_list. IDs must be unique.
        doc_names (str list): list of document names. Must be the same length
            as split_corpus_list.
        new_file_name (str): path to the file where the prepped corpus will be stored.
    Raises:
        AssertionError: IDs in doc_uniq_ids are not unique
        AssertionError: name, id, and document lists are not the same length
    Returns: none"""
    assert len(doc_uniq_ids) == len(set(doc_uniq_ids)
                                    ), "IDs in doc_uniq_ids are not unique."
    assert len(doc_names) == len(doc_uniq_ids) == len(
        split_corpus_list), "name, id, and document lists are not the same length."
    str_ids = [str(x) for x in doc_uniq_ids]
    with open(new_file_name, "w") as out:
        for i, line in enumerate(split_corpus_list):
            clean_line = line.translate(
                str.maketrans('', '', string.punctuation + "—"))  # remove punctuation
            # got this code snippet from stack overflow https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
            prepped_text = str_ids[i] + "\t" + doc_names[i] + "\t" + clean_line
            out.write(prepped_text + "\n")


import_corpus("test/")
