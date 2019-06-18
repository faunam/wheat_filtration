from typing import List
import os
import string

import nltk.data


def import_corpus(source: str):
    """Load corpus from source into one string.
    Arguments:
        source (str): the path to the text file or directory (containing text files)
            where the corpus is located. if source is a directory, it must end in /
            or \\ (whatever is appropriate to your system)! This function will parse all
            files in the given directory that are not system files.
    Returns:
        full_corp (str): a string containing the corpus"""

    full_corp = ""
    try:
        with open(source, "r") as in_file:
            full_corp += in_file.read()
    except IsADirectoryError:
        directory = os.fsencode(source)
        for thisfile in os.listdir(directory):
            filename = os.fsdecode(thisfile)
            # skip system files
            # maybe add functionality that allows them to give a regex for the kinds of files they want to parse
            if filename[0] == ".":  # eg or regex not in filename
                continue
            try:
                with open(source + filename, "r") as in_file:
                    full_corp += in_file.read()
            except IsADirectoryError:
                continue
    return full_corp


def corpus_to_documents(corpus: str):
    """Splits corpus into appropriately sized documents in the form of strings (250-500 words).
    Arguments:
        corpus (str): a string containing the corpus
    Returns:
        corpus_documents (iterable of str): a list containing strings representing documents in the corpus.
        Documents are 250-500 words, except possibly for the last document in the corpus. 
        They end on the next sentence punctuation after 250 words, or at 500 words, whichever comes first.
        If the sentence following a document is the last in the corpus, it is also included in the document."""

    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_detector.tokenize(
        corpus.strip())  # tokens have punc attached

    corpus_documents = []
    document_in_progress = []
    for sentence in sentences:
        while len(document_in_progress) > 500:
            corpus_documents.append(" ".join(document_in_progress[:500]))
            document_in_progress = document_in_progress[500:]
        if len(document_in_progress) < 250:
            document_in_progress.extend(sentence.split())
        else:
            # add document to corpus once it is between 250 and 500 words
            corpus_documents.append(" ".join(document_in_progress))
            document_in_progress = []
    # adds leftover at end of corpus to last document
    try:
        corpus_documents[-1] = corpus_documents[-1] + \
            " ".join(document_in_progress)
    # allow user to import an abnormally small corpus; warn at time of topic model creation
    except IndexError:
        corpus_documents.append(document_in_progress)

    return corpus_documents


def corpus_to_doc_tokens(corpus: str):
    """Splits corpus into tokenized documents, in the form of lists of strings, of appropriate size (250-500 words).
    Arguments:
        corpus (str): a string containing the corpus
    Returns:
        corpus_documents (iterable of iterable of str): a list of tokenized documents.
        Documents are 250-500 words, except possibly for the last document in the corpus.
        They end on the next sentence punctuation after 250 words, or at 500 words, whichever comes first.
        If the sentence following a document is the last in the corpus, it is also included in the document."""

    sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sentence_detector.tokenize(
        corpus.strip())  # tokens have punc attached

    def clean_punc(token):
        # remove punctuation
        # got this code snippet from stack overflow https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
        return token.translate(
            str.maketrans('', '', string.punctuation + "—"))

    corpus_documents = []
    document_in_progress = []
    for sentence in sentences:
        while len(document_in_progress) > 500:
            corpus_documents.append(" ".join(document_in_progress[:500]))
            document_in_progress = document_in_progress[500:]
        if len(document_in_progress) < 250:
            document_in_progress.extend(clean_punc(sentence).split())
        else:
            # add document to corpus once it is between 250 and 500 words
            corpus_documents.append(document_in_progress)
            document_in_progress = []
    # adds leftover at end of corpus to last document
    try:
        corpus_documents[-1].extend(document_in_progress)
    # allow user to import an abnormally small corpus; warn at time of topic model creation
    except IndexError:
        corpus_documents.append(document_in_progress)

    return corpus_documents


def write_clean_corpus(split_corpus_list: List[str], doc_uniq_ids: list,
                       doc_names: List[str], new_file_name: str):
    """Formats corpus for topic modeling in Mallet, and removes punctuation.
    writes formatted corpus to file, where each document is formatted thusly:
    <unique_id>\t<orig_doc_id>\t<text>
    new_file_name includes the filepath, or it will be created locally
    Arguments:
        split_corpus_list (iterable of str): list of all documents in corpus, 250-500
            words each.
        doc_uniq_ids (list): list of document unique IDs. Must be the same length
            as split_corpus_list. IDs must be unique.
        doc_names (iterable of str): list of document names. Must be the same length
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
