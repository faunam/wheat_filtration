from typing import List
import os
import string


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


def split_corpus(source: str):
    """Splits corpus into appropriately sized documents (100-500 words).
    Arguments:
        source (str): the path to the txt file or directory (containing txt files)
            where the corpus is located. if source is a directory, it must end in /
            or \\ (whatever is appropriate to your system)!
    Returns:
        string list: a list containing strings representing documents in the corpus.
            Documents are between 250 and 500 words. Documents end on sentences as much
            as possible within the wordcount limits."""

    full_corp = ""
    if source[-4:] == ".txt":
        # handles case where source is a text file
        with open(source, "r") as inf:
            full_corp += inf.read()

    else:
        # handles case where source is directory, or throws exception if it is not
        directory = os.fsencode(source)
        for thisfile in os.listdir(directory):
            filename = os.fsdecode(thisfile)
            if len(filename) < 4 or filename[-4:] != ".txt":
                continue
                # skips non text files
            with open(source + filename, "r") as inf:
                full_corp += inf.read()

    tokens = full_corp.split()  # tokens have punc attached
    num_tokens = len(tokens)
    split_corp = []
    i = 0
    while i < (num_tokens-250):
        j = _next_punc_index(i+250, tokens)
        new_seg = " ".join(tokens[i:j])
        if j > (num_tokens - 250):  # end of string
            new_seg += " " + " ".join(tokens[j:])
        split_corp.append(new_seg)
        i = j

    return split_corp


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
    with open(new_file_name, "w")as out:
        str_ids = [str(x) for x in doc_uniq_ids]
        for i, line in enumerate(split_corpus_list):
            clean_line = line.translate(
                str.maketrans('', '', string.punctuation + "—"))  # remove punctuation
            # got this code snippet from stack overflow https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
            prepped_text = str_ids[i] + "\t" + doc_names[i] + "\t" + clean_line
            out.write(prepped_text + "\n")
