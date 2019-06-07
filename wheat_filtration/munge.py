from typing import List


def split_corpus_from_file(txt_file: str):
        # splits corpus stored in txt_file into appropriately sized documents (100-500 words)
        # returns a string list
    pass


def split_corpus_from_directory(directory: str):
    # splits corpus stored in directory into appropriately sized documents (100-500 words)
    # returns a list of strings where each string is a document
    pass


def write_clean_corpus(split_corpus: List[str], doc_uniq_ids: list,
                       doc_names: List[str], new_file_name: str):
    # formats split_corpus for mallet
    # writes formatted corpus to file, where each document is formatted thusly:
    # <unique_id>\t<orig_doc_id>\t<text>
    # raises InvalidIDs if doc_uniq_ids has any repeats
    pass


def add_one(n):
    return n + 1
