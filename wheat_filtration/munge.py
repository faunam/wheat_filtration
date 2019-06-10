from typing import List
import os
import subprocess
import shlex


def call(x): return subprocess.call(shlex.split(x))


def split_corpus(source: str):
        # splits corpus stored in source into appropriately sized documents (100-500 words).
        # source can be path to text file or directory containing multiple text files.
        # if source is directory, must end in / or \ (whatever is appropriate to your system)!
        # if source does not end in .txt, it is handled as a directory
        # returns a string list.
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

    tokens = full_corp.split()
    num_tokens = len(tokens)
    # tokens have punc attached
    split_corp = []
    i = 0
    while i < (num_tokens-250):
        j = next_punc_index(i+250, tokens)
        new_seg = " ".join(tokens[i:j])
        if j > (num_tokens - 250):  # end of string #not triggering for some reason
            new_seg += " ".join(tokens[j:])
        split_corp.append(new_seg)
        i = j
    # split_corp.append(" ".join(tokens[i:]))  # the remaining tokens

    return split_corp


def next_punc_index(i, tokens):
    # returns the index of the element following the next punctuation after
    # index i in an array of tokens. if no punctuation within 250 indeces after i,
    # returns the index i + 250 (to ensure max 500 segment length)
    curr_index = i
    while curr_index < (i + 250):
        possible_punc = tokens[curr_index][-1]
        if possible_punc == "." or possible_punc == "!" or possible_punc == "?":
            return curr_index + 1
        else:
            curr_index += 1
    return curr_index


def write_clean_corpus(split_corpus: List[str], doc_uniq_ids: list,
                       doc_names: List[str], new_file_name: str):
    # formats split_corpus for mallet
    # writes formatted corpus to file, where each document is formatted thusly:
    # <unique_id>\t<orig_doc_id>\t<text>
    # raises InvalidIDs if doc_uniq_ids has any repeats
    pass
