import os
import subprocess
import shlex
import numpy as np


def call(x): return subprocess.call(shlex.split(x))


def correct_format(filename):
    # returns true if each line in file is of the format <unique_id>\t<orig_doc_id>\t<text>
    with open(filename, "r") as inf:
        for line in inf:
            if len(line.split("\t")) != 3:
                return False
        return True


class TopicModel:
    def __init__(self, filepath, num_topics, optimize_interval):  # include opt interval?
        # creates a mallet topic model of filepath, which is a txt file
        # containing the preprocessed, formatted corpus. each line in the file
        # should have the following format: <unique_id>\t<orig_doc_id>\t<text>
        # outputs a topic_model object that has fields: topic_keys_20, doc_prop_matrix, doc_names, doc_ids
        assert correct_format(filepath)
        num_docs = 0
        doc_ids = []
        doc_names = []
        with open(filepath, "r") as inf:
            for line in inf:
                num_docs += 1
                doc_ids.append(line.split("\t")[0])
                doc_names.append(line.split("\t")[1])
        self.doc_ids = doc_ids  # array of doc unique ids
        self.doc_names = doc_names  # array of doc names
        self.num_docs = len(inf.readlines())
        self.num_topics = num_topics
        filestem = filepath[:-4]  # remove .txt
        self.filestem = filestem

        # how to call this file from mallet dir?
        call(f"bin/mallet import-file --input " + filepath + " --output " +
             filestem + ".mallet --keep-sequence --remove-stopwords")
        call(f"bin/mallet train-topics --input " + filestem + ".mallet --num-topics "
             + str(num_topics) + " --output-topic-keys " +
             filestem + "_keys_20.txt --output-state "
             + filestem + "_state.gz --output-doc-topics " + filestem +
             "_doc_comp.txt --optimize-interval " + str(optimize_interval) +
             " --inferencer-filename " + filestem + "_inferencer.mallet")

        topic_array = []
        with open(filestem + "_keys_20.txt", "r") as inf:
            for line in inf:
                topic_array.append(line.split("\t")[2])
        self.topic_keys_20 = topic_array  # string list of top 20 keywords for each topic

        m = np.zeros((self.num_docs, num_topics))
        with open(filestem + "_doc_comp.txt", "r") as inf:
            next(inf)
            for i, line in enumerate(inf):
                m[i] = line.split("\t")[2:]
        self.doc_prop_matrix = m

        self.inferencer_path = filestem + "_inferencer.mallet"
        self.state_path = filestem + "_state.gz"

    def get_n_topic_keys(self, n):
      # returns a string list of the top n keywords for each topic
        call("bin/mallet train-topics --input-state " + self.state_path +
             " --input " + self.filestem + ".mallet --no-inference true --output-topic-keys "
             + self.filestem + "_keys_" + str(n) + ".txt --num-top-words " + str(n))

# Or returns outputs in a dataframe??
