import csv
import numpy as np
from collections import OrderedDict
import unittest

import mallet
import filter


class TestFilterMalletIntegration(unittest.TestCase):
    """Test class for methods in mallet.py and filter.py"""

    @classmethod
    def setUpClass(self):
        # self.gensim_dq_model = mallet.TopicModel("test_files/quixote.txt")
        self.mallet_dq_model = mallet.TopicModel(
            "test_files/quixote.txt", "test_files/mallet_outputs/dq_doc_topics.txt",
            "test_files/mallet_outputs/dq_topic_wordcounts.txt", "test_files/mallet_outputs/split_quixote.mallet",
            "test_files/split_quixote.txt")
        self.mallet_dq_filter = filter.FilterHelper(
            self.mallet_dq_model, [0, 1], superkeywords=["Dulcinea"])  # choose topics relating to fair maidens

        self.recession_model = mallet.TopicModel(
            "/Users/fauma/python/topmod/s_b_t_no_punc.txt", "/Users/fauma/python/topmod/phrases-50.doctopics",
            "/Users/fauma/python/topmod/phrases-50.counts.txt", "/Users/fauma/python/topmod/sent_based_chunks.mallet",
            "/Users/fauma/python/topmod/sent_based_trigram_mallet.txt")
        self.recession_filter = filter.FilterHelper(
            self.recession_model, [0, 14, 36, 44, 9, 28, 30, 31, 33], superkeywords=[
                "mortgage", "mortgages", "foreclosures", "subprime", "home_prices", "subprime_mortgage", "subprime_mortgages", "subprime_loans"])
        # hmmm idk what cutoffs they used...
        self.recession_subcorp = filter.filter_corpus(
            self.recession_model, self.recession_filter)

        self.notebook_keyword_props = np.load(
            "test_files/keyword_props_out.npy")
        self.notebook_total_topic_props = np.load(
            "test_files/total_topic_props_out.npy")
        self.notebook_docs_containing_superkeys = 12149

        with open('/Users/fauma/python/topmod/Coded_random_subset.csv', encoding='latin1') as subset:
            csvr = csv.DictReader(subset)
            relevance_labels = {row['doc_ids']: int(
                row['relevant']) for row in csvr}
        # this step puts the labels in the same order as the model labels

        labeled_docs = OrderedDict()
        to_remove = []
        for doc_id in relevance_labels:
            if doc_id in self.recession_model.docs:
                labeled_docs[doc_id] = self.recession_model.docs[doc_id]
            else:
                to_remove.append(doc_id)
        for doc_id in to_remove:
            relevance_labels.pop(doc_id)

        self.labeled_recession_docs = labeled_docs
        self.recession_labels = relevance_labels

    # tests for mallet files

    def test_general(self):
        # some generic tests for mallet and filter methods
        model = self.mallet_dq_model
        filter_helper = self.mallet_dq_filter
        for i, doc in enumerate(model.docs.values()):
            # check that total topic prop and keyword prop are proportions (between 0 and 1)
            # i think i sliced this right but , check
            assert filter.total_topic_proportion(
                model.doc_topic_proportions[i, :], filter_helper.relevant_topics) < 1
            assert filter.keyword_proportion(
                doc, filter_helper.keyword_list) < 1

        assert len(filter.filter_corpus(
            model, filter_helper)) < len(model.docs)  # checks that subcorp is smaller than corp. but by how much?

    def test_keyword_list_gen(self):
        # tests keyword list generation
        with open("test_files/keyword_out.txt", "r") as in_file:
            notebook_keywords = []
            for line in in_file:
                notebook_keywords.append(line.strip())
        self.assertEqual(len(self.recession_filter.keyword_list),
                         len(notebook_keywords))
        keylist_difference = set(notebook_keywords).symmetric_difference(
            set(self.recession_filter.keyword_list))
        # shows that lists contain roughly the same words
        # 82 word difference (41 diff words in each list)
        self.assertLessEqual(len(keylist_difference), 20)

    def test_keyword_proportion(self):
        # tests keywordlist proportion calculation
        # compare keyword proportions for each doc? given that keyword list is the same. compare list first
        for i, doc in enumerate(self.recession_model.docs.values()):
            doc_keyword_proportion = filter.keyword_proportion(
                doc, self.recession_filter.keyword_list)
            self.assertLessEqual(
                abs((self.notebook_keyword_props[i] - doc_keyword_proportion)), 0.15)
            # makes sense that this is failing given that half the lists are diff words

        # proportion
        self.assertEqual(filter.keyword_proportion(
            "mortgage credit loans market banks", self.recession_filter.keyword_list), 1)
        self.assertEqual(filter.keyword_proportion(
            "mortgage credit loans market booop", self.recession_filter.keyword_list), 0.8)

    def test_doc_topics(self):
        # tests doc topic proportion calculation and notebook comparison
        # calculation
        self.assertEqual(filter.total_topic_proportion(
            [0.25, 0.25, 0, 0, 0, 0.1, 0.4], [1, 2, 0]), 0.5)
        self.assertEqual(filter.total_topic_proportion(
            [0.25, 0.25, 0, 0, 0, 0.1, 0.4], [0, 2, 5]), 0.35)

        # notebook comparison
        doc_word_sums = np.load("test_files/doc_word_sums_out.npy")
        doc_props_nonzero = self.recession_model.doc_topic_proportions[doc_word_sums.nonzero(
        ), :]  # doc topics for docs with a nonzero wordcount
        # this might not be accurate because there may be discrepancies between the package and the notebook
        doc_props_nonzero = np.squeeze(doc_props_nonzero)
        for i, doc_topic_prop in enumerate(doc_props_nonzero):
            doc_ttp = filter.total_topic_proportion(
                doc_topic_prop, self.recession_filter.relevant_topics)
            self.assertLessEqual(
                abs((self.notebook_total_topic_props[i] - doc_ttp)), 0.1)  # i dont know what an acceptable error margin is. 0.1 seems too high.

    def test_superkeyword(self):
        # test superkey method and notebook comparison

        # superkey_presence
        self.assertEqual(filter.superkeyword_presence(
            "hello beautiful mortgage people", self.recession_filter.superkeywords), True)
        # fragment of superkey seed
        self.assertEqual(filter.superkeyword_presence(
            "hello subprime people", self.recession_filter.superkeywords), True)
        # no superkeys
        self.assertEqual(filter.superkeyword_presence(
            "hello subprim people", self.recession_filter.superkeywords), False)

        # notebook comparison
        docs_containing_superkeys = sum([1 for doc in self.recession_model.docs.values() if filter.superkeyword_presence(
            doc, self.recession_filter.superkeywords)])
        self.assertLessEqual(
            abs(self.notebook_docs_containing_superkeys -
                docs_containing_superkeys), self.recession_model.n_docs * .05)

    def test_recession_performance(self):
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
        for doc_id in self.labeled_recession_docs:
            # keep in mind that the subcorp was made from all docs, not just labeled ones
            if doc_id in self.recession_subcorp:
                if self.recession_labels[doc_id] == 1:
                    true_pos += 1
                else:
                    false_pos += 1
            else:
                if self.recession_labels[doc_id] == 0:
                    true_neg += 1
                else:
                    false_neg += 1

        print("true pos {}".format(true_pos))
        print("false pos {}".format(false_pos))
        print("true neg {}".format(true_neg))
        print("false neg {}".format(false_neg))

        # # magic numbers from the recession notebook
        # self.assertEqual(true_pos, 78)
        # self.assertEqual(false_pos, 8)
        # self.assertEqual(true_neg, 350)
        # self.assertEqual(false_neg, 63)

        total_relevant = sum(self.recession_labels.values())
        if true_pos + false_pos == 0:
            precision = 0  # guard against 0/0
        else:
            precision = true_pos/(true_pos + false_pos)
        recall = true_pos/total_relevant

        f1 = 2*((precision*recall)/(precision+recall))
        print(f1)  # getting .6 ... the best i was getting before was .8 ..


class TestFilterGensimIntegration(unittest.TestCase):
    """Test class for methods in mallet.py and filter.py"""

    # total topic proportion
    # keyword proportion
    # superkeyword presence

    # filter helper

    # filter_corpus

    @classmethod
    def setUpClass(self):
        self.dq_model = mallet.TopicModel(
            "test_files/quixote.txt", remove_stopwords=True)
        self.dq_filter = filter.FilterHelper(
            self.dq_model, [0, 1], superkeywords=["Dulcinea", "Toboso", "fair", "lady", "maiden"])  # choose topics relating to fair maidens
        # self.recession_model = mallet.TopicModel("test_files/full_recession_corpus.txt")
        # self.recession_filter = filter.FilterHelper(
        #    self.recession_model, # need to choose relevant topics

    def test_general(self):
        # some generic tests for mallet and filter methods
        model = self.dq_model
        filter_helper = self.dq_filter
        for i, doc in enumerate(model.docs.values()):
            # check that total topic prop and keyword prop are proportions (between 0 and 1)
            # i think i sliced this right but , check
            assert filter.total_topic_proportion(
                model.doc_topic_proportions[i, :], filter_helper.relevant_topics) < 1
            assert filter.keyword_proportion(
                doc, filter_helper.keyword_list) < 1

        assert len(filter.filter_corpus(
            model, filter_helper)) < len(model.docs)  # checks that subcorp is smaller than corp. but by how much?

    def test_keyword_proportion(self):
        # tests keywordlist proportion calculation
        print(self.dq_filter.keyword_list)
        # removed stopwords but top 100 keywords are still junk words...

    def test_doc_topics(self):
        # tests doc topic proportion calculation
        self.assertEqual(filter.total_topic_proportion(
            [0.25, 0.25, 0, 0, 0, 0.1, 0.4], [1, 2, 0]), 0.5)
        self.assertEqual(filter.total_topic_proportion(
            [0.25, 0.25, 0, 0, 0, 0.1, 0.4], [0, 2, 5]), 0.35)

    def test_superkeyword(self):
        # test superkey method

        print(self.dq_filter.superkeywords)
        # superkey_presence
        self.assertEqual(filter.superkeyword_presence(
            "the beautiful dulcinea of toboso", self.dq_filter.superkeywords), True)
        # fragment of superkey seed
        self.assertEqual(filter.superkeyword_presence(
            "hello, my lady", self.dq_filter.superkeywords), True)
        # no superkeys
        self.assertEqual(filter.superkeyword_presence(
            "hello dulcine", self.dq_filter.superkeywords), False)


if __name__ == '__main__':
    unittest.main()
