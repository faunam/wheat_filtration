import unittest
import warnings

import mallet


class TestMalletMethods(unittest.TestCase):
    """Test class for methods in mallet.py: make_topic_model"""

    @classmethod
    def setUpClass(self):
        self.gensim_dq_model = mallet.TopicModel("test_files/quixote.txt")
        self.mallet_dq_model = mallet.TopicModel(
            "test_files/quixote.txt", "test_files/mallet_outputs/dq_doc_topics.txt",
            "test_files/mallet_outputs/dq_topic_wordcounts.txt", "test_files/mallet_outputs/split_quixote.mallet",
            "test_files/split_quixote.txt")

    def test_make_topic_model_general(self):
        """Tests that mallet.make_topic_model raises an exception if the path to
        Mallet is not found, and if only some of the parameters starting with "mallet..."
        are provided arguments."""
        # raise exception if MALLET_PATH is not defined and mallet is not in path
        # with self.assertRaises(RuntimeError):
        #     mallet.TopicModel("test_files/quixote.txt")

        # raise exception if only some of the mallet arguments are provided
        with self.assertRaises(RuntimeError):
            mallet.TopicModel(
                "test_files/quixote.txt", "test_files/mallet_outputs/dq_doc_topics.txt",
                "test_files/mallet_outputs/dq_topic_wordcounts.txt")

        # warn if corpus is less than 100 docs
        with warnings.catch_warnings(record=True) as warning:
            warnings.simplefilter("always")
            # using kwargs, test warning
            raven_model = mallet.TopicModel(
                "test_files/the_raven.txt", optimize_interval=10)
            self.assertTrue(raven_model.n_docs < 100)
            assert issubclass(warning[-1].category, UserWarning)

    def test_make_topic_model_gensim(self):
        """Tests that mallet.make_topic_model creates an LdaMallet class object
        using the gensim Mallet wrapper"""
        # test that attributes look right, test kwargs
        self.assertEqual(self.gensim_dq_model.n_topics, 20)
        self.assertEqual(self.gensim_dq_model.topic_wordcounts.get_shape()[1],
                         self.gensim_dq_model.n_voc_words)

        # checking that doc topic props for each doc sum to 1
        self.assertAlmostEqual(
            self.gensim_dq_model.doc_topic_proportions.sum(), self.gensim_dq_model.n_docs)

        # check that there are same number of full_docs as processed docs
        self.assertEqual(len(self.gensim_dq_model.full_docs),
                         len(self.gensim_dq_model.docs))

    def test_make_topic_model_mallet(self):
        """Tests that mallet.make_topic_model creates an LdaMallet class object
        using Mallet output files"""
        # inputting mallet outpus
        # test that attributes look right, test that you can use kwargs
        self.assertEqual(self.mallet_dq_model.n_topics, 20)
        self.assertEqual(self.mallet_dq_model.topic_wordcounts.get_shape()[1],
                         self.mallet_dq_model.n_voc_words)

        # checking that doc topic props for each doc sum to 1
        self.assertAlmostEqual(
            self.mallet_dq_model.doc_topic_proportions.sum(), self.mallet_dq_model.n_docs)

        # check that there are same number of full_docs as processed docs
        self.assertEqual(len(self.mallet_dq_model.full_docs),
                         len(self.mallet_dq_model.docs))


if __name__ == '__main__':
    unittest.main()
