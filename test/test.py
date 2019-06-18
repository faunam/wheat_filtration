import os
import string
import unittest

import munge
import mallet


class PreprocessingUnitTestClass(unittest.TestCase):
    """Class of unit tests for preprocessing steps. Contains setUp and tearDown class methods that create
    relevant test files."""

    def file_factory(self, n_lines, sample_sentence, sentence_id):
        """Writes a file with n lines of sample_sentence to the file location
        test_files/simple_{sentence_id}_{n_lines}.txt"""
        with open("test_files/simple_" + sentence_id + "_" + str(n_lines) + ".txt", "w") as out:
            for _ in range(n_lines):
                out.write(sample_sentence)

    def generate_metadata(self, keys, texts):
        """Creates a dictionary mapping each corpus in keys to a dictionary of ids and names
        generated for each document in that corpus"""
        metadata = {}
        for i, corpus in enumerate(keys):
            text = munge.corpus_to_documents(munge.import_corpus(texts[i]))
            ids = list(range(len(text)))
            names = [corpus + str(x) for x in ids]
            metadata[corpus] = {"text": text, "ids": ids, "names": names}
        return metadata

    @classmethod
    def setUpClass(cls):
        """Creates temporary test files for class, located in test_files/
        Creates class variable sample_metadata containing generated metadata for angel and quixote corpora."""
        # sentence constants for text file generation
        cls.sample_sentence_whale = "the great big whale splashed the timid blue fox.\n"
        cls.sample_sentence_angel = "the big white angel was on fire, and the black angel put him out and said \"hey idiot\".\n"

        cls.file_factory(cls, 100, cls.sample_sentence_whale, "whale")
        cls.file_factory(cls, 50, cls.sample_sentence_angel, "angel")
        cls.file_factory(cls, 100, cls.sample_sentence_angel, "angel")
        # a dictionary containing metadata (ids and names of documents) for munged sample text files
        cls.sample_metadata = cls.generate_metadata(cls, ["angel", "quixote"],
                                                    ["test_files/simple_angel_50.txt", "test_files/quixote.txt"])

        munge.write_clean_corpus(
            cls.sample_metadata["angel"]["text"], cls.sample_metadata["angel"]["ids"], cls.sample_metadata["angel"]["names"], "test_files/munged_angel_50.txt")
        munge.write_clean_corpus(
            cls.sample_metadata["quixote"]["text"], cls.sample_metadata["quixote"]["ids"], cls.sample_metadata["quixote"]["names"], "test_files/munged_quixote.txt")

    @classmethod
    def tearDownClass(cls):
        """Deletes temporary files created for class. Runs after all tests in class"""
        test_file_list = (["test_files/simple_angel_100.txt", "test_files/munged_angel_50.txt",
                           "test_files/simple_whale_100.txt", "test_files/munged_quixote.txt", "test_files/simple_angel_50.txt"])
        for test_file in test_file_list:
            os.remove(test_file)


class TestMungeMethods(PreprocessingUnitTestClass):
    """Test class for methods in munge.py: import_corpus, corpus_to_documents,
    corpus_to_doc_tokens, write_clean_corpus"""

    def test_corpus_to_documents_simple(self):
        """Tests munge.corpus_to_documents on simple test corpora. checks:
                -function can handle text files and directories of text files
                -each line (document) is between 250 and 500 words
                -each line either ends on punctuation or is 500 words"""

        # import_corpus handles text files and directories containin txt files and other file types
        corpora = ["test_files/simple_whale_100.txt",
                   "test_files/simple_angel_50.txt", "test_files/"]
        for filename in corpora:
            corpus = munge.import_corpus(filename)
            for doc in munge.corpus_to_documents(corpus)[:-1]:
                # all documents (except last) are between 250 and 500 words
                self.assertTrue(len(doc.split()) >= 250)
                self.assertTrue(len(doc.split()) <= 500)
                # all documents (except last) either end on punctuation or are 500 words. 5 characters back to accommodate
                # for extra characters like quotes and parentheses
                self.assertTrue(
                    len(doc.split()) == 500 or "." in doc[-5:] or "!" in doc[-5:] or "?" in doc[-5:])

        # other possible conditions to test for:
        # test for exception handling -> if file is not a text file or directory
        # all tokens are words, that may have punc connected. but no "" or white space
        # all words from original file are in cleaned file

    def test_corpus_to_doc_tokens_simple(self):
        """Tests munge._corpus_to_doc_tokens on simple test corpora. checks:
        -function can handle text files and directories of text files
        -each element(document) is an array containing between 250 and 500 strings (tokens)
        -no string contains punctuation"""
        # handles text files and directories containin txt files and other file types
        corpora = ["test_files/simple_whale_100.txt",
                   "test_files/simple_angel_50.txt", "test_files/"]
        for filename in corpora:
            corpus = munge.import_corpus(filename)
            for doc in munge.corpus_to_doc_tokens(corpus):
                # all documents (except last) are between 250 and 500 tokens
                self.assertTrue(len(doc) >= 250)
                self.assertTrue(len(doc) <= 500)
                # no punctuation in any tokens
                for token in doc:
                    token_no_punc = token.translate(
                        str.maketrans('', '', string.punctuation + "—"))
                    self.assertTrue(token == token_no_punc)

    def test_write_clean_corpus_simple(self):
        """Tests munge.write_clean_corpus on simple test files. checks:
                -every line of out file has correct formatting: < unique_id >\t < orig_doc_id >\t < text >
                -AssertionError is raised when list of unique_ids are not unique"""
        # every line has correct formatting <unique_id>\t<orig_doc_id>\t<text>
        with open("test_files/munged_angel_50.txt", "r") as in_file:
            for i, line in enumerate(in_file):
                features = line.split("\t")
                self.assertEqual(features[0], str(
                    self.sample_metadata["angel"]["ids"][i]))
                self.assertEqual(
                    features[1], self.sample_metadata["angel"]["names"][i])
                self.assertEqual(len(features), 3)

        # ids are unique
        corpus = munge.import_corpus("test_files/simple_angel_50.txt")
        split_angels = munge.corpus_to_documents(corpus)
        angel_ids_shallow = self.sample_metadata["angel"]["ids"][:][:-1]
        angel_ids_shallow.append(angel_ids_shallow[-1])
        with self.assertRaises(AssertionError):  # last id is repeated twice
            munge.write_clean_corpus(split_angels, angel_ids_shallow, self.sample_metadata["angel"]["names"],
                                     "test_files/angels_nonunique.txt")

        # doc names, doc ids, document lists are same length
        with self.assertRaises(AssertionError):  # last id is repeated twice
            munge.write_clean_corpus(split_angels, angel_ids_shallow[:-1], self.sample_metadata["angel"]["names"],
                                     "test_files/angels_nonunique.txt")

    def test_munge_complex(self):
        """Tests munge.split_corpus and munge.write_clean_corpus on a real-world
        example corpus (Don Quixote). Uses same tests from test_split_corpus_simple
        and test_write_clean_corpus_simple."""  # do i need to repeat what I'm checking for? or is this sufficient?

        # corpus_to_documents tests
        corpus = munge.import_corpus("test_files/quixote.txt")
        for doc in munge.corpus_to_documents(corpus)[:-1]:
            # all documents (except last) are between 250 and 500 words
            self.assertTrue(len(doc.split()) >= 250)
            self.assertTrue(len(doc.split()) <= 500)
            # all documents (except last) either end on punctuation or are 500 words. 5 characters back to accommodate
            # for extra characters like quotes and parentheses
            self.assertTrue(
                len(doc.split()) == 500 or "." in doc[-5:] or "!" in doc[-5:] or "?" in doc[-5:])

        # write_clean_corpus tests
        with open("test_files/munged_quixote.txt", "r") as in_file:
            for i, line in enumerate(in_file):
                features = line.split("\t")
                self.assertEqual(features[0], str(
                    self.sample_metadata["quixote"]["ids"][i]))
                self.assertEqual(
                    features[1], self.sample_metadata["quixote"]["names"][i])
                self.assertEqual(len(features), 3)


class TestMalletMethods(PreprocessingUnitTestClass):
    """Test class for methods in mallet.py: make_topic_model"""

    def test_make_topic_model(self):
        """Tests that mallet.make_topic_model creates an LDAMallet class object, and raises
        an exception if the path to mallet is not found"""
        # raise exception if MALLET_PATH is not defined and mallet is not in path
        with self.assertRaises(RuntimeError):
            corpus = munge.corpus_to_doc_tokens(
                munge.import_corpus("test_files/simple_whale_100.txt"))
            mallet.make_topic_model(corpus, 10)


if __name__ == '__main__':
    unittest.main()
