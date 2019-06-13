import os
import unittest

import munge

# sentence constants for text file generation
WHALE_SENTENCE = "the great big whale splashed the timid blue fox.\n"
ANGEL_SENTENCE = "the big white angel was on fire, and the black angel put him out and said \"hey idiot\".\n"


def generate_metadata(keys, texts):
    """Creates a dictionary mapping each corpus in keys to a dictionary of ids and names
    generated for each document in that corpus"""
    metadata = {}
    for i, corpus in enumerate(keys):
        text = munge.split_corpus(texts[i])
        ids = list(range(len(text)))
        names = [corpus + str(x) for x in ids]
        metadata[corpus] = {"text": text, "ids": ids, "names": names}
    return metadata


# a dictionary containing metadata (ids and names of documents) for munged sample text files
SAMPLE_METADATA = generate_metadata(["angel", "quixote"],
                                    ["test_files/simple_angel_50.txt", "test_files/quixote.txt"])
# not sure how to handle this.. i wanted to create simple_angel_50.txt with my setup function but then this constant would get buried...


def file_factory(n_lines, sample_sentence, sentence_id):
    """Writes a file with n lines of sample_sentence to the file location
    test_files/simple_{sentence_id}_{n_lines}.txt"""
    with open("test_files/simple_" + sentence_id + "_" + str(n_lines) + ".txt", "w") as out:
        i = 0
        while i < n_lines:
            out.write(sample_sentence)
            i += 1


class TestMungeMethods(unittest.TestCase):
    """Test class for methods in munge.py: split_corpus and write_clean_corpus"""
    @classmethod
    def setUpClass(cls):
        """Creates temporary test files for class, located in test_files/"""
        file_factory(100, WHALE_SENTENCE, "whale")
        file_factory(50, ANGEL_SENTENCE, "angel")
        file_factory(100, ANGEL_SENTENCE, "angel")
        munge.write_clean_corpus(
            SAMPLE_METADATA["angel"]["text"], SAMPLE_METADATA["angel"]["ids"], SAMPLE_METADATA["angel"]["names"], "test_files/munged_angel_50.txt")
        munge.write_clean_corpus(
            SAMPLE_METADATA["quixote"]["text"], SAMPLE_METADATA["quixote"]["ids"], SAMPLE_METADATA["quixote"]["names"], "test_files/munged_quixote.txt")

    def make_linelength_tests(self, filename):
        """Generates line length tests for documents in filename: 
        checks that each document is 250-500 words long, and each document either
        ends on punctuation or is 500 words"""
        for doc in munge.split_corpus(filename):
            # all segments are more than 100, less than 500
            self.assertTrue(len(doc.split()) >= 250)
            self.assertTrue(len(doc.split()) <= 502)
            # all segments either end on punctuation or are 500 words
            self.assertTrue(
                len(doc.split()) == 500 or doc[-1] in (".", "!", "?"))

    def test_split_corpus_simple(self):
        """Tests munge.split_corpus on simple test corpora. checks:
                -function can handle text files and directories of text files
                -each line (document) is between 250 and 500 words
                -each line either ends on punctuation or is 500 words"""
        # handles text files and directories containin txt files and other file types
        corpora = ["test_files/simple_whale_100.txt", "test_files/simple_angel_50.txt", "test_files/"]
        for corpus in corpora:
          for doc in munge.split_corpus(corpus):
              # all segments are more than 100, less than 500
              self.assertTrue(len(doc.split()) >= 250)
              self.assertTrue(len(doc.split()) <= 502)
              # all segments either end on punctuation or are 500 words
              self.assertTrue(
                  len(doc.split()) == 500 or doc[-1] in (".", "!", "?"))


        # other possible conditions to test for:
        # test for exception handling -> if file is not a text file or directory
        # all tokens are words, that may have punc connected. but no "" or white space
        # all words from original file are in cleaned file

    def test_write_clean_corpus_simple(self):
        """Tests munge.write_clean_corpus on simple test files. checks:
                -every line of out file has correct formatting: < unique_id >\t < orig_doc_id >\t < text >
                -AssertionError is raised when list of unique_ids are not unique"""
        # every line has correct formatting <unique_id>\t<orig_doc_id>\t<text>
        with open("test_files/munged_angel_50.txt", "r") as inf:
            for i, line in enumerate(inf):
                features = line.split("\t")

                self.assertEqual(features[0], str(
                    SAMPLE_METADATA["angel"]["ids"][i]))
                self.assertEqual(
                    features[1], SAMPLE_METADATA["angel"]["names"][i])
                self.assertEqual(len(features), 3)

        # ids are unique
        split_angels = munge.split_corpus("test_files/simple_angel_50.txt")
        # last id is repeated twice
        angel_ids_shallow = SAMPLE_METADATA["angel"]["ids"][:][:-1]
        angel_ids_shallow.append(angel_ids_shallow[-1])
        with self.assertRaises(AssertionError):
            munge.write_clean_corpus(split_angels, angel_ids_shallow, SAMPLE_METADATA["angel"]["names"],
                                     "test_files/angels_nonunique.txt")

    def test_munge_complex(self):
        """Tests munge.split_corpus and munge.write_clean_corpus on a real-world
        example corpus (Don Quixote). Uses same tests from test_split_corpus_simple
        and test_write_clean_corpus_simple."""  # do i need to repeat what I'm checking for? or is this sufficient?

        # split_corpus tests
        for doc in munge.split_corpus("test_files/quixote.txt"):
            self.assertTrue(len(doc.split()) <= 500)
            self.assertTrue(len(doc.split()) >= 250)
            self.assertTrue(
                len(doc.split()) == 500 or doc[-1] in (".", "!", "?"))

        # write_clean_corpus tests
        with open("test_files/munged_quixote.txt", "r") as inf:
            for i, line in enumerate(inf):
                features = line.split("\t")
                self.assertEqual(features[0], str(
                    SAMPLE_METADATA["quixote"]["ids"][i]))
                self.assertEqual(
                    features[1], SAMPLE_METADATA["quixote"]["names"][i])
                self.assertEqual(len(features), 3)

    @classmethod
    def tearDownClass(cls):
        """Deletes temporary files created for class. Runs after all tests in class"""
        test_file_list = (["test_files/simple_angel_100.txt", "test_files/munged_angel_50.txt",
                           "test_files/simple_whale_100.txt", "test_files/munged_quixote.txt"])
        for fil in test_file_list:
            os.remove(fil)


if __name__ == '__main__':
    unittest.main()
