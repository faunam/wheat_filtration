import munge
import unittest


def file_factory(n):
    # makes a file with "the great big whale splashed the timid blue fox.\n" repeated n times
    with open("test_files/simple_whale_" + str(n) + ".txt", "w") as out:
        for i in range(n):
            out.write("the great big whale splashed the timid blue fox.\n")
    # ... how to use this??


def file_fact_2(n):
  # makes a file with "the big white angel was on fire, and the black angel put
  #  him out and said \"hey idiot\".\n" repeated n times
    with open("test_files/simple_angel_" + str(n) + ".txt", "w") as out:
        for i in range(n):
            out.write(
                "the big white angel was on fire, and the black angel put him out and said \"hey idiot\".\n")


class TestMungeMethods(unittest.TestCase):

    def test_split_corpus(self):
            # want to check: joined split file is same as original (? but i remove white space.. hmm)
            # (all tokens are words, that may have punc connected. but no "" or white space)

            # handles text files
        for x in munge.split_corpus("test_files/simple_whale_100.txt"):
          # all segments are more than 100, less than 500
            self.assertTrue(len(x.split()) > 250)
            self.assertTrue(len(x.split()) <= 502)
          # all segments end on punctuation or are 500 words
            self.assertTrue(
                len(x.split()) == 500 or x[-1] == "." or x[-1] == "!" or x[-1] == "?")

        for x in munge.split_corpus("test_files/quixote.txt"):
          # all segments are more than 100, less than 500
            self.assertTrue(len(x.split()) <= 500)
            self.assertTrue(len(x.split()) > 250)
          # all segments end on punctuation or are 500 words
            self.assertTrue(
                len(x.split()) == 500 or x[-1] == "." or x[-1] == "!" or x[-1] == "?")

        # handles directories with text files and other kinds of files
        for x in munge.split_corpus("test_files/"):
          # all segments are more than 100, less than 500
            self.assertTrue(len(x.split()) <= 500)
            self.assertTrue(len(x.split()) > 250)
          # all segments end on punctuation or are 500 words
            self.assertTrue(
                len(x.split()) == 500 or x[-1] == "." or x[-1] == "!" or x[-1] == "?")

        # throws exn if source is not a txt file or directory ?? need to test exceptions?
        # self.assertRaises(munge.split_corpus("dfd"),
        #                   FileNotFoundError)
        # self.assertRaises(munge.split_corpus("test_files/whale.txt"),
        #                   FileNotFoundError)
        # self.assertRaises(munge.split_corpus("test_files/fake.cry"),
        #                   NotADirectoryError)

    def test_write_clean_corpus(self):
        # check:every line has correct formatting
        # ids are unique (i could raise an error if not)
        pass


file_factory(100)
file_factory(101)
file_fact_2(50)

if __name__ == '__main__':
    unittest.main()
