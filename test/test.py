import munge
import unittest


def file_to_stringlist(filename):
    pass


def file_factory(n):
    # makes a file with "the great big whale splashed the timid blue fox." repeated n times, no linebreaks
    pass


class TestMungeMethods(unittest.TestCase):
    # def test_add(self):
    #     self.assertEqual(munge.add_one(6), 7)

    def test_split_corpus_from_file(self):
        # want to check: joined split file is same as original
        # split chunks are 100-500 words
        # dont split mid sentence?
        pass

    def test_split_corpus_from_directory(self):
        # want to check: joined split file is same as original
        # split chunks are 100-500 words
        # dont split mid sentence?
        # dont miss any files in directory, dont pick up non text files
        pass

    def test_write_clean_corpus(self):
        # check:every line has correct formatting
        # ids are unique (i could raise an error if not)
        pass


if __name__ == '__main__':
    unittest.main()
