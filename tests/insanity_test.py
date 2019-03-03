import unittest

from pandas.api.types import is_numeric_dtype

from ..data_analysis import HousePriceData


class DataIntegrity(unittest.TestCase):

    def setUp(self):
        self.constraints_filename = 'house_prices_constraints_mod.tdda'
        self.train = HousePriceData('train.csv', self.constraints_filename)
        self.test = HousePriceData('test.csv', self.constraints_filename)
        pass

    def test_no_nan_values(self):
        """
        Tests whether there are nan values in the features and response variable
        :return:
        """
        self.assertFalse(self.train.X.isnull().any(axis=None))
        self.assertFalse(self.train.y.isnull().any(axis=None))
        self.assertFalse(self.test.X.isnull().any(axis=None))

    def test_whether_only_numeric_values(self):
        """
        we want to work only with numerical representation of the data
        :return:
        """
        self.assertTrue(self.train.X.apply(is_numeric_dtype).all())
        self.assertTrue(is_numeric_dtype(self.train.y))
        self.assertTrue(self.test.X.apply(is_numeric_dtype).all())


if __name__ == '__main__':
    unittest.main(
        verbosity=2)  # for running it in jupyter https://medium.com/@vladbezden/using-python-unittest-in-ipython-or-jupyter-732448724e31
