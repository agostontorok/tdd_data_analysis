import numpy as np
import pandas as pd
from tdda.constraints.base import DatasetConstraints
from tdda.constraints.pd.constraints import verify_df


def generate_boolcolumn_from_zero(X, to_convert_cols):
    """
    creates a new column for the output of Value == 0
    :param X: data
    :param to_convert_cols: columns
    :return: data with new columns
    """
    for col in X:
        if col in to_convert_cols:
            X['has' + col] = (X[col] != 0).astype(int)
    return X


def vectorize_categorical_columns(X, constraints):
    """
    vectorize columns by first converting them to categorical and
    then to one hot encoding. Then it removes the original categorical
    columns for the output.

    :param X: data
    :param constraints: json file with TDDA constraints
    :return: data with one hot columns
    """
    cons = DatasetConstraints(loadpath=constraints)
    n_cat_cols = 0
    n_cats = 0
    initial_shape = X.shape[1]

    for key, value in cons.to_dict()['fields'].items():
        if value['type'] == 'string':
            if len(value['allowed_values']) < 20:
                # for checking
                n_cat_cols += 1
                n_cats += len(value['allowed_values'])

                X[key] = pd.Categorical(X[key], categories=value['allowed_values'])
                X = X.join(pd.get_dummies(X[key], prefix=key))
                X = X.drop(key, axis=1)

    expected_len = initial_shape + n_cats - n_cat_cols
    actual_len = X.shape[1]
    if actual_len != expected_len:
        raise ValueError('Expected shape mismatch after vectorizing: {} != {}'.format(
            expected_len, actual_len))

    return X

class HousePriceData(object):
    """
    the house price data object, this warrants that I do the same preprocessing on
    all datasets that I want to use.
    """
    def __init__(self, filename, constraints):
        self.X_cols = ['MSSubClass', 'LotArea', 'Street', 'LotShape', 'LandContour',
                       'LotConfig', 'LandSlope', 'Condition1', 'Condition2',
                       'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt',
                       'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond',
                       'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 
                       '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'FullBath',
                       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
                       'Fireplaces', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
                       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
                       'MoSold', 'YrSold', 'SaleCondition',
                      # new additions
                       'Alley', 'BsmtCond', 'BsmtExposure', 'BsmtFinSF1', 'BsmtFinSF2',
                       'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath',
                       'BsmtQual', 'BsmtUnfSF', 'Electrical', 'Exterior1st', 'Exterior2nd',
                       'Fence', 'FireplaceQu', 'Functional', 'GarageArea', 'GarageCars',
                       'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'GarageYrBlt',
                       'KitchenQual', 'LotFrontage', 'MSZoning', 'MasVnrArea',
                       'MasVnrType', 'MiscFeature', 'PoolQC', 'SaleType',
                       'TotalBsmtSF', 'Utilities']
        self.y_col = 'SalePrice'
        self.zero_to_bool_columns = ['2ndFlrSF', 'LowQualFinSF', 'HalfBath', 
                                     'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 
                                     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
                                     'PoolArea', 'MiscVal']
        data = pd.read_csv(filename, index_col=['Id'])
        data = self.fillna_with_zero_for_numeric(data)
        self._validate(data, constraints)
        self.X = data[self.X_cols]
        if self.y_col in data.columns:
            self.y = np.log(data[self.y_col])
        else:
            self.y = None
         
        self.X = generate_boolcolumn_from_zero(self.X, to_convert_cols=self.zero_to_bool_columns)
        self.X = vectorize_categorical_columns(self.X, constraints)
        

    @classmethod
    def _validate(self, data, constraints):
        """
        1. Check if everything is available and is conform to our expectations
        """
        result = verify_df(data, constraints, type_checking='strict')
        if result.failures != 0:
            raise KeyError("One or more columns were not fitting the validation constraints: failures: {}".format(result.failures))
        else:
            pass
        
    def fillna_with_zero_for_numeric(self, data):
        """
        for numeric columns it fill with zero the missing parts.
        :param data:
        :return:
        """
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        cols_to_fill = data.select_dtypes(include=numerics).columns
        fill_dict = {col : 0 for col in cols_to_fill}
        data = data.fillna(fill_dict)
        data[cols_to_fill] = data[cols_to_fill].astype(int)
        return data
