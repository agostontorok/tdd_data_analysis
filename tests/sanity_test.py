# coding=utf-8
"""The model is making sense feature tests."""

import numpy as np
from joblib import load
from pytest_bdd import (
    given,
    scenarios,
    then,
    when,
    parsers
)
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from ..data_analysis import HousePriceData

scenarios('features/sanity_test.feature')


def get_reference_model(model_type):
    if model_type == "Linear Regression":
        return LinearRegression()
    elif model_type == "Average of the outcome":
        return DummyRegressor(strategy="mean")
    else:
        raise NotImplementedError("only Linear Regression and Average Prediction are implemented")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


@given(parsers.parse('the {dataset} data and the {model_filename}'))
def the_data(dataset, model_filename):
    """the data."""
    data = HousePriceData(dataset + '.csv', 'house_prices_constraints_mod.tdda')
    model = load(model_filename + '.joblib')
    predictions = model.predict(data.X)
    return {dataset: data,
            model_filename: model,
            model_filename + '_predictions': predictions}


@when(parsers.parse('I use the {model_type}'))
@when(parsers.parse('I claim that the simplest is to take the {model_type}'))
def the_other_model(the_data, model_type):
    """the other model."""
    the_data[model_type] = get_reference_model(model_type)


@when(parsers.parse('I train the {model} on the {dataset} data'))
@when(parsers.parse('I get the {model} from the {dataset} data'))
def train_the_model(the_data, model, dataset):
    """I train a model in the test on a specified dataset"""
    the_data[model] = the_data[model].fit(the_data[dataset].X, the_data[dataset].y)


@when(parsers.parse('the rmse of the prediction of the {model} for the {dataset} as the {score}'))
@when(parsers.parse('the rmse of the prediction with {model} on the {dataset} as the {score}'))
@when(parsers.parse('the rmse of the prediction with the {model} on the {dataset} as {score}'))
@when(parsers.parse('the rmse of the prediction with {model} on the {dataset} as {score}'))
def predict_with_model(the_data, model, dataset, score):
    """the rmse of my prediction with a model"""
    the_data[model + '_predictions'] = the_data[model].predict(the_data[dataset].X)
    the_data[score] = rmse(the_data[dataset].y,
                           the_data[model + '_predictions'])


@when(parsers.parse('my {score} is {limit:f} and I expect lower value from my model'))
def add_reference_score(the_data, score, limit):
    """I specify simple my target score here"""
    the_data[score] = limit


@when(parsers.parse('I take the {dataset} data'))
def add_other_data(the_data, dataset):
    the_data[dataset] = HousePriceData(dataset + '.csv', 'house_prices_constraints_mod.tdda')


@when(parsers.parse('my target is less than {percent:d}% of the {reference}'))
@when(parsers.parse('my target is max {percent:d}% of the {reference}'))
def modify_reference(the_data, percent, reference):
    """modify the reference rmse with a percentage vlue"""
    percent /= 100
    the_data[reference] = the_data[reference] * percent


@then(parsers.parse('I see that {one} is indeed lower than the {other}'))
@then(parsers.parse('we see that {one} is lower than the {other}'))
@then(parsers.parse('I see that {one} is under this {other} limit'))
def one_is_smaller_than_other(the_data, one, other):
    """The first mentioned value is lower then the second mentioned"""
    assert the_data[one] < the_data[other]
