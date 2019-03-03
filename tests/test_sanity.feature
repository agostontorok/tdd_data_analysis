Feature: The model is making sense

  Background: Preparations
    Given the validation data and the trained_model

  Scenario: Better than simple average
    When I claim that the simplest is to take the Average of the outcome
    And I take the train_short data
    And I get the Average of the outcome from the train_short data
    And the RMSE of the prediction of the Average of the outcome for the validation as the reference score
    And the RMSE of the prediction with the trained_model on the validation as our RMSE
    Then we see that our RMSE is lower than the reference score

  Scenario: Better than linear_regression
    When I use the Linear Regression
    And I take the train_short data
    And I train the Linear Regression on the train_short data
    And the RMSE of the prediction with the Linear Regression on the validation as reference score
    And the RMSE of the prediction with trained_model on the validation as our RMSE
    And my target is less than 75% of the reference score
    Then we see that our RMSE is lower than the reference score

  Scenario: We are providing good results
    When the RMSE of the prediction with trained_model on the validation as our RMSE
    And my reference score is 0.13 and I expect lower value from my model
    Then I see that our RMSE is indeed lower than the reference score

  Scenario: We are not overfitting the training data
    When I take the train_short data
    And the RMSE of the prediction with trained_model on the validation as our RMSE
    And the RMSE of the prediction with trained_model on the train_short as the reference score
    And my target is max 150% of the reference score
    Then I see that our RMSE is under this reference score limit