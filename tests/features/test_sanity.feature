Feature: The model is making sense

  @wip
  Scenario: Better than simple average
    Given the data
    And the trained model
    When we predict using the model
    Then we measure smaller error than using only simple arithmetic average of the outcome
    
  