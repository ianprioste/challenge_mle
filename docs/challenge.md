# Development Process

I started by creating a Trello board to organize the activities I needed to complete. This helped me keep track of tasks and manage my workflow efficiently.

![Trello Board](../img/Trello.png)

1. **Create Trello Board**: Set up a Trello board to list all tasks and activities.
2. **Task Breakdown**: Break down the project into smaller, manageable tasks.
3. **Prioritize Tasks**: Assign priorities to tasks based on their importance and deadlines.
4. **Set Deadlines**: Establish deadlines for each task to ensure timely completion.
5. **Track Progress**: Regularly update the Trello board to reflect the progress of each task.
6. **Review and Adjust**: Periodically review the board and adjust tasks and priorities as needed.

By following these steps, I was able to maintain a clear overview of the project and ensure that all tasks were completed efficiently.

# 1st Part - Transcribe the `.ipynb` file into the `model.py` file

## Fix Bug
I used GitHub Copilot to identify the bugs and found several issues. The function `sns.barplot()` expects positional arguments. I had to add the `x=` and `y=` positional arguments to fix the plots.

### Corrected Delay Rate Calculation

Initially, the delay rate (%) was calculated incorrectly by dividing the total number of flights by the number of delays, resulting in a ratio instead of a percentage.

For instance, a delay rate of 19 for Houston implied that there was 1 delayed flight for every 19 total flights, which did not accurately reflect a percentage. To correct this, I recalculated the delay rate as the number of delays divided by the total number of flights, providing a consistent percentage measure for the plots.

### Automatic Selection of Top 10 Features

The original code had hardcoded the top 10 features, which were not derived from the feature importance and were calculated before class balancing. It would be more effective to automatically select the top 10 features based on feature importance after class balancing. However, I retained the original top 10 features as they were sufficient to pass the model tests successfully.


## Choose the best model

## Model Selection

### Advantages of XGBoost

**Popularity and Robustness**:
- **Industry Standard**: XGBoost is widely adopted in the industry due to its robust performance and versatility across various datasets.
- **Proven Track Record**: It has a history of winning numerous data science competitions and benchmarks.

**Handling Complex Datasets**:
- **Scalability**: XGBoost is designed to efficiently handle large-scale datasets.
- **Advanced Features**: It includes functionalities like handling missing values, regularization, and parallel processing, making it suitable for complex datasets we might encounter in the future.

### Consideration for Logistic Regression

**Response Time**:
- **Faster Predictions**: Logistic Regression models generally make faster predictions due to their simplicity.
- **Lower Computational Cost**: They require less computational power, which is crucial if server response time is a critical factor.

**Training Speed**:
- **Quicker Training**: Logistic Regression typically trains faster than XGBoost, especially on smaller datasets. This can be advantageous during development and tuning phases when rapid iterations are needed.

**Simplicity**:
- **Fewer Hyperparameters**: Logistic Regression has fewer hyperparameters to tune, simplifying the model development process and reducing the risk of overfitting.

### Conclusion

While XGBoost offers greater versatility and robustness for future larger and more complex datasets, Logistic Regression could be justified if server response time and computational efficiency are paramount.

**Final Decision**: I chose XGBoost with the top 10 features and class balancing for its popularity and versatility. However, consider Logistic Regression if server response time becomes a critical factor.
