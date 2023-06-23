# machine-learning-dq

Completing Machine Learning Course on Dataquest, adding notes and finished code here.

## Notes

- **Pandas DataFrames** are table-like structures that are easy and efficient to use with ML. Designed like SQL tables and can hold mixed data types, unlike arrays.
- **Scikit-learn** or **sklearn** is a Python library with many useful ML functions. Used for pre-processing, classification, regression, clustering, model selection & dimensionality reduction.

## General Method
1. **data.shape** and **data.size** to understand the data.
2. **data_df.isnull().sum()** to check number of missing values per column.
3. **train_test_split()** to split a dataset into training and testing data. Returns four variables for train/test data on both feature and target variables.
``` X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=417) ```
4. Instantiate chosen ML model. E.g. LinearSVC:
``` model = LinearSVC(penalty='l2', loss="squared_hinge", C=10) ```
5. Train model:
``` model.fit(X_train, y_train) ```
6. Evaluate model accuracy:
``` test_accuracy = model.score(X_test, y_test) ```
7. Fine-tune model if required. Redo steps 4-6, instantiating the model with different parameters.
8. Expand the y_test matrix to examine the model's predictions.
