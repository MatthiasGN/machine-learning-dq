# machine-learning-dq

Completing Machine Learning Course on Dataquest, adding notes and finished code here.

## Notes

- **Pandas DataFrames** are table-like structures that are easy and efficient to use with ML. Designed like SQL tables and can hold mixed data types, unlike arrays.
- **Scikit-learn** or **sklearn** is a Python library with many useful ML functions. Used for pre-processing, classification, regression, clustering, model selection & dimensionality reduction.

---

## General Method
1. Load the data with **pandas.read_csv()**.
2. Understand the data with **data.shape** and **data.size**.
3. **data_df.isnull().sum()** to check number of missing values per column, **(data_df['target'] == 0).sum()** to count the target column.
4. **train_test_split()** to split a dataset into training and testing data. Returns four variables for train/test data on both feature and target variables.
``` X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=417) ```
5. Instantiate chosen ML model. E.g. LinearSVC:
``` model = LinearSVC(penalty='l2', loss="squared_hinge", C=10) ```
6. Train model:
``` model.fit(X_train, y_train) ```
7. Evaluate model accuracy:
``` test_accuracy = model.score(X_test, y_test) ```
8. Fine-tune model if required. Redo steps 4-6, instantiating the model with different parameters.
9. Expand the y_test matrix to examine the model's predictions.

---

## K-Nearest Neighbours Algorithm (KNN)
Non-parametric supervised learning method used for classification and regression. Given a new data point X:
1. Choose K number of neighbours.
2. Calculate distance between X and all other data points.
3. Sort the data points by smallest to largest distance to X.
4. Pick the first K entries of the sorted data points.
5. Retrieve their classes/targets.
6. If classification, assign the majority class (**mode**) to X.
7. If regression, assign the **mean** target value to X.

A generally good choice for number of neighbours is `k = n^(1/2)`. Aim to set it to an odd number, keep fine-tuning with different K to increase accuracy.

### Advantages
- Simple and easy to implement
- One variable input makes fine-tuning easy
- Don't need to build an entire model. Just an algorithm

### Disadvantages
- Algorithm gets significantly slower as the size of the dataset increases
