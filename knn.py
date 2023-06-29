import pandas as pd
banking_df = pd.read_csv("subscription_prediction.csv") # Not available on this repo
banking_df["y"] = banking_df["y"].apply(lambda x: 1 if x=="yes" else 0)

# Longer version of train_test_split()
train_df = banking_df.sample(frac=0.85, random_state=417)
test_df = banking_df.drop(train_df.index)

X_train = train_df.drop("y", axis=1)
y_train = train_df["y"]

X_test = test_df.drop("y", axis=1)
y_test = test_df["y"]

# KNN for one row
def knn(features, test_input, k):
    distance = 0
    for feature in features:
        distance += (X_train[feature] - test_input[feature])**2
    X_train["distance"] = (distance)**0.5
    
    prediction = y_train[X_train["distance"].nsmallest(n=k).index].mode()[0]
    return prediction

# Normalization
for dataset in [X_train, X_test]:
    for feature in ["age", "campaign"]:
        min_feature = dataset[feature].min()
        max_feature = dataset[feature].max()
        for index, row in dataset.iterrows():
            row[feature] = (row[feature]-min_feature)/(max_feature-min_feature)
            
features = ["age", "campaign", "marital_married", "marital_single"]
k = 3
correct = 0
for index, row in X_test.iterrows():
    X_test["predicted_y"] = knn(features, X_test.iloc[417], k)
    if X_test["predicted_y"][index] == y_test[index]:
        correct += 1

accuracy = 100*correct/X_test.shape[0]
print(accuracy)
    


