import pandas as pd
banking_df = pd.read_csv("subscription_prediction.csv")
banking_df["y"] = banking_df["y"].apply(lambda x: 1 if x=="yes" else 0)

train_df = banking_df.sample(frac=0.85, random_state=417)
test_df = banking_df.drop(train_df.index)

X_train = train_df.drop("y", axis=1)
y_train = train_df["y"]

X_test = test_df.drop("y", axis=1)
y_test = test_df["y"]

# print(X_train["age"])
# print(X_train["age"][1])
# print(X_train["age"].sort_values().loc[0])


def knn(feature, single_test_input, k):
    X_train["distance"] = 0
    
    for i, sample in enumerate(X_train):
        X_train["distance"][i] = 1
    neighbour_indices = X_train.sort_values(by=["distance"])[0:k].index
    
    target_labels = []
    for i in neighbour_indices:
        target_labels.append(y_train[i])
    prediction = pd.Series(target_labels).value_counts().index[0]
    return prediction
        
    
single_input = X_test.sample(1, random_state=417)
k = 3
feature = "age"
prediction = knn(feature, single_input, k)
print(prediction)
print(single_input.index)
print(y_test[5146])
