import pandas as pd

X_train = pd.read_csv("Data_Table.csv")
X_train = X_train.drop(X_train.columns[0], axis=1)

X = pd.DataFrame()

#Remove the extra columns which do not contribute to the analysis

for i in range(784):
    if sum(X_train["col"+str(i)]) > 0:
        X["col"+str(i)] = X_train["col"+str(i)]


sum_list = []
for i in X.columns:
    sum_list.append(sum(X[i]))

print max(sum_list)
print min(sum_list)

print len(X.columns)
X.to_csv("Data_Table_Preprocessed.csv")

#After this, execute Neural_Network.py and other classifiers to train them