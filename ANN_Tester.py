import pandas as pd
from sklearn.externals import joblib
mlp = joblib.load('Neural_Network.pkl')

X_test = pd.read_csv("Test_Data_Table.csv")
y_test = X_test.label
X_test = X_test.drop(X_test.columns[0], axis=1)
X_test = X_test.drop(labels=['label'], axis=1)

X_train = pd.read_csv("Data_Table_Preprocessed.csv")
X_train = X_train.drop(X_train.columns[0], axis=1)

X_final_test = pd.DataFrame()

for i in X_train.columns:
    X_final_test[i] = X_test[i]

print mlp.score(X_final_test,y_test)