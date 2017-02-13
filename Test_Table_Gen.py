import numpy as np
import pandas as pd

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

for i in range(10000):
    for j in range(28):
        for k in range(28):
            if X_test[i][j][k]<=140:
                X_test[i][j][k] = 0
            else:
                X_test[i][j][k] = 1

X_test = X_test.reshape([10000,784])

col_list=[]
for i in range(784):
    col_list.append("col" + str(i))  
    
X_test = pd.DataFrame(data=X_test, columns=col_list)
X_test['label'] = y_test
X_test.to_csv("Test_Data_Table.csv")