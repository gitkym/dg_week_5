import numpy as np
import pandas as pd
import pickle


## set directory to location of script 
import os
import sys
os.chdir(sys.path[0])


data = pd.read_csv('tvmarketing.csv')
X = data.iloc[:,0]
X = np.asarray(data.iloc[:,0])
X = X.reshape(-1,1)
y = data.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('w_4_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('w_4_model.pkl','rb'))
print(model.predict([[33]]))