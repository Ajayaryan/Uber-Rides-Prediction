import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('taxi.csv')
# print(data.head())

data_x = data.iloc[:,0:-1].values    #dependent Variable   all the row and starting se last-1 tak
data_y = data.iloc[:,-1].values      #independent Variable  all the row and last one
print(data_y)                         #data ko read kar lega

X_train,X_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)     #training 70% and testing 30 % me split kar lete hai

reg = LinearRegression()
reg.fit(X_train,y_train)

print("Train Score:", reg.score(X_train,y_train))  #training of score is 94%
print("Test Score:", reg.score(X_test,y_test))     #testing of score is 91%

pickle.dump(reg, open('taxi.pkl','wb'))    #create model dump karta hai files me wb means write in binary

model = pickle.load(open('taxi.pkl','rb'))      # for testing read binary
print(model.predict([[80, 1770000, 6000, 85]]))   #predict kar dega diye input value per Priceperweek,Population,Monthlyincome,Averageparkingpermonth
