from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import os.path
from sklearn.linear_model import LinearRegression


data = pd.read_csv('LogicLINKNew1.csv')
#data = pd.read_csv('LogicalLink-2.csv')
X = data[['UT', 'CML', 'RP', 'LP','SP']]
y = data['TT']

#X = df2[['TT', 'CML', 'RP', 'LP','SP']]
#y = df2['UT']


# X = df2[['TT', 'UT', 'RP', 'LP','SP']]
# y = df2['CML']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

robust = SGDRegressor(loss='huber',
                          penalty='l2',
                          alpha=0.0001,
                          fit_intercept=False,
                          shuffle=True,
                          verbose=0,
                          epsilon=0.1,
                          random_state=42,
                          learning_rate='invscaling',
                          eta0=0.01,
                          power_t=0.5)
#robust = LinearRegression()

#sc = StandardScaler()
#X_train_ = sc.fit_transform(X_train)
#X_test_ = sc.transform(X_test)
robust.fit(X_train, y_train)
coeff_df = pd.DataFrame(robust.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
y_pred = robust.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df.to_csv(r'regressionStreamWithoutWindow.csv', index = False)


data = pd.read_csv('regressionStreamWithoutWindow.csv')
dataset = data.values
dataset = data.astype('float32')

print(dataset)
test=[]
predict = []

for p in range(0,len(dataset)):
    #if(dataset.iloc[p]['Actual'] <= 40):
        test.append([dataset.iloc[p]['Actual']])
    #if(dataset.iloc[p]['Predicted'] <= 100):
        predict.append([dataset.iloc[p]['Predicted']])


scaler = MinMaxScaler(feature_range=(-1, 1))

liSmaller = scaler.fit_transform(test)
liGrater = scaler.fit_transform(predict)


r = liSmaller[0:300]
p = liGrater[0:300]

label2 = plt.plot(scaler.inverse_transform(p),c='g' ,label = 'Predicted TT')
label1 = plt.plot(scaler.inverse_transform(r),c='r',label = 'Actual TT')

plt.legend()
plt.show()



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))