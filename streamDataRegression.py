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

#data = pd.read_csv('LogicalLink-2.csv')
data = pd.read_csv('LogicLINKNew1.csv')
#data['pandas_SMA_3'] = data.iloc[:,1].rolling(window=3).mean()
#print(data.head())


def makeWindowPrediction():

    if (os.path.isfile("regressionStream.csv")):
        os.remove("regressionStream.csv")
        print("File Removed!")

    Str2 = 'Actual' + ',' + 'Predicted'
    with open('regressionStream.csv', 'a', encoding="UTF8") as fd:
        fd.write(Str2 + "\n")


    i = 0
    windowSize = 5
    testSize = 0.2
    saveTestinstances = []
    while ((i + windowSize) < (len(data)* testSize)):
        df2 = pd.DataFrame(columns=data.columns)
        j = i + windowSize
        rows = data.iloc[i:j]
        df2 = df2.append(rows, ignore_index=True)
        #print(df2.head())

        X = df2[['UT', 'CML', 'RP', 'LP','SP']]
        y = df2['TT']

        #X = df2[['TT', 'CML', 'RP', 'LP','SP']]
        #y = df2['UT']


        # X = df2[['TT', 'UT', 'RP', 'LP','SP']]
        # y = df2['CML']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, shuffle=False)

        robust = SGDRegressor(loss='huber',
                          penalty='l2',
                          alpha=0.0001,
                          fit_intercept=False,
                          shuffle=False,
                          verbose=0,
                          epsilon=0.1,
                          random_state=0,
                          learning_rate='invscaling',
                          eta0=0.01,
                          power_t=0.5)
        #robust = LinearRegression()

        #sc = StandardScaler()
        #X_train_ = sc.fit_transform(X_train)
        #X_test_ = sc.transform(X_test)
        robust.fit(X_train, y_train)
        coeff_df = pd.DataFrame(robust.coef_, X.columns, columns=['Coefficient'])
        #print(coeff_df)
        ##Check if there is instance inside X_test for which we already did a prediction,if yes,we need to remove it before prediction
        u = 0
        x = X_test
        y = y_test
        listOfIndexes = []
        #print(X_test)
        for h in range(0, len(X_test)):
            if (X_test.iloc[h]["SP"] in saveTestinstances):
                u = X_test[(X_test.SP == X_test.iloc[h]["SP"])].index.values
                listOfIndexes.append(int(u))

        if (windowSize * 0.2 > 1):
            if(len(listOfIndexes) >= 1) :
                for l in range(0,len(listOfIndexes)):
                    X_test = X_test.drop(listOfIndexes[l])
                    y_test = y_test.drop(listOfIndexes[l])
                    x = X_test
                    y = y_test
        #print(x)
        #print(listOfIndexes)
        listOfIndexes = []

        """
        if (windowSize*0.2 > 1):
                for h in range(0, len(X_test)):
                    if(X_test.iloc[h]["SP"] in saveTestinstances):
                        #print(X_test.iloc[h]["SP"])
                        u = X_test[(X_test.SP == X_test.iloc[h]["SP"])].index.values
                        print(X_test)
                        X_test = X_test.drop(u)
                        y_test = y_test.drop(u)
                        print(X_test)
                        x=X_test
                        y=y_test
        """
        #print(x)
        #print(y)
        y_pred = robust.predict(x)
        df = pd.DataFrame({'Actual': y, 'Predicted': y_pred})
        #print(df)
        #print(df.head())
        #print(str(df.iloc[0]['Actual']) + ',' + str(df.iloc[0]['Predicted']))
        with open('regressionStream.csv', 'a', encoding="UTF8") as fd:
            for t in range(0, len(df)):
                fd.write(str(df.iloc[t]['Actual']) + ',' + str(df.iloc[t]['Predicted']) + "\n")
        #df.iloc[0, 1].to_csv(r'regressionStream.csv', index = False)
        df2.iloc[0:0]
        i = i + 1
        for k in range(0, len(x)):
            if(x.iloc[k]["SP"] not in saveTestinstances):
                saveTestinstances.append(x.iloc[k]["SP"])
        #print(saveTestinstances)

#------------------------------------------------------------------------------
makeWindowPrediction()

data = pd.read_csv('regressionStream.csv')
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

liSmaller = test
liGrater = predict


r = liSmaller[0:1000]
p = liGrater[0:1000]

label2 = plt.plot(p,c='g' ,label = 'Predicted TT')
label1 = plt.plot(r,c='r',label = 'Actual TT')

plt.legend()
plt.show()



#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))