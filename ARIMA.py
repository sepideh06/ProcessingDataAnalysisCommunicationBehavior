import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

data = pd.read_csv('UpStateFunction(303DARIMA).csv',index_col=0)
data.head()


data.index = pd.to_datetime(data.index)

data.plot()
pyplot.show()

result = seasonal_decompose(data, freq = 52, model='multiplicative')
result.plot()
pyplot.show()


from pyramid.arima import auto_arima

stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

print(stepwise_model.aic())




train = data.loc['6/19/2019':'4/3/2026']
test = data.loc['4/13/2026':]

stepwise_model.fit(train)

future_forecast = stepwise_model.predict(n_periods=54)

# This returns an array of predictions:
#print(future_forecast)

future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])

print(test.head())

plt.figure(figsize=(10,6))
plt.plot(test, color='blue', label='Actual Up-State Function values')
plt.plot(future_forecast , color='red', label='Predicted Up-State Function values')
plt.title('Up-State Function Prediction')
plt.xlabel('TimeInterval')
plt.ylabel('Up-State Function values')
plt.legend()
plt.show()