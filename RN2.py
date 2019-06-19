import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import tensorflow as tf

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


apple_training_processed = pd.read_csv('UpStateFunction(1).csv', usecols=[1])
scaler = MinMaxScaler(feature_range = (0, 1))
apple_training_scaled = scaler.fit_transform(apple_training_processed)


features_set = []
labels = []
for i in range(10, 5022):
    features_set.append(apple_training_scaled[i-10:i, 0])
    labels.append(apple_training_scaled[i, 0])


features_set, labels = np.array(features_set), np.array(labels)

features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))



model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(features_set, labels, epochs = 20, batch_size = 32)



apple_testing_processed = pd.read_csv('upstatefunction_testing.csv',usecols=[1])
apple_total = pd.concat((apple_training_processed['Up-State-Value'], apple_testing_processed['Up-State-Value']), axis=0)

test_inputs = apple_total[len(apple_total) - len(apple_testing_processed) - 10:].values
test_inputs = test_inputs.reshape(-1,1)
test_inputs = scaler.transform(test_inputs)


test_features = []
for i in range(10, 2476):
    test_features.append(test_inputs[i-10:i, 0])

test_features = np.array(test_features)
test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))


predictions = model.predict(test_features)

predictions = scaler.inverse_transform(predictions)

plt.figure(figsize=(10,6))
plt.plot(apple_testing_processed, color='blue', label='Actual Up-State Function values')
plt.plot(predictions , color='red', label='Predicted Up-State Function values')
plt.title('Up-State Function Prediction')
plt.xlabel('TimeInterval')
plt.ylabel('Up-State Function values')
plt.legend()
plt.show()





