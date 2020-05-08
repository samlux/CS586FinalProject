import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(7)

path = 'Data/'
filename = 'WA_cases'
fileObject = open(path + filename, 'rb')
NJ_cases = pkl.load(fileObject)
fileObject.close()

scaler = MinMaxScaler(feature_range = (0, 1))
data = scaler.fit_transform(NJ_cases.reshape((-1,1)))

# split into train and test sets
train_size = int(len(data) * 0.85)
test_size = len(data) - train_size
train, test = data[0:train_size], data[train_size:len(data)]


def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print(len(trainX), len(testX))
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(1000))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
model.summary()
#plot_model(model, to_file='Figures/WA_model.png', show_shapes=True, show_layer_names=True)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))

model_json = model.to_json()
with open("Models/WA_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Models/WA_model.h5")
print("Saved model to disk")

trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(data)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(data), label = 'Actual')
plt.plot(trainPredictPlot, label = 'Train Prediction')
plt.plot(testPredictPlot, label = 'Test Prediction')
plt.title('Learning WA COVID19 Spread with LSTM')
plt.xlabel('Days since First Case')
plt.ylabel('Number of CVID19 Cases')
plt.legend()
plt.savefig('Figures/WA_LSTM_Test_num_cases.png')
