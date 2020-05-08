import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import math
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

np.random.seed(7)

path = 'Data/'

sample_length = [0]

filename = 'NJ.pkl'
fileObject = open(path + filename, 'rb')
NJ_data = pkl.load(fileObject)
fileObject.close()
sample_length.append(sample_length[-1] + len(NJ_data[0]))

filename = 'NY.pkl'
fileObject = open(path + filename, 'rb')
NY_data = pkl.load(fileObject)
fileObject.close()
sample_length.append(sample_length[-1] + len(NY_data[0]))

filename = 'CA.pkl'
fileObject = open(path + filename, 'rb')
CA_data = pkl.load(fileObject)
fileObject.close()
sample_length.append(sample_length[-1] + len(CA_data[0]))

filename = 'TX.pkl'
fileObject = open(path + filename, 'rb')
TX_data = pkl.load(fileObject)
fileObject.close()
sample_length.append(sample_length[-1] + len(TX_data[0]))

filename = 'WA.pkl'
fileObject = open(path + filename, 'rb')
WA_data = pkl.load(fileObject)
fileObject.close()
sample_length.append(sample_length[-1] + len(WA_data[0]))

filename = 'FL.pkl'
fileObject = open(path + filename, 'rb')
FL_data = pkl.load(fileObject)
fileObject.close()
sample_length.append(sample_length[-1] + len(FL_data[0]))

filename = 'MA.pkl'
fileObject = open(path + filename, 'rb')
MA_data = pkl.load(fileObject)
fileObject.close()
sample_length.append(sample_length[-1] + len(MA_data[0]))



data = np.concatenate((NJ_data, NY_data, CA_data, TX_data, WA_data, FL_data, MA_data), axis = 1)
# data = np.concatenate((NJ_data, NY_data, MA_data), axis = 1)

# select features
feature_used = [0,1,2,3,4,5]
data = data[feature_used,:]
feature_count = len(feature_used)

# data normalization
def data_norm(data):
	data_to_scale = np.transpose(np.array(data))
	scaler = MinMaxScaler(feature_range = (0, 1))
	scaled_data = scaler.fit_transform(data_to_scale)
	return scaler, scaled_data

scaler, scaled_data = data_norm(data)





# convert series to supervised learning
def series_to_supervised(data, sample_length, look_back_points = 5, label_loc = 1):
	learnable_data = []
	learnable_label = []
	# print(sample_length)
	# print(np.array(data).shape)
	for j in range(len(sample_length)-1):
		data_to_convert = data[sample_length[j]:sample_length[j+1], :]
		timepoints = len(data_to_convert)
		for i in range(timepoints - look_back_points):
			learnable_data.append(data[i:i+look_back_points, :])
			learnable_label.append(data[i+look_back_points, label_loc])
	return np.array(learnable_data), np.array(learnable_label)




# for multiple state
look_back = 5
train_data = scaled_data[0:sample_length[-2], :]
train_sample_length = sample_length[0:-1]
test_data = scaled_data[sample_length[-2]:sample_length[-1], :]
converted_train, train_label = series_to_supervised(train_data, train_sample_length, look_back, 1)






model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(look_back, feature_count)))
model.add(LSTM(1000))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(converted_train, train_label, epochs=1000, batch_size=200, verbose=2)
model.summary()
# plot_model(model, to_file='Figures/NJ_model.png', show_shapes=True, show_layer_names=True)

model.save('Models/Multi_model.h5')
print("Saved model to disk")



# test_data = scaled_data[sample_length[0]:sample_length[1], :]
# Multi_State
model = load_model('Models/Multi_model_NJ_NY.h5')
ma_data = np.array(test_data)
actual_label_for_plot = MA_data[1]
for i in range(look_back, len(ma_data)):
	data_to_prdict = np.expand_dims(ma_data[i-look_back:i], axis=0)
	testPredict = model.predict([data_to_prdict])
	ma_data[i,1] = testPredict
	if i != look_back:
		testPredict -= ma_data[look_back,1]
	ma_data[i,1] = testPredict
ma_data[look_back,1] = 0
testPredictPlot = scaler.inverse_transform(ma_data)
# plot baseline and predictions
plt.plot(actual_label_for_plot, label = 'Actual')
plt.plot(testPredictPlot[:,1], label = 'Test Prediction')
plt.title('MA COVID19 Spread Prediction with LSTM')
plt.xlabel('Days since First Case')
plt.ylabel('Number of CVID19 Cases')
plt.legend()
plt.savefig('Figures/MA_Multi_State_10_days_prediction_NJ_NY.png')

# predict future
future_days = 10
model = load_model('Models/Multi_model.h5')
actual_label_for_plot = MA_data[1]
predict_data = np.transpose(np.array(MA_data))
POP = predict_data[0,2]
AREA = predict_data[0,3]

ma_data = np.array(test_data)
for i in range(look_back, len(ma_data)):
	data_to_prdict = np.expand_dims(ma_data[i-look_back:i], axis=0)
	testPredict = model.predict([data_to_prdict])
	ma_data[i,1] = testPredict
	if i != look_back:
		testPredict -= ma_data[look_back,1]
	ma_data[i,1] = testPredict
a = ma_data[look_back,1]
ma_data[look_back,1] = 0
testPredictPlot = scaler.inverse_transform(ma_data)
predict_data[:,1] = testPredictPlot[:,1]
predict_data[:,4] = testPredictPlot[:,1] / POP
predict_data[:,5] = testPredictPlot[:,1] / AREA


for i in range(future_days):
	scaled_predict_data = scaler.transform(predict_data)
	data_to_prdict = np.expand_dims(scaled_predict_data[-5:], axis=0)
	testPredict = model.predict(data_to_prdict)
	temp= scaled_predict_data[-1]
	temp[1] = testPredict[0] - a
	new_data_to_add = scaler.inverse_transform(np.expand_dims(temp, axis=0))
	new_data_to_add[0][0] = predict_data[-1,0] + 1
	new_data_to_add[0][4] = new_data_to_add[0][1] / POP
	new_data_to_add[0][5] = new_data_to_add[0][1] / AREA
	predict_data = np.append(predict_data, new_data_to_add, axis = 0)
PredictPlotData = np.array(predict_data[:,1])
PredictPlot = np.empty_like(PredictPlotData)
PredictPlot[:] = np.nan
PredictPlot[-future_days:] = PredictPlotData[-future_days:]

# plot baseline and predictions
plt.plot(actual_label_for_plot, label = 'Actual')
plt.plot(testPredictPlot[:,1], label = 'Test Prediction')
plt.plot(PredictPlot, label = 'Future Prediction')
plt.title('MA COVID19 Spread Prediction with LSTM')
plt.xlabel('Days since First Case')
plt.ylabel('Number of CVID19 Cases')
plt.legend()
plt.savefig('Figures/MA_Multi_State_10_days_prediction.png')



# save data

b=testPredictPlot[:,1]
c=actual_label_for_plot
b=np.array(b).flatten()
c=np.array(c).flatten()
day=list(range(b.shape[0]))
#%%
import pandas
df = pandas.DataFrame(data={"day": day,"actual": c, "testPredict": b})
df.to_csv("MA_Data.csv", sep=',',index=False)
