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

filename = 'WA.pkl'
fileObject = open(path + filename, 'rb')
data = pkl.load(fileObject)
fileObject.close()
sample_length.append(sample_length[-1] + len(data[0]))


# select features
feature_used = [0,1,2,3,4,5]
data = np.array(data)
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


# for single state

look_back = 5
converted_data, converted_label = series_to_supervised(scaled_data, sample_length, look_back, 1)
train_size = int(len(converted_label) * 1)
# test_size = len(converted_label) - train_size
train = converted_data[0:train_size]
train_label = converted_label[0:train_size]
# train, test = converted_data[0:train_size], converted_data[train_size:]
# train_label, test_label = converted_label[0:train_size], converted_label[train_size:]
train = np.array(train)
# test = np.array(test)


model = Sequential()
model.add(LSTM(10, return_sequences=True, input_shape=(look_back, feature_count)))
model.add(LSTM(1000))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train, train_label, epochs=1000, batch_size=100, verbose=2)
model.summary()
# plot_model(model, to_file='Figures/NJ_model.png', show_shapes=True, show_layer_names=True)

model.save('Models/WA_model.h5')
print("Saved model to disk")





# Single State
model = load_model('Models/NJ_model.h5')
Predict_label = model.predict(converted_data)
Predict_data = scaled_data
Predict_data[look_back:, 1] = Predict_label[:,0]
PredictPlot = scaler.inverse_transform(Predict_data)

actual_label_for_plot = data[1,:]


plt.plot(actual_label_for_plot, label = 'Actual')
plt.plot(PredictPlot[:,1], label = 'Prediction')
plt.title('Learning WA COVID19 Spread with LSTM')
plt.xlabel('Days since First Case')
plt.ylabel('Number of CVID19 Cases')
plt.legend()
plt.savefig('Figures/WA_LSTM_Test_num_cases.png')


# predict future
future_days = 10
model = load_model('Models/WA_model.h5')
actual_label_for_plot = data[1]
predict_data = np.transpose(np.array(data))
POP = predict_data[0,2]
AREA = predict_data[0,3]


for i in range(future_days):
	scaled_predict_data = scaler.transform(predict_data)
	data_to_prdict = np.expand_dims(scaled_predict_data[-5:], axis=0)
	testPredict = model.predict(data_to_prdict)
	temp= scaled_predict_data[-1]
	temp[1] = testPredict[0]
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
# plt.plot(testPredictPlot[:,1], label = 'Test Prediction')
plt.plot(PredictPlot, label = 'Future Prediction')
plt.title('WA COVID19 Spread Prediction with LSTM')
plt.xlabel('Days since First Case')
plt.ylabel('Number of CVID19 Cases')
plt.legend()
plt.savefig('Figures/WA_Indivisual_State_10_days_prediction.png')



b=PredictPlot
c=actual_label_for_plot
b=np.array(b).flatten()
c=np.array(c).flatten()
b_length=b.shape[0]
c=np.pad(c,(0,future_days),'constant', constant_values=0)
day=list(range(b_length))

# print(day.shape)
import pandas
df = pandas.DataFrame(data={"day": day,"actual": c, "testPredict": b})
df.to_csv("WA_Data.csv", sep=',',index=False)
