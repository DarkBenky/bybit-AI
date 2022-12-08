from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.layers import Dense , Dropout , LSTM
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import goose

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def accuracy(predictions, y_test):
	"""Returns the accuracy of the model"""
	accuracy = 0
	for i in range(len(predictions)):
		accuracy += abs(predictions[i] - y_test[i]) / y_test[i]
		print(predictions[i], y_test[i])
	accuracy = (accuracy / len(predictions)) * 100
	return accuracy

def main():
	scalar = MinMaxScaler(feature_range=(0,1))

	df = goose.load_data_csv("formatted_data.csv")

	y = goose.select_data_at_index(df, 0)
	y = pd.DataFrame(y).astype(float)
	y = scalar.fit_transform(y)
	y = pd.DataFrame(y).astype(float)
	

	x = goose.pop_all_data_at_index(df, 0)
	x = pd.DataFrame(x).astype(float)

	

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	#reshape data
	x_train = np.array(x_train)
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
	x_test = np.array(x_test)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
	
	model = Sequential()
	model.add(LSTM(256, input_shape=(x_train.shape[1],1), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(512, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(256, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(64, return_sequences=True))
	model.add(Dropout(0.2))
	model.add(LSTM(32, return_sequences=True))
	model.add(Dense(1))
		
		
	model.compile(loss="mean_squared_error", optimizer="adam")
	checkpoint = ModelCheckpoint("model_best.h5", monitor="loss", verbose=1, save_best_only=True, mode="auto")
	with tf.device("/gpu:0"):
		model.fit(x_train, y_train, epochs=10000, batch_size=32, validation_data=(x_test, y_test), callbacks=[checkpoint])

	model.save("model.h5")
	model.save_weights("model_weights.h5")
		
	predictions = model.predict(x_test)
	predictions = scalar.inverse_transform(predictions)
	y_test = scalar.inverse_transform(y_test)
	print("Accuracy: ", accuracy(predictions, y_test))

		
 

def predict_price(model, data, scalar_data):
	scalar = MinMaxScaler(feature_range=(0,1))
	data = scalar.fit_transform(scalar_data)
	#load tensorflow model
	model = load_model(model)
	#load weights
	model.load_weights("model_weights.h5")
	#predict price
	prediction = model.predict(data)
	prediction = scalar.inverse_transform(prediction)
	return prediction


if __name__ == "__main__":
	main()
	
