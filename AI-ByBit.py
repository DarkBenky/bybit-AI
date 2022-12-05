from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.layers import Dense , Dropout , LSTM
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import goose 

def accuracy(predictions, y_test):
	accuracy = 0
	for i in range(len(predictions)):
		accuracy += abs(predictions[i] - y_test[i])
	accuracy = accuracy / len(predictions)
	return accuracy

def average_direction_accuracy(predictions, y_test):
	accuracy = 0
	for i in range(len(predictions)):
		if predictions[i] > y_test[i]:
			accuracy += 1
		elif predictions[i] < y_test[i]:
			accuracy -= 1
	accuracy = accuracy / len(predictions)
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
	print(x_train, y_train)
	with tf.device("/gpu:0"):
		model = Sequential()
		model.add(LSTM(256, input_shape=(x_train.shape[1],1), return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(512, return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(32, return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(16))
		model.add(Dropout(0.2))
		model.add(Dense(1))
		
		
		model.compile(loss="mean_squared_error", optimizer="adam")
		model.fit(x_train, y_train, epochs=50, batch_size=32)

		model.save("model.h5")
		model.save_weights("model_weights.h5")
		
		predictions = model.predict(x_test)
		predictions = scalar.inverse_transform(predictions)
		y_test = scalar.inverse_transform(y_test)
		print(accuracy(predictions, y_test))
		print(average_direction_accuracy(predictions, y_test))


		
 

def predict_price(model, data):
	#load tensorflow model
	model = load_model(model)
	#load weights
	model.load_weights("model_weights.h5")
	#predict price
	prediction = model.predict(data)
	return prediction


if __name__ == "__main__":
	main()
	
