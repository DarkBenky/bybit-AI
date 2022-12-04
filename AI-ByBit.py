from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.layers import Dense , Dropout , LSTM
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import goose 

print(tf.config.list_physical_devices('GPU'))


def main():
	df = goose.load_data_csv("formatted_data.csv")
	df_pandas = pd.DataFrame(df)

	x = goose.pop_all_data_at_index(df, 1)
	x = pd.DataFrame(x).astype(float)
	y = goose.select_data_at_index(df, 1)
	y = pd.DataFrame(y).astype(float)
	
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	with tf.device("/gpu:0"):
		model = Sequential()
		model.add(LSTM(256, input_shape=(x_train.shape[1],1), activation="relu", return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(512, activation="relu"))
		model.add(Dropout(0.2))
		model.add(Dense(128, activation="relu"))
		model.add(Dense(1, activation="sigmoid"))
		
		model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
		model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

		model.save("model.h5")
		predictions = model.predict(x_test)
		print(accuracy_score(y_test, predictions))
 

def predict_price(model, data):
	try:
		model = load_model(model)
		result = model.predict(data)
		return result
	except:
		print("model does not exist")


if __name__ == "__main__":
	main()

