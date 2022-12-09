import pandas as pd
import plotly.express as px
import numpy as np
import AI_ByBit as ai
import goose
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


def create_graph():
	x_test , y_test = ai.prepare_data()
	ai.predict_price_export_csv("model.h5", y_test, x_test)

	df = pd.read_csv("predictions.csv")
	fig = px.line(df, x=df.index, y=["Predictions", "Actual"])
	fig.show()
	fig.write_html("predictions.html")

def predict_next_value():
	scalar = MinMaxScaler(feature_range=(0,1))

	df = goose.load_data_csv("formatted_data.csv")

	y = goose.select_data_at_index(df, 0)
	y = pd.DataFrame(y).astype(float)
	y = scalar.fit_transform(y)
	y = pd.DataFrame(y).astype(float)
	

	x = goose.pop_all_data_at_index(df, 0)
	x = pd.DataFrame(x).astype(float)

	

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1, random_state=42)
	#reshape data
	x_test = np.array(x_test)
	print(x_test.shape)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
	print(x_test.shape)
	#get last value of x_test
	x_test = x_test[-1]
	print(x_test.shape)
	x_test = np.reshape(x_test, (1, x_test.shape[0], 1))
	print(x_test.shape)


	prediction = load_model("model.h5").predict(x_test)
	prediction = scalar.inverse_transform(prediction)

	df = pd.read_csv("bybit_data.csv")
	print(df)

predict_next_value()





