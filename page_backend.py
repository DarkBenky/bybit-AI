import pandas as pd
import plotly.express as px
import numpy as np
import AI_ByBit as ai
import goose
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import bybit
import time

client = bybit.bybit(test=False,api_key="sak71uwI7bOi4VFMIv",api_secret="Ry7djlWRAOcbzRXdTxqEXIzIZTq2jZKKl3mnK")

def log_error(error):
	with open("error.txt", "a") as file:
		now = time.time()
		file.write(str(error) + " " + str(now) + "\n")

def predict_next_value():
	scalar = MinMaxScaler(feature_range=(0,1))

	df = goose.load_data_csv("formatted_data.csv")

	y = goose.select_data_at_index(df, 0)
	y = pd.DataFrame(y).astype(float)
	y_test = np.array(y)
	y = scalar.fit_transform(y)
	y = pd.DataFrame(y).astype(float)
	
	x = goose.pop_all_data_at_index(df, 0)
	x = pd.DataFrame(x).astype(float)
	x_test = np.array(x)

	
	print(x_test.shape)
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
	print(x_test.shape)
	model = load_model("model.h5")
	model_best = load_model("model_best.h5")
	predictions = []
	for i in range(0 , x_test.shape[0]):
		x = x_test[i]
		x = np.reshape(x, (1, x.shape[0], 1))
		prediction = model.predict(x)
		prediction = scalar.inverse_transform(prediction)
		prediction_best = model_best.predict(x)
		prediction_best = scalar.inverse_transform(prediction_best)
		print(prediction[0][0], prediction_best[0][0], y_test[i][0])
		predictions.append([prediction[0][0], prediction_best[0][0], y_test[i][0]])

	df = pd.DataFrame(predictions)
	df.columns = ["Predictions", "Predictions_best", "Actual"]

	fig = px.line(df, x=df.index, y=df.columns , title="Predictions vs Actual")
	fig.write_html("predictions.html")

def bybit_buy_sell(side):
	try:
		if side == "Buy":
			print(client.Order.Order_new(
				qty=1,
				symbol="BTCUSD",
				order_type="Market",
				time_in_force ="GoodTillCancel" ,
				side="Buy", ).result())
		elif side == "Sell":
			print(client.Order.Order_new(
				qty=1,
				symbol="BTCUSD",
				order_type="Market",
				side="Sell",
				time_in_force ="GoodTillCancel" 
				).result())
	except Exception as e:
		print("Error",e)
		log_error(e)


def bot_predict():
	data = goose.load_data_csv("bybit.csv")
	# get last 60 values
	data = data[-60:]
	price = goose.select_data_at_index(data, 1)
	current_price = price[-1]
	price = pd.DataFrame(price).astype(float)

	scalar = MinMaxScaler(feature_range=(0,1))
	scalar.fit_transform(price)

	data_to_predict = []
	for array in data:
		for item in array:
			data_to_predict.append(item)
	
	data_to_predict = np.array(data_to_predict).astype(float)
	data_to_predict = np.reshape(data_to_predict, (1, data_to_predict.shape[0], 1))


	model = load_model("model.h5")
	model_best = load_model("model_best.h5")

	prediction = model.predict(data_to_predict)
	prediction = scalar.inverse_transform(prediction)

	prediction_best = model_best.predict(data_to_predict)
	prediction_best = scalar.inverse_transform(prediction_best)

	prediction = (prediction[0][0] + prediction_best[0][0]) / 2

	if prediction > float(current_price):
		print("buy")
		print("prediction: ", prediction , "current price: ", current_price)
		bybit_buy_sell("Buy")
	else:
		print("sell")
		print("prediction: ", prediction , "current price: ", current_price)
		bybit_buy_sell("Sell")
	




	
	




