import bybit
import pandas as pd
import time
from requests import Request, Session
import pandas as pd
import datetime as dt

def str_to_round(number):
	try:
		if float(number) < 1:
			return float(number)
		return round(float(number))
	except Exception as e:
		print("Error: ", e)

def log_error(error_massage):
	try:
		with open("error.txt", "r") as file:
			print("file exists")
			now = dt.datetime.now()
			now = now.strftime("%Y-%m-%d %H:%M:%S")
			file.write(now+"\n"+error_massage+"\n")
	except:
		print("file does not exist")
		now = dt.datetime.now()
		now = now.strftime("%Y-%m-%d %H:%M:%S")
		with open("error.txt", "w") as file:
			file.write(now+"\n"+error_massage+"\n")
			
	

def get_data():
	try:
		client = bybit.bybit(test=False,api_key="FwnzUmI53zrJNQErck",api_secret="oXFqQ4dnxeKBx8ghdhwBMdc1AAtNLHeg3wlM")
		info = client.Market.Market_symbolInfo().result()
		keys = info[0]["result"]
		btc = keys[41]
		price = btc["last_price"]
		prev24_price = btc["prev_price_24h"]
		percent = (int(float(price)) - int(float(prev24_price))) / int(float(prev24_price)) * 100
		high_24h = btc["high_price_24h"]
		low_24h = btc["low_price_24h"]
		prev1_price = btc["prev_price_1h"]
		open_interest = btc["open_interest"]
		turnover24 = btc["turnover_24h"]
		volume24 = btc["volume_24h"]
		funding_rate = btc["funding_rate"]
		current_time = time.time()
		
		temp = [current_time, price , prev24_price, percent, high_24h, low_24h, prev1_price, open_interest, turnover24, volume24, funding_rate]
		labels = ["time", "price" , "prev24_price", "percent", "high_24h", "low_24h", "prev1_price", "open_interest", "turnover24", "volume24", "funding_rate"]
		data = [str_to_round(value) for value in temp]
		df = pd.DataFrame([data], columns=labels)
		return df
	except Exception as e:
		print("Error",e)
		log_error(str(e))

def main():
	while True:
		try:
			df = get_data()
			df.to_csv("bybit.csv", mode="a", header=False)
			print("Data saved")
		except Exception as e:
			print("Error",e)
			log_error(str(e))
		time.sleep(60)


if __name__ == "__main__":
	main()


