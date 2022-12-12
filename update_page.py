import page_backend
import bybit_data
import time

while True:
	try:
		page_backend.predict_next_value()
		print("Updated page")
	except Exception as e:
		print("Error",e)
		bybit_data.log_error(e)
	time.sleep(600)