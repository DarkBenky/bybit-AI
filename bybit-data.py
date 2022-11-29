import bybit
import pandas as pd
import time
from requests import Request, Session
import smtplib

def get_data():
	client = bybit.bybit(test=False,api_key="FwnzUmI53zrJNQErck",api_secret="oXFqQ4dnxeKBx8ghdhwBMdc1AAtNLHeg3wlM")
	info = client.Market.Market_symbolInfo().result()
	keys = info[0]["result"]
	btc = keys[0]
	print(btc)

get_data()


