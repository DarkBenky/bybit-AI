import pandas as pd
import os

def load_df():
	lines = []
	with open("bybit.csv", "r") as file:
		for line in file:
			temp = line.split(",")
			temp.pop(0)
			split = []
			for element in temp:
				element = element.replace("\n", "")
				split.append(element)
			lines.append(split)
	return lines

def unpack(data):
	unpacked_data = []
	for i in range(0, len(data)):
		for j in range(0, len(data[i])):
			unpacked_data.append(data[i][j])
	return unpacked_data

def formatting(look_up = 50, data = []):
	formatted_data = []
	for i in range(look_up, len(data)):
		lines = []
		predicted_data = data[i][1]
		lines.append([predicted_data])
		for j in range(i-1 , i - look_up-  1, -1):
			lines.append(data[j])
		formatted_data.append(lines)
	return formatted_data
		
	

def main():
	df = load_df()
	formatted_data = formatting(look_up=60, data=df)
	# check if file formatted_data.csv exists
	try:
		with open("formatted_data.csv", "r") as file:
			print("file exists")
		os.remove("formatted_data.csv")
	except:
		print("file does not exist")

	for i in range(1, len(formatted_data)-1):
		data = pd.DataFrame(unpack(formatted_data[i]))
		data = data.T
		data.to_csv("formatted_data.csv", mode="a", header=False)
main()
			

