def load_data_csv(filename , split_char = ","):
	try:
		df = []
		with open(filename, "r") as file:
			print("file exists")
			for line in file:
				temp = line.split(split_char)
				temp.pop(0)
				split = []
				for element in temp:
					element = element.replace("\n", "")
					split.append(element)
				df.append(split)
		return df
	except:
		print("file does not exist")
		return None

def pop_all_data_at_index(df, index):
	for line in df:
		line.pop(index)
	return df

def select_data_at_index(df, index):
	data = []
	for line in df:
		data.append(line[index])
	return data

			