import re
import json
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os

def grab_final_data(methods = ["baseline", "AL1", "AL2"], sizes = ["small", "medium", "large"], types = ["binary", "multi"]):

	database = {learning: {class_type: {} for class_type in types} for learning in methods}

	# Clean up the numbers
	pattern = r'\d+(?:\.\d+)?'

	for learning in methods:
		for size in sizes:
			for class_type in types:
				path = f"models/{learning}/{size}/{class_type}_{size}/results.txt"

				with open(path, "r") as f:

					# Grab Results from File
					lines = "".join(f.readlines())
					found_results = lines.split("===== Results =====")[1].strip()

					# Normalize to dictionary
					split_stats = [(x.split(":")[0], x.split(":")[1]) for x in found_results.split("\n")]
					stripped_dict = {stat.strip():val.strip() for stat,val in split_stats}
					cleaned_results = {stat:float(re.findall(pattern, val)[0]) for stat, val in stripped_dict.items()}

					# Put in Database
					database[learning][class_type][size] = cleaned_results

	return database

# Easy viewing of data
def print_json(db):
	with open("results.json", "w") as f:
		json.dump(db, f, indent=4)


# Basic Bar Graph Creator
def bar_graph(db, classification_type, statistic, save=True):
	data = []
	methods = ["baseline", "AL1", "AL2"]
	sizes = ["small", "medium", "large"]

	for method in methods:
		for size in sizes:
			value = db[method][classification_type][size][statistic]
			data.append({
				"Method": method,
				"Size": size,
				statistic: value
			})

	df = pd.DataFrame(data)
	min_val = df[statistic].min()
	max_val = df[statistic].max()
	padding = (max_val - min_val) * 0.1 if max_val != min_val else 0.1
	y_min = min_val - padding
	y_max = max_val + padding

	# Plot
	plt.figure(figsize=(8, 6))
	sb.barplot(data=df, x="Size", y=statistic, hue="Method")
	plt.ylim(y_min, y_max)
	plt.title(f"{classification_type.capitalize()} Classification {statistic} by Dataset Size and Method")
	plt.ylabel(statistic)
	plt.xlabel("Dataset Size")
	plt.legend(title="Method")
	plt.tight_layout()

	if not save:
		plt.show()
	else:
		save_dir = "graphs"
		os.makedirs(save_dir, exist_ok=True)
		filename = f"{classification_type}_{statistic.replace(' ', '_')}.png"
		filepath = os.path.join(save_dir, filename)
		plt.savefig(filepath)
		plt.close()


final_data = grab_final_data()
# print_json(db)

for class_type in final_data["baseline"].keys():
	available_statistics = final_data["baseline"][class_type]["small"].keys()

	for stat in available_statistics:
		bar_graph(final_data, class_type, stat)






