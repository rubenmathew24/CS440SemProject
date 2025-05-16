import re
import json
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os

custom_palette = {
    "baseline": "#4c72b0",
    "AL1": "#55a868",
    "AL2": "#c44e52"
}


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

def grab_round_data(methods = ["AL1", "AL2"], sizes = ["small", "medium", "large"], types = ["binary", "multi"], statistics = ["Accuracy","F1 score","Training Time"]):

	database = {learning: {class_type: {size: {} for size in sizes} for class_type in types} for learning in methods}

	# Clean up the numbers
	pattern = r'\d+(?:\.\d+)?'

	for learning in methods:
		for size in sizes:
			for class_type in types:
				path = f"models/{learning}/{size}/{class_type}_{size}/results.txt"

				with open(path, "r") as f:

					# Grab Results from File
					lines = "".join(f.readlines())
					found_rounds = lines.split("Round")[1:]

					found_rounds[-1] = found_rounds[-1].split("===== Results =====")[0]

					# print(*found_results, sep="\n\n\n")

				for i, round in enumerate(found_rounds, 1):
					database[learning][class_type][size][i] = {}

					for line in round.split("\n"):
						if ":" not in line:
							continue
						cleaned = line.strip().split(":")
						stat, value = cleaned[0], cleaned[1]

						if stat not in statistics:
							continue
						
						database[learning][class_type][size][i][stat.lower()] = float(re.findall(pattern, value)[0])
						
	return database

# Easy viewing of data (Debugging purposes)
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
	sb.barplot(data=df, x="Size", y=statistic, hue="Method", palette=custom_palette)
	plt.ylim(y_min, y_max)
	plt.title(f"{classification_type.capitalize()} Classification {statistic} by Dataset Size and Method")
	plt.ylabel(statistic)
	plt.xlabel("Dataset Size")
	plt.legend(title="Method")
	plt.tight_layout()

	if not save:
		plt.show()
	else:
		save_dir = "graphs/statistics"
		os.makedirs(save_dir, exist_ok=True)
		filename = f"{classification_type}_{statistic.replace(' ', '_')}.png"
		filepath = os.path.join(save_dir, filename)
		plt.savefig(filepath)
		plt.close()

# Round Line Graph Creator
def rounds_line_graph(db, class_type, size, statistic, baseline_value, save=True):
	methods = ["AL1", "AL2"]
	data = {"Round": [], "Method": [], statistic: []}

	# Dynamically determine rounds from one method
	for method in methods:
		round_keys = list(db[method][class_type][size].keys())
		round_numbers = sorted(int(r) for r in round_keys)

		for r in round_numbers:
			val = db[method][class_type][size][r][statistic]
			data["Round"].append(r)
			data["Method"].append(method)
			data[statistic].append(val)

	df = pd.DataFrame(data)

	# Plot
	plt.figure(figsize=(8, 6))
	sb.lineplot(data=df, x="Round", y=statistic, hue="Method", marker="o", palette=custom_palette)

	# Add baseline line
	plt.axhline(baseline_value, color=custom_palette['baseline'], linestyle='--', label='Baseline')

	plt.title(f"{class_type.capitalize()} {statistic} over Rounds ({size.capitalize()} Set)")
	plt.xlabel("Round Number")
	plt.ylabel(statistic)
	plt.xticks(round_numbers)
	plt.legend()
	plt.tight_layout()

	if not save:
		plt.show()
	else:
		save_dir = "graphs/rounds"
		os.makedirs(save_dir, exist_ok=True)
		filename = f"{class_type}_{size}_{statistic.replace(' ', '_')}.png"
		filepath = os.path.join(save_dir, filename)
		plt.savefig(filepath)
		plt.close()

# Iterator for Bar Graphs
def basic_stats_graphs():
	final_data = grab_final_data()
	# print_json(db)

	for class_type in final_data["baseline"].keys():
		available_statistics = final_data["baseline"][class_type]["small"].keys()

		for stat in available_statistics:
			bar_graph(final_data, class_type, stat)

# Iterator for Line Graphs
def all_round_graphs():
	round_db = grab_round_data()
	final_db = grab_final_data()

	statistics = ["Accuracy", "F1 Score"]

	for class_type in final_db["baseline"].keys():
		for size in final_db["baseline"][class_type].keys():
			for stat in statistics:
				
				base_stat = stat
				if class_type == "multi" and stat == "F1 Score":
					base_stat = "Weighted F1 Score"

				# print(class_type, size, base_stat)

				base = final_db["baseline"][class_type][size][base_stat]
				rounds_line_graph(round_db, class_type, size, stat.lower(), base)




def main():
	basic_stats_graphs()
	all_round_graphs()

if __name__ == "__main__":
	main()