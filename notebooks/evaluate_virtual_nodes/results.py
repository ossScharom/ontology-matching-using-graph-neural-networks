# %%
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# %%
metrics_of_interest = ["precision", "recall", "f1", "mrr", "h1"]


def get_test_set_results(results_history_path):
    test_set_results = list()
    files = Path(results_history_path).rglob("test_result.pkl")
    for file in files:
        settings = str(file).split("/")
        (
            _,
            _,
            experiment_name,
            randomseed,
            negative_sampling,
            task_name_with_timestamp,
            _,
        ) = settings
        task = task_name_with_timestamp.split("_2024")[0]
        settings_as_dict = {
            "experiment_name": experiment_name,
            "randomseed": randomseed,
            "negative_sampling": negative_sampling,
            "task": task,
        }
        resutls = pkl.load(open(file, "rb"))
        test_set_results.append({**settings_as_dict, **resutls})
    return test_set_results


results_history = "../result_history"
test = get_test_set_results(results_history)


# %%
df = pd.DataFrame(test)
df.drop(columns=["randomseed", "negative_sampling"], inplace=True)

grouped_per_experiment_name = df.drop(columns=["task"]).groupby(["experiment_name"])
mean = grouped_per_experiment_name.mean()
std = grouped_per_experiment_name.std()
mean.to_csv("mean_per_experiment_name.csv")
std.to_csv("std_per_experiment_name.csv")

grouped_per_experiment_name_and_task = df.groupby(["experiment_name", "task"])
mean = grouped_per_experiment_name_and_task.mean()
std = grouped_per_experiment_name_and_task.std()
mean.to_csv("mean_per_experiment_name_and_task.csv")
std.to_csv("std_per_experiment_name_and_task.csv")

# %%
# remove undercores from the experiment_name
df["Experiment Name"] = df["experiment_name"].str.replace("_", " ")
df["Experiment Name"] = df["Experiment Name"].str.title()

# rename experiment values with a dict
replacements = {
    "NCIT_TO_DOID": "NCIT-DOID",
    "OMIM_TO_ORDO": "OMIM-ORDO",
    "SNOMED_TO_FMA_BODY": "SNOMED-FMA (Body)",
    "SNOMED_TO_NCIT_NEOPLAS": "SNOMED-NCIT (Neoplas)",
    "SNOMED_TO_NCIT_PHARM": "SNOMED-NCIT (Pharm)",
}

df.replace(replacements, inplace=True)
# remove the underscores form the tasks
df["Task"] = df["task"].str.replace("_", " ")

sns.pointplot(
    data=df,
    x="Task",
    y="f1",
    hue="Experiment Name",
    dodge=0.2,
    linestyle="none",
    marker="_",
    markersize=10,
    markeredgewidth=3,
)
# Rename the y-axis label
plt.ylabel("F1-score")
# Rotate the x-axis ticks to make thme more readable
plt.xticks(rotation=10)
# fit plot
plt.tight_layout()
# Save plot with high resolution for publication
plt.savefig("f1_score_test_virtual_nodes.png", dpi=300)


# %%
