# %%
import pickle as pkl
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results_history_path = "../result_history/with_virtual_nodes_and_gcn"

# %%
test_set_results = list()

for file in Path(results_history_path).rglob("test_result.pkl"):
    _, _, _, experiment, _, _, task_with_timestamp, _ = str(file).split("/")
    task = task_with_timestamp.split("_2024")[0]
    settings = {
        "experiment_name": experiment,
        "task": task,
    }
    test_results = pkl.load(open(file, "rb"))
    test_set_results.append({**settings, **test_results})


# %%
test_df = pd.DataFrame(test_set_results)

# %%
test_df

# %%
grouped = test_df.groupby(["experiment_name", "task"])


# standard deviation by randomseed
mean = grouped.mean()
std = grouped.std()

# save to disk
mean.to_csv("mean.csv")
std.to_csv("std.csv")

# %%
# remove undercores from the experiment_name
test_df["experiment_name"] = test_df["experiment_name"].str.replace("_", " ")
test_df["experiment_name"] = test_df["experiment_name"].str.replace("-", " ")
test_df["Experiment Name"] = test_df["experiment_name"].str.title()

replacements = {
    "NCIT_TO_DOID": "NCIT-DOID",
    "OMIM_TO_ORDO": "OMIM-ORDO",
    "SNOMED_TO_FMA_BODY": "SNOMED-FMA (Body)",
    "SNOMED_TO_NCIT_NEOPLAS": "SNOMED-NCIT (Neoplas)",
    "SNOMED_TO_NCIT_PHARM": "SNOMED-NCIT (Pharm)",
}

test_df.replace(replacements, inplace=True)


# remove the underscores form the tasks
test_df["Task"] = test_df["task"].str.replace("_", " ")

sns.pointplot(
    data=test_df,
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
plt.savefig("f1_score_test_negative_node_sampling.png", dpi=300)
# %%
