# %%
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_history_path = "../result_history/with_virtual_nodes_and_gcn"

# %%
train_set_results = list()

for file in Path(results_history_path).rglob("train_result.pkl"):
    _, _, _, experiment, random_seed, _, task_with_timestamp, _ = str(file).split("/")
    task = task_with_timestamp.split("_2024")[0]
    settings = {
        "experiment_name": experiment,
        "task": task,
        "random_seed": random_seed,
    }
    test_results = pkl.load(open(file, "rb"))
    for epoch, epoch_val in enumerate(test_results["val"]):
        train_set_results.append({**settings, **epoch_val, "epoch": epoch})


# %%
train_df = pd.DataFrame(train_set_results)

# %%
train_df

# %%
grouped = train_df.groupby(["experiment_name", "task"])


# %%
# remove undercores from the experiment_name
train_df["experiment_name"] = train_df["experiment_name"].str.replace("_", " ")
train_df["experiment_name"] = train_df["experiment_name"].str.replace("-", " ")
train_df["Experiment Name"] = train_df["experiment_name"].str.title()
train_df["Epoch"] = train_df["epoch"]
train_df["Loss"] = train_df["loss"]

replacements = {
    "NCIT_TO_DOID": "NCIT-DOID",
    "OMIM_TO_ORDO": "OMIM-ORDO",
    "SNOMED_TO_FMA_BODY": "SNOMED-FMA (Body)",
    "SNOMED_TO_NCIT_NEOPLAS": "SNOMED-NCIT (Neoplas)",
    "SNOMED_TO_NCIT_PHARM": "SNOMED-NCIT (Pharm)",
}

train_df.replace(replacements, inplace=True)

# remove the underscores form the tasks
train_df["Task"] = train_df["task"].str.replace("_", " ")

# sns.lineplot(data=fmri, x="timepoint", y="signal", hue="event")
ax = sns.lineplot(data=train_df, x="Epoch", y="Loss", hue="Experiment Name")

# Rename the y-axis label
plt.ylabel("Loss")
# Rotate the x-axis ticks to make thme more readable
plt.xlabel("Epoch")
# fit plot
plt.tight_layout()
# Save plot with high resolution for publication
plt.savefig("f1_score_training_negative_node_sampling.png", dpi=300)

# %%

plt.clf()
sns.relplot(
    data=train_df,
    x="Epoch",
    y="Loss",
    hue="Experiment Name",
    col="Task",
    col_wrap=2,
    kind="line",
)
# Rename the y-axis label
plt.ylabel("Loss")
# Rotate the x-axis ticks to make thme more readable
plt.xlabel("Epoch")
# Fit plot
plt.tight_layout()
plt.savefig("f1_score_training_negative_node_sampling_per_task.png", dpi=300)

# %%
