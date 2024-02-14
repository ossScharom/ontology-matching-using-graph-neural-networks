# %%
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_history_path = "../result_history/full_evaluation"

# %%
test_set_results = list()

for file in Path(results_history_path).rglob("*_result.pkl"):
    _, _, _, file_name = str(file).split("/")
    task = file_name.split("_result.pkl")[0]
    test_results = pkl.load(open(file, "rb"))
    test_set_results.append({"experiment": task, **test_results})


# %%
df = pd.DataFrame(test_set_results)
# save
df.to_csv("full_evaluation.csv")

# %%
df
# %%
