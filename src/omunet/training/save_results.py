import pickle as pkl
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def save_train_val_result(history, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    pkl.dump(history, open(str(Path(path, "train_result.pkl")), "wb"))


def save_test_result(test_metrics, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    pkl.dump(test_metrics, open(str(Path(path, "test_result.pkl")), "wb"))


def save_plots(history, path):
    for metric in history["train"][0].keys():
        epochs = len(history["train"])
        data_train = [x[metric] for x in history["train"]]
        data_val = [x[metric] for x in history["val"]]
        save_plot(metric, epochs, data_train, data_val, path)


def save_plot(metric, epochs, data_train, data_val, path):
    fig, ax = plt.subplots()

    ax.plot(range(epochs), data_train, linewidth=2.0, label="train")
    ax.plot(range(epochs), data_val, linewidth=2.0, label="val")

    ax.set(xlim=(0, epochs), xticks=np.arange(0, epochs, 5))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.legend()
    fig.tight_layout()
    plt.savefig(str(Path(path, f"{metric}.png")))


def save_current_node_embeddings(task, embedding_path, h_first, h_second):
    name = task.value.split("_")
    if len(name) == 3:
        first, _, second = name
    if len(name) == 4:
        first, _, second, onto = name
        first = first + "_" + onto
        second = second + "_" + onto

    pkl.dump({first: h_first, second: h_second}, open(embedding_path, "wb"))
