import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm

from omunet.shared.graph_dataset import OmunetGraphDataset
from omunet.shared.util import OntoMatchingPairs


def evaluate(
    task: OntoMatchingPairs,
    processed_path: Path,
    processed_dgl_graphs: Path,
    model_history_path: Path,
):
    src, tgt = task.value.split("_TO_")

    src_folder = src.lower()
    tgt_folder = tgt.lower()
    snomed_in_task = "SNOMED" in task.value
    if snomed_in_task:
        tgt, category = tgt.split("_")
        src_folder = src_folder + "." + category.lower()
        tgt_folder = tgt.lower() + "." + category.lower()
        tgt = tgt + "_" + category
        src = src + "_" + category

    candidates_path = Path(processed_path.parent, "candidates", f"{task.value}.parquet")
    candidates = pd.read_parquet(candidates_path)

    nodes_src = pd.read_parquet(
        Path(processed_path, src_folder, "nodes.parquet"),
        columns=["id", "use_in_alignment"],
    )
    nodes_tgt = pd.read_parquet(
        Path(processed_path, tgt_folder, "nodes.parquet"),
        columns=["id", "use_in_alignment"],
    )

    # sanity check, that candidates only contain node ids that are used for alignment
    assert all(candidates.src.isin(nodes_src[nodes_src.use_in_alignment].id))
    assert all(candidates.tgt.isin(nodes_tgt[nodes_tgt.use_in_alignment].id))

    g = OmunetGraphDataset(task.value, processed_dgl_graphs)
    train_edges = g.reference_matchings[g.reference_matchings.train][["src", "tgt"]]
    val_edges = g.reference_matchings[g.reference_matchings.val][["src", "tgt"]]

    test_edges = g.reference_matchings[g.reference_matchings.test][["src", "tgt"]]

    node_embeddings = pkl.load(
        open(str(Path(model_history_path, f"{task.value}_node_embeddings.pkl")), "rb")
    )

    # sanity check that the node embeddings have a correct 0 dimension
    assert len(node_embeddings[src]) == g.first_graph.num_nodes()
    assert len(node_embeddings[tgt]) == g.second_graph.num_nodes()

    # Remove train and val edges from candidates
    candidates = candidates.drop(
        candidates.reset_index().merge(pd.concat([train_edges, val_edges]))["index"]
    ).reset_index(drop=True)

    # Select node embeddings
    src_embeddings = torch.index_select(
        node_embeddings[src], 0, torch.IntTensor(candidates["src"].values)
    )
    tgt_embeddings = torch.index_select(
        node_embeddings[tgt], 0, torch.IntTensor(candidates["tgt"].values)
    )

    # Calculate similarities, to get matching scores
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    candidates["sims"] = cos(src_embeddings, tgt_embeddings).numpy()
    # candidates["score"] = torch.sum(src_embeddings * tgt_embeddings, dim=-1).numpy()
    result = max_both(candidates)
    test_edges = set(test_edges.itertuples(index=False, name=None))

    TP = len(result.intersection(test_edges))

    precision = TP / len(result)
    recall = TP / len(test_edges)
    F1 = (2 * precision * recall) / (precision + recall)
    path = Path("result_history/full_evaluation")

    path.mkdir(parents=True, exist_ok=True)
    pkl.dump(
        {
            "precision": precision,
            "recall": recall,
            "F1": F1,
            "amount_test_edges": len(test_edges),
        },
        open(f"{str(path)}/{task.value}_result.pkl", "wb"),
    )


def get_best_links_for_partition(candidates_with_similarities, swapped=False):
    L_MAX1 = set()

    for a in tqdm.tqdm(candidates_with_similarities.src.drop_duplicates()):
        tmp = candidates_with_similarities[candidates_with_similarities["src"] == a]
        best_link = candidates_with_similarities.iloc[tmp.sims.idxmax()]

        if not best_link.empty:
            best = (best_link.src, best_link.tgt)
            if swapped:
                best = tuple(reversed(best))
            L_MAX1.add(best)

    return L_MAX1


def max_both(candidates_with_similarities):
    return get_best_links_for_partition(candidates_with_similarities).intersection(
        get_best_links_for_partition(
            candidates_with_similarities.rename({"src": "tgt", "tgt": "src"}, axis=1),
            swapped=True,
        )
    )
