import itertools
import logging
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm

from omunet.shared.util import OntoMatchingPairs

logger = logging.getLogger("omunet")


def create_candidates(
    task: OntoMatchingPairs,
    src_onto_name: str,
    tgt_onto_name: str,
    processed_path: Path,
    candidates_per_node: int,
):
    target_path = Path(processed_path.parent, "candidates")
    target_path.mkdir(exist_ok=True)
    target_file = Path(target_path, f"{task.value}.parquet")

    if target_file.exists():
        logging.info("Candidates for %s already exist.", task)
        return

    src_processed_path = Path(processed_path, src_onto_name)
    tgt_processed_path = Path(processed_path, tgt_onto_name)
    src_embeddings = pd.read_parquet(
        str(Path(src_processed_path, "embeddings.parquet"))
    )
    src_nodes = pd.read_parquet(str(Path(src_processed_path, "nodes.parquet")))[
        ["id", "use_in_alignment"]
    ]
    # merge nodes and embeddings
    src_nodes = pd.merge(src_nodes, src_embeddings, on="id")
    # load tgt embeddings
    tgt_embeddings = pd.read_parquet(
        str(Path(tgt_processed_path, "embeddings.parquet"))
    )
    tgt_nodes = pd.read_parquet(str(Path(tgt_processed_path, "nodes.parquet")))[
        ["id", "use_in_alignment"]
    ]
    # merge tgt_nodes and emebddings on id
    tgt_nodes = pd.merge(tgt_nodes, tgt_embeddings, on="id")

    src_nodes = src_nodes[src_nodes.use_in_alignment]
    tgt_nodes = tgt_nodes[tgt_nodes.use_in_alignment]

    src_index = pkl.load(
        open(
            Path(src_processed_path, "token_index.pkl"),
            "rb",
        )
    )
    tgt_index = pkl.load(
        open(
            Path(tgt_processed_path, "token_index.pkl"),
            "rb",
        )
    )

    candidates = generate_candidates(
        src_index, tgt_index, src_nodes, tgt_nodes, candidates_per_node
    )

    candidates = pd.DataFrame(candidates, columns=["src", "tgt"])
    candidates.to_parquet(target_file)


def generate_candidates(
    src_index,
    tgt_index,
    src_nodes,
    tgt_nodes,
    k,
):
    """
    Creates a combined set of candidates. First the nodes of the source onto is used
    to create candidates using the index of the target onto, and then the other way
    around. The mappings of the second ontology to the first are reversed to eliminate
    duplicates. This follows the candidate generation of BERTmap.
    """

    src_to_tgt = query_index(
        src_nodes,
        len(tgt_nodes),
        tgt_index,
        k,
    )
    tgt_to_src = query_index(
        tgt_nodes,
        len(src_nodes),
        src_index,
        k,
    )

    # reverse tgt_to_src to be src_to_tgt again
    return {*src_to_tgt, *{(b, a) for a, b in tgt_to_src}}


def query_index(
    nodes: pd.DataFrame,
    amount_target_classes: int,
    index: dict,
    k: int,
):
    candidates = set()
    index = [[(token_id, c) for c in node_ids] for token_id, node_ids in index.items()]
    index = itertools.chain(*index)
    index = pd.DataFrame(index, columns=["token_id", "node_id"]).set_index("token_id")

    for node in tqdm.tqdm(nodes.itertuples(index=False), total=len(nodes)):
        target_candidates = index.loc[index.index.intersection(node.token_ids)]

        if len(target_candidates) <= 0:
            continue

        # calculate idf for each token
        target_candidates["idf"] = np.log10(
            amount_target_classes
            / target_candidates.groupby("token_id")
            .count()
            .rename({"node_id": "idf"}, axis=1)
        )

        # The target_candidates contain duplicate ids, we sum the idfs for each token
        # with an aggregation over the potential node_ids of the target
        aggregated_idfs = target_candidates.groupby("node_id").sum()

        # Sorts the target_candidates, higher summed idf score is better, selects
        # the top k
        top_k = (
            aggregated_idfs.sort_values(by="idf", ascending=False)[:k]
            .reset_index()
            .node_id
        )

        # Convert DataFrame to tuples with (src, tgt)
        final_candidates = set(list((node.id, tgt) for tgt in top_k))

        candidates = candidates.union(final_candidates)

    return candidates
