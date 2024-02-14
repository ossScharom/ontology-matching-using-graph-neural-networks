import logging
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from omunet.shared.graph_dataset import EMBEDDING_LENGTH

logger = logging.getLogger("omunet")


def add_virtual_nodes(processed: Path, token_index: Path):
    token_index.mkdir(exist_ok=True)

    for nodes_file_path in processed.glob("**/nodes.parquet"):
        onto_path = nodes_file_path.parent
        edges_file_path = Path(onto_path, "edges.parquet")
        embeddings_file_path = Path(onto_path, "embeddings.parquet")
        token_index_path = Path(onto_path, "token_index.pkl")

        onto_name = onto_path.name
        nodes_df = pd.read_parquet(nodes_file_path)

        if "is_virtual" in nodes_df.columns:
            continue
        logger.info("\tAdding virtual nodes to %s", onto_name)

        index = pkl.load(open(token_index_path, "rb"))
        nodes_df = pd.read_parquet(nodes_file_path)
        edges_df = pd.read_parquet(edges_file_path)
        embeddings = pd.read_parquet(embeddings_file_path)

        if "is_virtual" in nodes_df.columns:
            logger.info("\tVirtual nodes for %s already exist", onto_name)
            continue

        df = nodes_df.merge(embeddings, on="id")
        # The token index already contains all the nodes used for the alignment, so
        # we now add the nodes that are not used for alignment to the index. The virtual
        # nodes are only used for the training.
        df = df[~df["use_in_alignment"]]
        for _, row in df.iterrows():
            for token in row["token_ids"]:
                index[token].append(row["id"])

        # Now the nodes not used for the alignment (only for training) are also in the
        # index.
        # We add one virtual node for each token in the index. Then this virtual node
        # is connected to all the nodes that contain the token in question.

        nodes_df["is_virtual"] = False
        for nodes_of_token in tqdm.tqdm(index.values()):
            # append new virtual node
            new_node_id = len(nodes_df)
            nodes_df.loc[new_node_id] = [new_node_id, None, False, True]
            embeddings.loc[len(embeddings)] = [
                new_node_id,
                # Set embedding to a random value
                np.random.random(EMBEDDING_LENGTH).tolist(),
                None,
            ]

            new_edges = pd.DataFrame(
                np.vstack(
                    [
                        np.repeat(
                            new_node_id,
                            len(nodes_of_token),
                        ),
                        nodes_of_token,
                    ]
                ).T,
                columns=["src", "tgt"],
            )

            # append new edges
            edges_df = pd.concat([edges_df, new_edges])

        logger.info("\tAdded %s nodes", nodes_df.is_virtual.value_counts()[True])
        logger.info(
            "\tAdded %s edges",
            edges_df.src.isin(nodes_df[nodes_df.is_virtual].id).value_counts()[True],
        )

        nodes_df.to_parquet(nodes_file_path)
        embeddings.to_parquet(embeddings_file_path)
        edges_df.to_parquet(edges_file_path)
