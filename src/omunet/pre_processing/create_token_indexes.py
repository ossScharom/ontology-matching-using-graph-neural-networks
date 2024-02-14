import logging
import pickle as pkl
from collections import defaultdict
from pathlib import Path

import pandas as pd

logger = logging.getLogger("omunet")


"""
One node has a text, the text has tokens, tokens are used to create an embedding.
This function maps all tokens that occur within a single ontology to node ids.
Each unique token maps to a list of node ids that contain the token in question.
"""


def create_token_indexes(processed: Path, token_index: Path):
    token_index.mkdir(exist_ok=True)

    for nodes_file_path in processed.glob("**/nodes.parquet"):
        onto_path = nodes_file_path.parent
        onto_name = onto_path.name
        token_index_path = Path(onto_path, "token_index.pkl")
        embeddings_path = Path(nodes_file_path.parent, "embeddings.parquet")

        if token_index_path.exists():
            logger.info("\tToken index for %s already exist", onto_name)
            continue

        logger.info("\t Creating token index for %s", onto_name)
        df_nodes = pd.read_parquet(nodes_file_path, columns=["id", "use_in_alignment"])
        df_embeddings = pd.read_parquet(embeddings_path)
        df = df_embeddings.merge(df_nodes, on="id")
        df = df[df["use_in_alignment"]]

        index = defaultdict(list)
        for _, row in df.iterrows():
            for token in row["token_ids"]:
                index[token].append(row["id"])

        pkl.dump(index, open(token_index_path, "wb"))
