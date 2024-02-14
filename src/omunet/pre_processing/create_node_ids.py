import logging
import pickle as pkl
from pathlib import Path

import pandas as pd

from omunet.pre_processing.label_encoder import LabelEncoder

logger = logging.getLogger("omunet")


def create_node_ids_and_label_encoder(processed: Path):
    for nodes_file_path in processed.glob("**/nodes.parquet"):
        onto_path = nodes_file_path.parent
        edges_file_path = Path(onto_path, "edges.parquet")
        label_encoder_path = Path(onto_path, "label_encoder.pkl")
        onto_name = onto_path.name

        if label_encoder_path.exists():
            logger.info("\t Label encoder and node ids already exist")
            continue

        logger.info("\t Creating node ids and label encoder for %s", onto_name)

        nodes_df = pd.read_parquet(nodes_file_path)
        edges_df = pd.read_parquet(edges_file_path)

        le = LabelEncoder(nodes_df.id)
        nodes_df.id = le.transform(nodes_df.id)
        edges_df.src = le.transform(edges_df.src)
        edges_df.tgt = le.transform(edges_df.tgt)

        pkl.dump(le, open(label_encoder_path, "wb"))
        nodes_df.to_parquet(nodes_file_path)
        edges_df.to_parquet(edges_file_path)
