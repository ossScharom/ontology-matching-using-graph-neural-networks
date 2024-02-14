import logging
import os
from pathlib import Path

from omunet.shared.graph_dataset import OmunetGraphDataset
from omunet.shared.util import OntoMatchingPairs

os.environ["DGLBACKEND"] = "pytorch"


logger = logging.getLogger("omunet")


def create_graph_datasets(
    processed_path: Path,
    processed_dgl_graphs: Path,
    ontology_matching_pairs: dict,
    use_virtual_nodes: bool,
):
    """
    TODO
    """
    if not processed_dgl_graphs.exists():
        processed_dgl_graphs.mkdir()

    for matching_pair in OntoMatchingPairs:
        pair_config = ontology_matching_pairs[matching_pair.value]

        if len(list(Path(processed_dgl_graphs).glob(f"{matching_pair.value}*"))) > 0:
            logger.info("\tDGLGraph %s already exist", matching_pair.value)
            continue

        OmunetGraphDataset(
            matching_pair.value,
            processed_dgl_graphs,
            Path(
                processed_path,
                pair_config["first_onto"],
            ),
            Path(
                processed_path,
                pair_config["second_onto"],
            ),
            Path(
                processed_path,
                "reference_matches",
                pair_config["reference_matching"],
            ),
            use_virtual=use_virtual_nodes,
        ).save()
