import logging
import os
import pickle as pkl
from pathlib import Path

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
import tqdm
from annoy import AnnoyIndex
from dgl import load_graphs, save_graphs
from dgl.data.utils import load_info, save_info

logger = logging.getLogger("omunet")
EMBEDDING_LENGTH = 768


class OmunetGraphDataset(dgl.data.DGLDataset):
    """
    TODO
    """

    def __init__(
        self,
        name,
        save_dir,
        first_onto=None,
        second_onto=None,
        reference_matchings=None,
        use_virtual=False,
    ):
        self.first_onto = first_onto
        self.second_onto = second_onto
        self.reference_matchings = reference_matchings
        self.use_virtual = use_virtual

        self.first_graph = None
        self.second_graph = None

        self.le_first = None
        self.le_second = None
        self.bfs_neighborhood_mappings = None
        self.first_index = None
        self.second_index = None
        super().__init__(name=name, save_dir=save_dir)

    def process(self):
        logger.info("\tProcessing %s", self.name)
        (
            first_onto_nodes,
            first_onto_edges,
            first_onto_embeddings,
            le_first,
        ) = OmunetGraphDataset.load_onto(self.first_onto)
        (
            second_onto_nodes,
            second_onto_edges,
            second_onto_embeddings,
            le_second,
        ) = OmunetGraphDataset.load_onto(self.second_onto)
        reference_matchings = pd.read_parquet(f"{self.reference_matchings}.parquet")
        first_onto_nodes = first_onto_nodes.merge(first_onto_embeddings, on="id")
        second_onto_nodes = second_onto_nodes.merge(second_onto_embeddings, on="id")

        # Is use virtual is set to False, we remove the virtual nodes
        if not self.use_virtual:
            first_onto_nodes = first_onto_nodes.query("~is_virtual")
            second_onto_nodes = second_onto_nodes.query("~is_virtual")

            first_onto_edges = first_onto_edges[
                first_onto_edges["src"].isin(first_onto_nodes["id"])
            ]
            second_onto_edges = second_onto_edges[
                second_onto_edges["src"].isin(second_onto_nodes["id"])
            ]

        # Annoy index creation. The annoy index is used for negative sampling of
        # nodes. Because we are in self-supervised setting. The graph has only
        # positive examples, the existing edges. So we need to create negative
        # examples to have smth. to learn.
        logger.info(
            "\tBuilding annoy index of %s nodes of %s",
            len(first_onto_nodes),
            self.first_onto.name,
        )
        first_index = self.create_annoy_index(first_onto_nodes.query("~is_virtual"))

        logger.info(
            "\tBuilding annoy index of %s nodes of %s",
            len(second_onto_nodes),
            self.second_onto.name,
        )
        second_index = self.create_annoy_index(second_onto_nodes.query("~is_virtual"))

        # TODO Why not transforming the reference matches in an earlier stage?
        # There is already a step in the preprocessing pipeline that does some
        # string transformations on the reference matchings.
        reference_matchings = reference_matchings.assign(
            src=le_first.transform(reference_matchings["src"]),
            tgt=le_second.transform(reference_matchings["tgt"]),
        )

        # Make the edges undirected by adding reversed edges
        first_onto_edges = self.add_reversed_edges_and_shuffle(first_onto_edges)
        second_onto_edges = self.add_reversed_edges_and_shuffle(second_onto_edges)

        first_graph = dgl.graph(
            (
                torch.tensor(first_onto_edges.src.values),
                torch.tensor(first_onto_edges.tgt.values),
            )
        )
        second_graph = dgl.graph(
            (
                torch.tensor(second_onto_edges.src.values),
                torch.tensor(second_onto_edges.tgt.values),
            )
        )
        first_graph.ndata["feat"] = torch.FloatTensor(
            np.stack(first_onto_nodes["embeddings"])
        )
        second_graph.ndata["feat"] = torch.FloatTensor(
            np.stack(second_onto_nodes["embeddings"])
        )

        # Create bfs neighborhood mappings used for the negative sampling
        # during the training
        logger.info("\tCreating bfs_neighborhood_mappings")

        bfs_neighborhood_mappings = {
            "first": generate_bfs_neighborhood_mappings(
                reference_matchings.src,
                first_graph,
                first_onto_nodes[first_onto_nodes.is_virtual].id,
            ),
            "second": generate_bfs_neighborhood_mappings(
                reference_matchings.tgt,
                second_graph,
                second_onto_nodes[second_onto_nodes.is_virtual].id,
            ),
        }

        self.reference_matchings = reference_matchings
        self.bfs_neighborhood_mappings = bfs_neighborhood_mappings
        self.first_graph = first_graph
        self.second_graph = second_graph
        self.le_first = le_first
        self.le_second = le_second
        self.first_index = first_index
        self.second_index = second_index

    def __getitem__(self, idx):
        return self.g

    def __len__(self):
        return 1

    def save(self):
        # save graphs and labels
        save_graphs(f"{self.save_path}.bin", [self.first_graph, self.second_graph])
        save_info(
            f"{self.save_path}_info.pkl",
            {
                "le_first": self.le_first,
                "le_second": self.le_second,
                "bfs_neighborhood_mappings": self.bfs_neighborhood_mappings,
                "reference_matchings": self.reference_matchings,
            },
        )
        self.first_index.save(str(Path(self.save_dir, f"{self.name}_first_index.ann")))
        self.second_index.save(
            str(Path(self.save_dir, f"{self.name}_second_index.ann"))
        )

    def load(self):
        # load processed data from directory `self.save_path`
        graphs = load_graphs(f"{self.save_path}.bin")
        self.first_graph = graphs[0][0]
        self.second_graph = graphs[0][1]

        info = load_info(f"{self.save_path}_info.pkl")
        self.le_first = info["le_first"]
        self.le_second = info["le_second"]
        self.bfs_neighborhood_mappings = info["bfs_neighborhood_mappings"]
        self.reference_matchings = info["reference_matchings"]
        self.first_index = AnnoyIndex(EMBEDDING_LENGTH, "euclidean")
        self.second_index = AnnoyIndex(EMBEDDING_LENGTH, "euclidean")

        self.first_index.load(str(Path(self.save_dir, f"{self.name}_first_index.ann")))
        self.second_index.load(
            str(Path(self.save_dir, f"{self.name}_second_index.ann"))
        )

    def has_cache(self):
        # check whether there are processed data in `self.save_path`

        return os.path.exists(f"{self.save_path}.bin") and os.path.exists(
            f"{self.save_path}_info.pkl"
        )

    def transform_edges(self, edges, le):
        return edges.assign(
            src=le.transform(edges["src"]), tgt=le.transform(edges["tgt"])
        )

    def add_reversed_edges_and_shuffle(self, edges):
        """
        This method takes a numpy array of directed edges and returns a new
        array that also includes the reversed edges. The returned array is twice
        the size of the input, with the original and reversed edges
        interlaced and their order randomized.

        """
        interlaced = np.empty(
            (edges.shape[0] * 2, edges.shape[1]), dtype=edges.dtypes.src
        )
        interlaced[0::2] = edges.values
        interlaced[1::2] = np.vstack([edges.values[:, 1], edges.values[:, 0]]).T
        interlaced = interlaced.reshape((int(len(interlaced) / 2), 2, 2))
        np.random.shuffle(interlaced)
        interlaced = interlaced.reshape((len(edges) * 2, 2, -1)).squeeze(2)
        return pd.DataFrame(interlaced, columns=edges.columns)

    @staticmethod
    def load_onto(path: Path):
        """
        TODO
        """
        return [
            *[
                pd.read_parquet(str(Path(path, f"{x}.parquet")))
                for x in ["nodes", "edges", "embeddings"]
            ],
            pkl.load(open(Path(path, "label_encoder.pkl"), "rb")),
        ]

    def create_annoy_index(self, node_data: pd.DataFrame):
        t = AnnoyIndex(EMBEDDING_LENGTH, "euclidean")
        for node_id, _, _, _, embedding, _ in node_data.itertuples(index=False):
            # Add normalized vector to index
            t.add_item(node_id, embedding / np.linalg.norm(embedding))

        t.build(10)  # 10 trees
        return t

    def augmented_match_edges(self, node_data: pd.DataFrame, index_of_other_onto):
        logger.info("\tAugmenting matches")
        augmented_matches = {"src": [], "tgt": []}
        augmented_edges_per_node = 2
        for _, id, _, embedding in node_data.sample(
            int(len(node_data) * 0.2)
        ).itertuples():
            targets = index_of_other_onto.get_nns_by_vector(
                embedding / np.linalg.norm(embedding), augmented_edges_per_node
            )
            augmented_matches["src"].extend([id] * augmented_edges_per_node)
            augmented_matches["tgt"].extend(targets)

        return pd.DataFrame(augmented_matches)


def generate_bfs_neighborhood_mappings(reference_matchings, graph, nodes_to_filter):
    onto = graph.to_networkx().to_undirected()
    bfs_neighborhood_mappings = []

    # Iterate over sources of the reference matchings
    for x in tqdm.tqdm(reference_matchings):
        bfs_generator = nx.bfs_layers(onto, x)
        output = []

        # Iterate over the layers of the bfs
        for layer in bfs_generator:
            output.extend([x for x in list(layer) if x not in nodes_to_filter])

            # We use 5-10 negative samples at most, so stop if double the amount is added
            if len(output) > 20:
                bfs_neighborhood_mappings.append((x, output))
                break

    return dict(bfs_neighborhood_mappings)
