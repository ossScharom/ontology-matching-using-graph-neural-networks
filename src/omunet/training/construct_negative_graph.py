from itertools import chain
from random import sample

import numpy as np
import torch


def get_negative_supervision_edges(
    negative_sampling_type,
    g_supervision_positive,
    number_of_nodes,
    language_embeddings,
    k,
    bfs_neighborhood_mappings,
    index,
):
    if negative_sampling_type == "structural":
        return construct_negative_graph_structural_based(
            g_supervision_positive,
            k,
            bfs_neighborhood_mappings,
        )
    if negative_sampling_type == "similarity":
        return construct_negative_graph_similarity_based(
            g_supervision_positive,
            k,
            index,
            language_embeddings,
        )

    if negative_sampling_type == "uniform":
        return construct_negative_graph_uniform_sampling_based(
            g_supervision_positive,
            k,
            # The number of nodes of the destination ontology
            number_of_nodes,
        )


def construct_negative_graph_uniform_sampling_based(
    g_supervision_positive,
    k,
    number_of_nodes,
):
    src, _ = g_supervision_positive
    neg_src = torch.tensor(src).repeat_interleave(k)
    neg_tgt = torch.randint(0, number_of_nodes, (len(src) * k,))

    return (neg_src, neg_tgt)


def construct_negative_graph_structural_based(
    g_supervision_positive,
    k,
    bfs_mappings,
):
    src, tgt = g_supervision_positive
    neg_src = torch.tensor(src).repeat_interleave(k)

    neg_tgt = torch.LongTensor(
        list(
            chain(
                *[sample(bfs_mappings[single_tgt.item()][1:], k) for single_tgt in tgt]
            )
        )
    )

    return (neg_src, neg_tgt)


def construct_negative_graph_similarity_based(
    g_supervision_positive,
    k,
    index,
    language_embeddings,
):
    src, tgt = g_supervision_positive
    neg_src = torch.tensor(src).repeat_interleave(k)

    neg_tgt = []

    for src_node, dst_node in zip(src, tgt):
        # If node is from the first onto, then, the onto data value is set to 0, take
        # the index of the second onto

        src_embedding = language_embeddings[src_node]

        # Normalize embedding
        src_embedding = src_embedding / np.linalg.norm(src_embedding)
        # Need to sample k+1 nodes, because the original destination node can be in the results
        neg_dst_candidates = index.get_nns_by_vector(src_embedding, k + 1)

        # Is original destination node in the candidates, then remove
        if dst_node.item() in neg_dst_candidates:
            neg_dst_candidates.remove(dst_node.item())
        # If not, remove the last candidate, to get k nodes
        else:
            neg_dst_candidates = neg_dst_candidates[:-1]

        neg_tgt.extend(neg_dst_candidates)

    neg_tgt = torch.LongTensor(neg_tgt)

    return (neg_src, neg_tgt)
