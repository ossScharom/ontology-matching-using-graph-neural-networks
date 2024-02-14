from datetime import datetime
from pathlib import Path
from random import random, seed
from typing import Tuple

import dgl
import numpy as np
import pandas as pd
import torch
from omunet.shared.graph_dataset import OmunetGraphDataset
from omunet.shared.util import OntoMatchingPairs
from omunet.training.construct_negative_graph import get_negative_supervision_edges
from omunet.training.evaluation import (
    compute_mrr_and_hits_at_1,
    compute_roc_f1_precision_recall,
    test,
)
from omunet.training.loss import compute_loss
from omunet.training.model import Model
from omunet.training.save_results import (
    save_current_node_embeddings,
    save_plots,
    save_test_result,
    save_train_val_result,
)
from omunet.training.utils import create_random_splits


def omunet_train(
    task: OntoMatchingPairs,
    processed_dgl_graphs: Path,
    result_history_path: Path,
    model_history: Path,
    experiment_config: dict,
    random_seed: int = None,
    save_test_evaluation: bool = True,
    save_train_plots: bool = True,
):
    if not random_seed:
        random_seed = int(random())

    seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    dgl.seed(random_seed)

    g = OmunetGraphDataset(
        task.value,
        processed_dgl_graphs,
    )

    # unpack experiment configs located in config/dev.conf
    amount_negative_edges = experiment_config["amount_negative_edges"]
    negative_sampling_type = experiment_config["negative_sampling_type"]
    loss_type = experiment_config["loss_type"]
    amount_convolutions = experiment_config["amount_convolutions"]
    input_dimension = experiment_config["input_dimension"]
    hidden_dimension = experiment_config["hidden_dimension"]
    output_dimension = experiment_config["output_dimension"]
    epochs = experiment_config["epochs"]
    learning_rate = experiment_config["learning_rate"]
    experiment_name = experiment_config["experiment_name"]

    path = Path(model_history, experiment_name)
    model_path = Path(path, f"{task.name}_best_model.pkl")
    embedding_path = Path(path, f"{task.name}_node_embeddings.pkl")
    path.mkdir(parents=True, exist_ok=True)

    (
        edges_train_positive_supervision,
        edges_val_positive_supervision,
        edges_test_positive_supervision,
    ) = create_random_splits(g.reference_matchings)

    # Supervision edges need to be duplicated and reversed, because the out put of the
    # negative candidate generation differs depending on the direction of the
    # given true matching edge
    # Splitting is done here, to avoid doing this for every epoch
    (
        edges_train_positive_supervision_src_tgt,
        edges_train_positive_supervision_tgt_src,
    ) = duplicate_and_reverse(edges_train_positive_supervision)
    (
        edges_val_positive_supervision_src_tgt,
        edges_val_positive_supervision_tgt_src,
    ) = duplicate_and_reverse(edges_val_positive_supervision)
    (
        edges_test_positive_supervision_src_tgt,
        edges_test_positive_supervision_tgt_src,
    ) = duplicate_and_reverse(edges_test_positive_supervision)

    # duplicate and convert pd.DataFrame to tuple of node lists
    edges_train_positive_supervision = convert_df_to_edges_lists(
        pd.concat([edges_train_positive_supervision, edges_train_positive_supervision])
    )
    edges_val_positive_supervision = convert_df_to_edges_lists(
        pd.concat([edges_val_positive_supervision, edges_val_positive_supervision])
    )
    edges_test_positive_supervision = convert_df_to_edges_lists(
        pd.concat([edges_test_positive_supervision, edges_test_positive_supervision])
    )

    model = Model(
        input_dimension,
        hidden_dimension,
        output_dimension,
        amount_convolutions,
    )

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_val_history = {
        "train": [],
        "val": [],
    }
    epoch_loop(
        task,
        g,
        amount_negative_edges,
        negative_sampling_type,
        loss_type,
        epochs,
        model_path,
        embedding_path,
        edges_train_positive_supervision,
        edges_val_positive_supervision,
        edges_train_positive_supervision_src_tgt,
        edges_train_positive_supervision_tgt_src,
        edges_val_positive_supervision_src_tgt,
        edges_val_positive_supervision_tgt_src,
        model,
        opt,
        train_val_history,
    )

    path = Path(
        str(result_history_path),
        str(random_seed),
        experiment_name,
        f"{task.name}_{datetime.now().isoformat()}",
    )
    save_train_val_result(train_val_history, path)

    if save_train_plots:
        save_plots(train_val_history, path)

    if save_test_evaluation:
        edges_test_negative_supervision = create_negative_edges(
            negative_sampling_type,
            edges_test_positive_supervision_src_tgt,
            edges_test_positive_supervision_tgt_src,
            amount_negative_edges,
            g,
        )

        test_metrics, _, _ = test(
            g.first_graph,
            g.second_graph,
            edges_test_positive_supervision,
            edges_test_negative_supervision,
            model,
            amount_negative_edges,
            loss_type,
        )
        save_test_result(test_metrics, path)


def epoch_loop(
    task,
    g,
    amount_negative_edges,
    negative_sampling_type,
    loss_type,
    epochs,
    model_path,
    embedding_path,
    edges_train_positive_supervision,
    edges_val_positive_supervision,
    edges_train_positive_supervision_src_tgt,
    edges_train_positive_supervision_tgt_src,
    edges_val_positive_supervision_src_tgt,
    edges_val_positive_supervision_tgt_src,
    model,
    opt,
    train_val_history,
    best_loss=10000,
):
    for epoch in range(epochs):
        print("Epoch: ", epoch)

        edges_train_negative_supervision = create_negative_edges(
            negative_sampling_type,
            edges_train_positive_supervision_src_tgt,
            edges_train_positive_supervision_tgt_src,
            amount_negative_edges,
            g,
        )
        pos_score, neg_score, _, _ = model(
            g.first_graph,
            g.second_graph,
            edges_train_positive_supervision,
            edges_train_negative_supervision,
        )

        train_loss = compute_loss(pos_score, neg_score, loss_type)

        opt.zero_grad()
        train_loss.backward()
        opt.step()

        with torch.no_grad():
            classification_metrics = compute_roc_f1_precision_recall(
                pos_score, neg_score
            )
            ranking_metrics = compute_mrr_and_hits_at_1(
                pos_score, neg_score, amount_negative_edges
            )
            train_metrics = (
                classification_metrics | ranking_metrics | {"loss": train_loss.item()}
            )
        print_eval_scores("Train", train_metrics)
        print()

        edges_val_negative_supervision = create_negative_edges(
            negative_sampling_type,
            edges_val_positive_supervision_src_tgt,
            edges_val_positive_supervision_tgt_src,
            amount_negative_edges,
            g,
        )

        validation_metrics, h_first, h_second = test(
            g.first_graph,
            g.second_graph,
            edges_val_positive_supervision,
            edges_val_negative_supervision,
            model,
            amount_negative_edges,
            loss_type,
        )
        print_eval_scores("Validation", validation_metrics)
        print()

        train_val_history["train"].append(train_metrics)
        train_val_history["val"].append(validation_metrics)

        #  Is checkpoint epoch and is current loss better than best loss?
        if epoch != 0 and epoch % 20 == 0 and validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": validation_metrics["loss"],
                },
                model_path,
            )
            save_current_node_embeddings(task, embedding_path, h_first, h_second)


def print_eval_scores(split_name, metrics):
    print(split_name)
    for metric_name, value in metrics.items():
        print("  ", f"{metric_name} : {value:.2f}")


def duplicate_and_reverse(reference_matchings):
    return (
        convert_df_to_edges_lists(reference_matchings),
        # reverse second half by swapping column names
        convert_df_to_edges_lists(
            reference_matchings.rename({"src": "tgt", "tgt": "src"}, axis=1)
        ),
    )


def create_negative_edges(
    negative_sampling_type,
    edges_positive_supervision_src_tgt,
    edges_positive_supervision_tgt_src,
    k,
    g,
):
    # non-reversed half of supervised edges point from second onto to first
    # and sample the new negative tails from the first ontology
    edges_negative_supervision_src_tgt = get_negative_supervision_edges(
        negative_sampling_type,
        edges_positive_supervision_src_tgt,
        g.second_graph.num_nodes(),
        g.first_graph.ndata["feat"],
        k,
        g.bfs_neighborhood_mappings["second"],
        g.second_index,
    )

    # reversed half of supervised edges point from second onto to
    # first and sample the new negative tails from the first ontology
    edges_negative_supervision_tgt_src = get_negative_supervision_edges(
        negative_sampling_type,
        edges_positive_supervision_tgt_src,
        g.first_graph.num_nodes(),
        g.second_graph.ndata["feat"],
        k,
        g.bfs_neighborhood_mappings["first"],
        g.first_index,
    )

    # The negative edge generation is completed, negative edges can be combined again,
    # but the reversed half (target to source) needs to be reversed again (source to target)
    edges_negative_supervision_src_tgt_second_half = tuple(
        reversed(edges_negative_supervision_tgt_src)
    )
    combined_src_tgt = tuple(
        torch.concat(a)
        for a in list(
            zip(
                edges_negative_supervision_src_tgt,
                edges_negative_supervision_src_tgt_second_half,
            )
        )
    )

    return combined_src_tgt


def convert_df_to_edges_lists(df: pd.DataFrame) -> Tuple:
    if all(~df.columns.isin(["src", "tgt"])):
        raise ValueError("")
    return (df.src.values, df.tgt.values)
