import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torchmetrics.retrieval import RetrievalHitRate, RetrievalMRR

from omunet.training.loss import compute_loss


def test(
    first_graph,
    second_graph,
    edges_positive_supervision,
    edges_negative_supervision,
    model,
    k,
    loss,
):
    with torch.no_grad():
        pos_score, neg_score, h_first, h_second = model(
            first_graph,
            second_graph,
            edges_positive_supervision,
            edges_negative_supervision,
        )
        ranking_metrics = compute_mrr_and_hits_at_1(pos_score, neg_score, k)
        classification_metrics = compute_roc_f1_precision_recall(pos_score, neg_score)
        loss = compute_loss(pos_score, neg_score, loss).item()

    return classification_metrics | ranking_metrics | {"loss": loss}, h_first, h_second


def compute_roc_f1_precision_recall(pos_score, neg_score):
    pos_score = pos_score.reshape(-1)
    neg_score = neg_score.reshape(-1)

    scores = torch.cat([pos_score, neg_score]).detach().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()

    predicted_labels = scores > 0.5
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    } | {
        metric_name: metric_func(labels, predicted_labels)
        for metric_name, metric_func in zip(
            ["f1", "precision", "recall"],
            [f1_score, precision_score, recall_score],
        )
    }


def find_optimal_threshold(target, predicted):
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame(
        {
            "tf": pd.Series(tpr - (1 - fpr), index=i),
            "threshold": pd.Series(threshold, index=i),
        }
    )
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t["threshold"])[0]


def compute_mrr_and_hits_at_1(pos_score, neg_score, k):
    neg_and_pos_score = torch.hstack(
        [pos_score, neg_score.reshape(-1).reshape(pos_score.shape[0], k)]
    ).reshape(-1)
    indexes = torch.arange(pos_score.shape[0]).repeat_interleave(k + 1)
    target = torch.tensor([True, *[False] * k]).repeat(pos_score.shape).reshape(-1)

    hits_at_1 = RetrievalHitRate(top_k=1)(
        neg_and_pos_score, target, indexes=indexes
    ).item()
    mrr = RetrievalMRR()(neg_and_pos_score, target, indexes=indexes).item()

    return {"mrr": mrr, "h1": hits_at_1}
