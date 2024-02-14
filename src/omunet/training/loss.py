import torch


def compute_loss(pos_score, neg_score, loss):
    # Margin loss
    # TODO: disjunkheits axiome in den loss mit reinrechnen, kein plan wie :o
    match loss:
        case "margin":
            n_edges = pos_score.shape[0]
            return (0.5 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()
        case "bce":
            bce = torch.nn.BCEWithLogitsLoss()
            targets = torch.concat(
                [
                    torch.ones_like(pos_score).squeeze(),
                    torch.zeros_like(neg_score).squeeze(),
                ]
            )
            return bce(torch.concat([pos_score, neg_score]).squeeze(), targets)
        case "bpr":
            distances = pos_score - neg_score.view(pos_score.shape[0], -1)
            return -torch.sum(torch.nn.LogSigmoid()(distances.reshape(-1)))
