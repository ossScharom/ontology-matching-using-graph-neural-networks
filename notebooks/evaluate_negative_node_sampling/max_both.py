# %%
L = {
    ("a1", "b1"),
    ("a2", "b2"),
    ("a2", "b3"),
    ("a3", "b4"),
    ("a3", "b5"),
    ("a4", "b5"),
    ("a4", "b6"),
    ("a5", "b7"),
    ("a6", "b7"),
}

sims = {
    ("a1", "b1"): 1,
    ("a2", "b2"): 0.8,
    ("a2", "b3"): 0.9,
    ("a3", "b4"): 0.85,
    ("a3", "b5"): 0.9,
    ("a4", "b5"): 0.8,
    ("a4", "b6"): 0.7,
    ("a5", "b7"): 0.75,
    ("a6", "b7"): 0.7,
}


def sim(a, b):
    # Replace this with your actual similarity calculation logic
    return sims[(a, b)]


def get_best_links_for_partition(L, reversed=False):
    if reversed:
        L = {(k[1], k[0]) for k in L}

    L_MAX1 = set()
    for a in {record[0] for record in L}:
        best_link = None
        best_similarity = 0.0

        for b in {record[1] for record in L if record[0] == a}:
            similarity = sim(b, a) if reversed else sim(a, b)
            if similarity > best_similarity:
                best_similarity = similarity
                best_link = (b, a) if reversed else (a, b)

        if best_link is not None:
            L_MAX1.add(best_link)
    return L_MAX1


def max_both():
    return get_best_links_for_partition(L).intersection(
        get_best_links_for_partition(L, reversed=True)
    )


max_both()
# %%
import pandas as pd

candidates_with_similarities = pd.DataFrame(
    {
        "src": ["a1", "a2", "a2", "a3", "a3", "a4", "a4", "a5", "a6"],
        "tgt": ["b1", "b2", "b3", "b4", "b5", "b5", "b6", "b7", "b7"],
        "sims": [1, 0.8, 0.9, 0.85, 0.9, 0.8, 0.7, 0.75, 0.7],
    }
)


def get_best_links_for_partition(candidates_with_similarities, swapped=False):
    L_MAX1 = set()
    for a in candidates_with_similarities.src.drop_duplicates():
        tmp = candidates_with_similarities[candidates_with_similarities["src"] == a]
        if any(tmp.tgt.duplicated()):
            print("duplicates")
        best_link = candidates_with_similarities.iloc[tmp.sims.idxmax()]

        if not best_link.empty:
            best = (best_link.src, best_link.tgt)
            if swapped:
                best = tuple(reversed(best))
            L_MAX1.add(best)

    return L_MAX1


def max_both(candidates_with_similarities):
    return get_best_links_for_partition(candidates_with_similarities).intersection(
        get_best_links_for_partition(
            candidates_with_similarities.rename({"src": "tgt", "tgt": "src"}, axis=1),
            swapped=True,
        )
    )


max_both(candidates_with_similarities)
# %%
