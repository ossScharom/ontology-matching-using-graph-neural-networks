import logging
from pathlib import Path

import pandas as pd
from owlready2 import World
from owlready2.class_construct import And, Or
from owlready2.entity import ThingClass

logger = logging.getLogger("omunet")


def convert_owl(
    raw_path: Path,
    processed_path: Path,
):
    """
    Converts the ontologies in the owl format to a columnar format
    Args:
        raw_mondo_equiv_match: The path where the raw mondo ontologies of the
            equivalence matching task are saved
        raw_umls_equiv_match: The path where the raw umls ontologies of the
            equivalence matching task are saved
        processed_mondo: The path where the processed mondo ontologies and the
            reference matchings will be saved
        processed_umls: The path where the processed umls ontologies and the
            reference matchings will be saved
        processed_path: The path of the directory that will hold the all the
            processed files
    """

    for task_path in Path(raw_path, "bio-ml").iterdir():
        logger.info("\tStart to convert ontologies of task %s", task_path.name)
        convert_owls(task_path, processed_path)
        logger.info("\tFinished converting ontologies %s", task_path.name)


def convert_owls(task_path: Path, path_processed_owl: Path) -> None:
    """
    This function converts multiple owl files into parquet files. The owl files
    present a hierarchical data structure where as the the parquet file is a flat
    tabular data file, representing the owl graph as a list of nodes and an edge
    list.
    Args:
        path_raw_owl_equiv_match: The path of the global config, where the raw,
            unconverted ontologies are saved. We are only interested in equivalence
            matching task.
        path_processed_owl: The path where the processed ontologies should be
            saved


    """
    for owl_file in task_path.glob("*.owl"):
        logger.info("\t\tStart converting: %s", owl_file.name)
        convert_single_owl(owl_file, path_processed_owl)


def convert_single_owl(owl_file: Path, path_processed_owl: Path) -> None:
    """
    This function converts multiple owl files into parquet files. The owl files
    present a hierarchical data structure where as the the parquet file is a flat
    tabular data file, representing the owl graph as a list of nodes and an edge
    list.

    Args:
        owl_file: The path where a single, unconverted ontologies is saved
        path_processed_owl: The path where the processed ontologies should be
            saved
    """

    # Conversion from hierarchical owl tree to nodes and edge list
    new_dir = Path(path_processed_owl, owl_file.stem)
    nodes_path = Path(new_dir, "nodes.parquet")
    edges_path = Path(new_dir, "edges.parquet")

    if Path.exists(nodes_path) and Path.exists(edges_path):
        logger.info("\t\t%s Datasets are already converted", owl_file.stem)
        return

    onto = World().get_ontology(str(owl_file)).load()
    nodes = pd.DataFrame(
        [
            {
                "id": x.name,
                "labels": x.label,
                "is_a": expand_logical_operators(x.is_a),
            }
            for x in onto.classes()
        ]
    )

    use_in_alignment_ann_prop = [
        x for x in onto.annotation_properties() if x.name == "use_in_alignment"
    ][0]

    # Only the Concepts that have this annotation property. It is always set to False,
    # e.g. do not use in alignment
    do_not_use_in_alignment = list(
        [x[0].name for x in use_in_alignment_ann_prop.get_relations()]
    )

    # negate truth series
    nodes["use_in_alignment"] = ~nodes.id.isin(do_not_use_in_alignment)

    edges = (
        nodes.drop(["labels", "use_in_alignment"], axis=1)
        .explode("is_a")
        .rename(columns={"id": "src", "is_a": "tgt"})
        .dropna()
    )

    nodes = nodes.drop("is_a", axis=1)

    Path.mkdir(new_dir, parents=True, exist_ok=True)

    nodes.to_parquet(nodes_path)
    edges.to_parquet(edges_path)


def expand_logical_operators(is_a):
    parent_concepts = []
    for potential_parent in set(is_a):
        if isinstance(potential_parent, And) or isinstance(potential_parent, Or):
            parent_concepts.extend(
                [
                    x.name
                    for x in potential_parent.get_Classes()
                    if isinstance(x, ThingClass)
                ]
            )
        elif not isinstance(potential_parent, ThingClass):
            continue
        else:
            parent_concepts.append(potential_parent.name)

    return list(set(parent_concepts) - {"Thing"})
