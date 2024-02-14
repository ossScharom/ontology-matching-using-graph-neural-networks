import logging
from pathlib import Path

import typer

import omunet.create_candidates
import omunet.evaluate
from omunet import __title__, __version__
from omunet.pre_processing.add_virtual_nodes import add_virtual_nodes
from omunet.pre_processing.convert_owl import convert_owl
from omunet.pre_processing.convert_reference_matchings import (
    convert_reference_matchings,
)
from omunet.pre_processing.create_graph_datasets import create_graph_datasets

# from omunet.create_node_embeddings import create_node_embeddings
# from omunet.pre_processing.create_graph_datasets import create_graph_datasets
from omunet.pre_processing.create_node_embeddings import create_node_embeddings
from omunet.pre_processing.create_node_ids import create_node_ids_and_label_encoder
from omunet.pre_processing.create_token_indexes import create_token_indexes
from omunet.pre_processing.download_data import download_data, extract_data
from omunet.shared import util
from omunet.training.train import omunet_train

logger = logging.getLogger("omunet")

app = typer.Typer(
    name="omunet",
    help="A short summary of the project",
)


def version_callback(version: bool):
    if version:
        typer.echo(f"{__title__} {__version__}")
        raise typer.Exit()


ConfigOption = typer.Option(
    None, "-c", "--config", metavar="PATH", help="path to the program configuration"
)
VersionOption = typer.Option(
    None,
    "-v",
    "--version",
    callback=version_callback,
    is_eager=True,
    help="print the program version and exit",
)


@app.command("train")
def train(
    task: util.OntoMatchingPairs,
    config_file: str = ConfigOption,
    random_seed: int = typer.Option(42, "--random-seed", "-r"),
):
    config = util.load_config(config_file)
    util.logging_setup(config)
    omunet_train(
        task,
        config.get("processed_dgl_graphs"),
        config.get("result_history_path"),
        config.get("model_history_path"),
        config.get("experiment_config"),
        random_seed,
        config.get("save_test_evaluation"),
        config.get("save_train_plots"),
    )


@app.command("create-candidates")
def create_candidates(
    task: util.OntoMatchingPairs,
    config_file: str = ConfigOption,
):
    config = util.load_config(config_file)
    util.logging_setup(config)
    task_config = config.get("ontology_matching_pairs")[task.value]
    first_onto_name = task_config["first_onto"]
    second_onto_name = task_config["second_onto"]

    omunet.create_candidates.create_candidates(
        task,
        first_onto_name,
        second_onto_name,
        Path(config.get("processed")),
        config.get("candidates_per_node"),
    )


@app.command("evaluate")
def evaluate(
    task: util.OntoMatchingPairs,
    config_file: str = ConfigOption,
):
    config = util.load_config(config_file)
    util.logging_setup(config)

    # Add sub experiment folder
    model_history_path = Path(
        config.get("model_history_path"),
        config.get("experiment_config")["experiment_name"],
    )
    omunet.evaluate.evaluate(
        task,
        Path(config.get("processed")),
        Path(config.get("processed_dgl_graphs")),
        model_history_path,
    )


@app.command("preprocess")
def preprocess(config_file: str = ConfigOption, version: bool = VersionOption):
    if version:
        logger.debug("Started preprocessing OMUNET %s", __version__)

    config = util.load_config(config_file)
    util.logging_setup(config)
    logger.info("Config is loaded")

    logger.info("Downloading data")
    download_data(config.get("dataset_url"), Path(config.get("downloads")))

    logger.info("Extracting data")
    extract_data(Path(config.get("downloads")), Path(config.get("raw")))

    logger.info("Converting owls")
    convert_owl(
        Path(config.get("raw")),
        Path(config.get("processed")),
    )

    logger.info("Creating label encoder and node ids")
    create_node_ids_and_label_encoder(Path(config.get("processed")))

    logger.info("Converting reference mappings")
    convert_reference_matchings(
        Path(config.get("raw")),
        Path(config.get("processed"), "reference_matches"),
    )

    logger.info("Creating embeddings for node labels")
    create_node_embeddings(Path(config.get("processed")))

    logger.info("Creating token indexes for candidate generation")
    create_token_indexes(Path(config.get("processed")), Path(config.get("token_index")))

    logger.info("Creating virtual nodes using the token indexes")
    add_virtual_nodes(Path(config.get("processed")), Path(config.get("token_index")))

    logger.info("Creating graph datasets for matching pairs")
    create_graph_datasets(
        Path(config.get("processed")),
        Path(config.get("processed_dgl_graphs")),
        config.get("ontology_matching_pairs"),
        config.get("use_virtual_nodes"),
    )
    logger.info("Preprocessing finished")


if __name__ == "__main__":
    app()
