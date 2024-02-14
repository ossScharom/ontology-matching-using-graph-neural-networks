import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("omunet")


def convert_reference_matchings(raw: Path, processed_reference_matches: Path):
    """
    Converts the reference mappings of all the given ontologies
    """

    for path_raw_owl_equiv_match in Path(raw, "bio-ml").iterdir():
        path_raw_reference_matching_file = Path(
            path_raw_owl_equiv_match, "refs_equiv", "full.tsv"
        )
        path_processed_reference_matching_file = Path(
            processed_reference_matches,
            f"{path_raw_reference_matching_file.parent.parent.name}.parquet",
        )

        if path_processed_reference_matching_file.exists():
            logger.info(
                "\tReference matchings for %s are already converted",
                path_raw_reference_matching_file.parent.parent.name,
            )
            continue

        processed_reference_matches.mkdir(exist_ok=True)

        convert_single_reference_matching(
            path_raw_reference_matching_file,
            path_processed_reference_matching_file,
        )


def convert_single_reference_matching(
    path_raw_reference_matching_file: Path, path_processed_reference_matching_file: Path
):
    """
    TODO
    """

    df = pd.read_csv(
        path_raw_reference_matching_file, usecols=["SrcEntity", "TgtEntity"], sep="\t"
    )
    df = df.rename(columns={"SrcEntity": "src", "TgtEntity": "tgt"})
    df["src"] = df["src"].apply(
        lambda x: Path(x).name.split("#")[1] if "#" in Path(x).name else Path(x).name
    )
    df["tgt"] = df["tgt"].apply(
        lambda x: Path(x).name.split("#")[1] if "#" in Path(x).name else Path(x).name
    )

    logger.info(
        "\tConverted: %s %s",
        path_raw_reference_matching_file.parent.parent.name,
        path_raw_reference_matching_file.stem,
    )

    df.to_parquet(str(path_processed_reference_matching_file))
