import logging
import zipfile
from collections.abc import Iterable
from pathlib import Path
from urllib import request

logger = logging.getLogger("omunet")


def filter_hidden_files(file_list: Iterable[Path]) -> list[Path]:
    """
    This utility function filters all dictionaries and files that start with a
    single dot. Files starting with a single dot are hidden files in unix
    operating systems.
    Args:
        file_list: The list of path containing files that should be filtered
    """
    return [x for x in file_list if not x.name.startswith(".")]


def download_data(dataset_url: str, downloads_path: Path) -> None:
    """
    This function retrieves the zipped ontologies form the given dataset_urls
    and places them into the downloads_path.
    Args:
        dataset_urls: The dictionary of urls that specifies the location of
            the datasets in the web
        downloads_path: The path where the downloaded zip files should be saved
        to on the disk

    """
    downloads_path.mkdir(parents=True, exist_ok=True)

    if len(list(downloads_path.iterdir())) > 0:
        logger.info("\tDatasets are already downloaded")
        return

    request.urlretrieve(
        dataset_url,
        str(Path(downloads_path, "raw-dataset").with_suffix(".zip")),
    )


def extract_data(downloads_path: Path, raw_path: Path):
    """
    Extracts the data from a zip within the downloads_path to a folder under the
    give raw_path
    Args:
        downloads_path: The path where the data was downloaded
        raw_path: The path where data should be extracted to
    """
    if Path.exists(raw_path) and len(list(raw_path.iterdir())) > 0:
        logger.info("\tDatasets are already extracted")
        return

    if not Path.exists(raw_path):
        Path.mkdir(raw_path)

    for file_path in filter_hidden_files(downloads_path.iterdir()):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(str(raw_path))

    # delete macosx system files
    for ds_file in [child for child in raw_path.glob("**/.DS*")]:
        ds_file.unlink()
