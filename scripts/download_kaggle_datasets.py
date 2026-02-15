from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

from configs import configure_logging
from configs.setting import get_settings

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

DEFAULT_OUTPUT_ROOT: Final[Path] = Path("datasets")


def get_kaggle_api():
    import os

    username = settings.dataset.kaggle_username
    key = settings.dataset.kaggle_key.get_secret_value()

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    import kaggle

    return kaggle.KaggleApi()


def download_kaggle_dataset(
    dataset_ref: str,
    output_root: Path | str,
    *,
    unzip: bool = True,
) -> Path:
    """Download a Kaggle dataset to the given directory.

    Args:
        dataset_ref: Dataset identifier as "owner/dataset-slug", e.g. "evgeniumakov/images4k".
        output_root: Root directory under which the dataset will be placed.
        unzip: If True, extract the downloaded zip and remove it.

    Returns:
        Resolved path to the output directory.
    """
    root = Path(output_root).resolve()
    dataset_name = dataset_ref.split("/")[-1]
    out = root / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    api = get_kaggle_api()
    logger.info("Downloading dataset %s to %s", dataset_ref, out)
    api.dataset_download_files(
        dataset=dataset_ref,
        path=str(out),
        unzip=unzip,
    )

    children = [child for child in out.iterdir() if child.is_dir()]
    if len(children) == 1 and children[0].name.lower() == out.name.lower():
        nested = children[0]
        for item in nested.iterdir():
            item.rename(out / item.name)
        nested.rmdir()

    logger.info("Done: %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset by owner/slug.",
    )
    parser.add_argument(
        "dataset",
        metavar="OWNER/SLUG",
        help="Dataset ref, e.g. evgeniumakov/images4k",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for datasets (default: %(default)s)",
    )
    parser.add_argument(
        "--no-unzip",
        action="store_true",
        help="Keep the downloaded zip without extracting",
    )
    args = parser.parse_args()

    download_kaggle_dataset(
        args.dataset,
        args.output_root,
        unzip=not args.no_unzip,
    )


if __name__ == "__main__":
    main()
