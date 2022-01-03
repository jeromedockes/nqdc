from pathlib import Path
import logging

from lxml import etree
import pandas as pd

from nqdc._coordinates import CoordinateExtractor
from nqdc._metadata import MetadataExtractor
from nqdc._text import TextExtractor


_LOG = logging.getLogger(__name__)


def extract_data(articles_dir):
    articles_dir = Path(articles_dir)
    all_coords = []
    all_metadata = []
    all_text = []
    n_articles, n_with_coords = 0, 0
    coord_extractor = CoordinateExtractor()
    metadata_extractor = MetadataExtractor()
    text_extractor = TextExtractor()
    for subdir in sorted([f for f in articles_dir.glob("*") if f.is_dir()]):
        try:
            subdir_nb = int(subdir.name, 16)
        except ValueError:
            continue
        _LOG.info(f"Processing directory: {subdir.name}")
        for article_file in subdir.glob("pmcid_*.xml"):
            n_articles += 1
            coords_found = False
            _LOG.debug(
                f"In directory {subdir.name} "
                f"({subdir_nb / 0xfff:.0%}), "
                f"processing article: {article_file.name}"
            )
            try:
                article = etree.parse(str(article_file))
            except Exception:
                _LOG.exception(f"Failed to parse {article_file}")
                continue
            metadata = metadata_extractor(article)
            all_metadata.append(metadata)
            text = text_extractor(article)
            text["pmcid"] = metadata["pmcid"]
            all_text.append(text)
            coords = coord_extractor(article)
            if coords.shape[0]:
                coords["pmcid"] = metadata["pmcid"]
                all_coords.append(coords)
                coords_found = True
            n_with_coords += coords_found
            _LOG.info(
                f"Processed in total {n_articles} articles, {n_with_coords} "
                f"({n_with_coords / n_articles:.0%}) had coordinates"
            )
    return {
        "coordinates": pd.concat(all_coords),
        "metadata": pd.DataFrame(all_metadata),
    }