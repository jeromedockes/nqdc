"""Extracting all URLs from an article XML."""
from lxml import etree
import pandas as pd

from nqdc._typing import Extractor
from nqdc import _utils


class LinkExtractor(Extractor):
    """Extracting all URLs in an article.

    Currently only ext-link elements are considered, but this may change in the
    future.
    """

    fields = ("pmcid", "ext-link-type", "href")
    name = "links"

    def extract(self, article: etree.ElementTree) -> pd.DataFrame:
        pmcid = _utils.get_pmcid(article)
        all_links = []
        xlink = "http://www.w3.org/1999/xlink"
        for link in article.iterfind(f"//ext-link[@{{{xlink}}}href]"):
            href = link.get(f"{{{xlink}}}href")
            link_type = link.get("ext-link-type")
            all_links.append(
                {"pmcid": pmcid, "ext-link-type": link_type, "href": href}
            )
        return pd.DataFrame(all_links, columns=self.fields)
