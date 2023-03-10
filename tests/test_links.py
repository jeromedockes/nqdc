import pathlib

import pytest
from lxml import etree
import pandas as pd

from pubget import _links


@pytest.mark.parametrize("n_links", [0, 1, 5])
def test_link_extractor(n_links):
    xml_template = b"""<?xml version='1.0' encoding='ASCII'?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
    <front>
        <article-meta>
            <article-id pub-id-type="pmc">9057060</article-id>
        </article-meta>
    </front>
    <body>
    some links:
    %s
    </body>
</article>
    """
    one_link = (
        b'link <ext-link xlink:href="http:example.com/%d">'
        b"http:example.com</ext-link>"
    )
    links_text = b"\n".join([one_link % i for i in range(n_links)])
    xml = xml_template % links_text
    document = etree.ElementTree(etree.XML(xml))
    extracted = _links.LinkExtractor().extract(
        document, pathlib.Path("pmc_9057060"), {}
    )
    assert extracted.shape == (n_links, 3)


def test_link_content_extractor():
    links = pd.DataFrame(
        {
            "href": [
                "https://neurovault.org/collections/12/",
                "identifiers.org/neurovault.collection:a13",
                "neurovault.org/collections/14a",
                "https://neurovault.org/images/3",
                "identifiers.org/neurovault.image:2",
            ],
            "pmcid": 7,
        }
    )
    data = {"links": links}
    col_extract, img_extract = _links.neurovault_id_extractors()
    col = col_extract.extract(None, None, data)
    assert (
        (
            col
            == pd.DataFrame(
                {"pmcid": 7, "collection_id": ["12", "a13", "14a"]}
            )
        )
        .all()
        .all()
    )
    img = img_extract.extract(None, None, data)
    assert (
        (img == pd.DataFrame({"pmcid": 7, "image_id": ["3", "2"]})).all().all()
    )
    col = col_extract.extract(None, None, pd.DataFrame())
    assert col.shape == (0, 2)
