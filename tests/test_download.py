import json
from unittest.mock import Mock

from nqdc import _download


def test_download_articles_for_query(tmp_path, entrez_mock, monkeypatch):
    entrez_mock.fail_efetch = True
    download_dir, code = _download.download_articles_for_query(
        "fMRI[abstract]", tmp_path, retmax=3
    )
    assert code == 1
    assert download_dir == tmp_path.joinpath(
        "query-7838640309244685021f9954f8aa25fc", "articlesets"
    )
    assert download_dir.joinpath("batch_00000.xml").is_file()
    assert not download_dir.joinpath("batch_00001.xml").is_file()
    assert not json.loads(
        download_dir.joinpath("info.json").read_text("utf-8")
    )["download_complete"]

    entrez_mock.fail_efetch = False
    download_dir, code = _download.download_articles_for_query(
        "fMRI[abstract]", tmp_path, retmax=3
    )
    assert code == 0
    assert download_dir.joinpath("batch_00001.xml").is_file()
    assert download_dir.joinpath("batch_00002.xml").is_file()
    assert json.loads(download_dir.joinpath("info.json").read_text("utf-8"))[
        "download_complete"
    ]
    mock = Mock()
    monkeypatch.setattr(_download, "EntrezClient", mock)
    download_dir, code = _download.download_articles_for_query(
        "fMRI[abstract]", tmp_path, retmax=3
    )
    assert code == 0
    assert not mock.called