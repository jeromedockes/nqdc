"""'fit_neuroquery' step: fit a neuroquery.NeuroQueryModel"""
from pathlib import Path
import logging
import argparse
import json
from typing import Mapping, Tuple, Optional, Sequence, Dict

import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import normalize

from nibabel import Nifti1Image
from neuroquery import img_utils
from neuroquery.smoothed_regression import SmoothedRegression
from neuroquery.tokenization import TextVectorizer
from neuroquery.encoding import NeuroQueryModel

from nqdc._typing import PathLikeOrStr, BaseProcessingStep, ArgparseActions
from nqdc import _utils


_LOG = logging.getLogger(__name__)
_STEP_NAME = "fit_neuroquery"
_STEP_DESCRIPTION = "Fit a NeuroQuery encoder on the downloaded data."
_HELP = (
    "Fit a NeuroQuery encoder on the downloaded data. "
    "Note this can be a more computationally intensive step for "
    "large datasets. Moreover, it will not yield "
    "good results for small datasets (less than ~10K articles with "
    "coordinates). See details about neuroquery at neuroquery.org and "
    "https://github.com/neuroquery/neuroquery ."
)

_MIN_DOCUMENT_FREQUENCY = 10


def _load_tfidf_for_frequent_terms(
    tfidf_dir: Path,
) -> Tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Load TFIDF and vocabulary, keeping only terms that appear > 10 times."""
    tfidf = sparse.load_npz(str(tfidf_dir.joinpath("merged_tfidf.npz")))
    feature_names = pd.read_csv(
        tfidf_dir.joinpath("feature_names.csv"), header=None
    )
    full_voc = pd.read_csv(tfidf_dir.joinpath("vocabulary.csv"), header=None)
    voc_mapping = json.loads(
        tfidf_dir.joinpath(
            "vocabulary.csv_voc_mapping_identity.json"
        ).read_text("utf-8")
    )
    kept = np.asarray(
        (tfidf > 0).sum(axis=0) > _MIN_DOCUMENT_FREQUENCY
    ).ravel()
    new_tfidf = tfidf[:, kept]
    new_feat_names = feature_names[kept]
    new_feat_names_set = set(new_feat_names.iloc[:, 0].values)
    new_voc_mapping = {
        source: target
        for (source, target) in voc_mapping.items()
        if target in new_feat_names_set
    }
    new_voc_set = new_feat_names_set.union(new_voc_mapping.keys())
    new_full_voc = full_voc[full_voc.iloc[:, 0].isin(new_voc_set)]
    return new_tfidf, new_full_voc, new_voc_mapping


def _fit_regression(
    tfidf: sparse.csr_matrix,
    pmcids: Sequence[int],
    coordinates: pd.DataFrame,
    output_dir: Path,
    n_jobs: int,
) -> Tuple[SmoothedRegression, sparse.csr_matrix, Sequence[int], Nifti1Image]:
    """Fit the linear regression part of the neuroquery model."""
    tfidf = normalize(tfidf, norm="l2", axis=1, copy=False)
    output_dir.mkdir(exist_ok=True, parents=True)
    memmap_file = output_dir.joinpath("brain_maps.dat")
    brain_maps, masker = img_utils.coordinates_to_maps(
        coordinates,
        target_affine=(4, 4, 4),
        id_column="pmcid",
        output_memmap_file=memmap_file,
        n_jobs=n_jobs,
    )
    brain_maps = brain_maps[(brain_maps.values != 0).any(axis=1)]
    kept_pmcids = brain_maps.index.intersection(pmcids)
    rindex = pd.Series(np.arange(len(pmcids)), index=pmcids)
    kept_tfidf = tfidf.A[rindex.loc[kept_pmcids].values, :]
    brain_maps = brain_maps.loc[kept_pmcids, :]
    regressor = SmoothedRegression()
    regressor.fit(kept_tfidf, brain_maps.values)
    return (
        regressor,
        sparse.csr_matrix(kept_tfidf),
        kept_pmcids,
        masker.mask_img_,
    )


def _do_fit_neuroquery(
    tfidf_dir: Path,
    extracted_data_dir: Path,
    output_dir: Path,
    n_jobs: int,
) -> NeuroQueryModel:
    """Helper for `fit_neuroquery` that performs the actual model fitting."""
    metadata = pd.read_csv(extracted_data_dir.joinpath("metadata.csv"))
    coordinates = pd.read_csv(extracted_data_dir.joinpath("coordinates.csv"))
    tfidf, full_voc, voc_mapping = _load_tfidf_for_frequent_terms(tfidf_dir)

    regressor, kept_tfidf, kept_pmcids, mask_img = _fit_regression(
        tfidf, metadata["pmcid"].values, coordinates, output_dir, n_jobs
    )
    metadata.set_index("pmcid", inplace=True)
    metadata = metadata.loc[kept_pmcids, :]
    metadata.index.name = "pmcid"
    metadata.reset_index(inplace=True)
    vectorizer = TextVectorizer.from_vocabulary(
        full_voc.iloc[:, 0].values,
        full_voc.iloc[:, 1].values,
        voc_mapping=voc_mapping,
        norm="l2",
    )
    encoder = NeuroQueryModel(
        vectorizer,
        regressor,
        mask_img,
        corpus_info={
            "tfidf": kept_tfidf,
            "metadata": metadata,
        },
    )
    return encoder


def fit_neuroquery(
    tfidf_dir: PathLikeOrStr,
    extracted_data_dir: Optional[PathLikeOrStr] = None,
    output_dir: Optional[PathLikeOrStr] = None,
    n_jobs: int = 1,
) -> Tuple[Path, int]:
    """Fit a NeuroQuery encoder.

    Parameters
    ----------
    vectorized_dir
        The directory containing the vectorized text (TFIDF features). It is
        the directory created by `nqdc.vectorize_corpus_to_npz` using
        `extracted_data_dir` as input.
    extracted_data_dir
        The directory containing extracted metadata and coordinates. It is a
        directory created by `nqdc.extract_data_to_csv`. If `None`, this
        function looks for a sibling directory of the `vectorized_dir` whose
        name ends with `_extractedData`.
    output_dir
        Directory in which to store the NeuroQuery model. If not specified, a
        sibling directory of `vectorized_dir` whose name ends with
        `_neuroqueryModel` is created. It will contain a `neuroquery_model`
        subdirectory that can be loaded with
        `neuroquery.NeuroQueryModel.from_data_dir`

    Returns
    -------
    output_dir
        The directory in which the neuroquery model is stored.
    exit_code
        0 if the neuroquery model was fitted and 1 otherwise. Used by the
        `nqdc` command-line interface.

    """
    tfidf_dir = Path(tfidf_dir)
    extracted_data_dir = _utils.get_extracted_data_dir_from_tfidf_dir(
        tfidf_dir, extracted_data_dir
    )
    output_dir = _utils.get_output_dir(
        tfidf_dir, output_dir, "_vectorizedText", "_neuroqueryModel"
    )
    status = _utils.check_steps_status(tfidf_dir, output_dir, __name__)
    if not status["need_run"]:
        return output_dir, 0
    encoder = _do_fit_neuroquery(
        tfidf_dir,
        extracted_data_dir,
        output_dir,
        n_jobs,
    )
    model_dir = output_dir.joinpath("neuroquery_model")
    encoder.to_data_dir(model_dir)
    is_complete = bool(status["previous_step_complete"])
    _utils.write_info(output_dir, name=_STEP_NAME, is_complete=is_complete)
    return output_dir, 0


class FitNeuroQueryStep(BaseProcessingStep):
    """Fitting NeuroQuery model as part of a pipeline (nqdc run)."""

    name = _STEP_NAME
    short_description = _STEP_DESCRIPTION

    def edit_argument_parser(self, argument_parser: ArgparseActions) -> None:
        argument_parser.add_argument(
            "--fit_neuroquery",
            action="store_true",
            help=_HELP,
        )
        _utils.add_n_jobs_argument(argument_parser)

    def run(
        self,
        args: argparse.Namespace,
        previous_steps_output: Mapping[str, Path],
    ) -> Tuple[Optional[Path], int]:
        if not args.fit_neuroquery:
            return None, 0
        return fit_neuroquery(
            previous_steps_output["vectorize"],
            previous_steps_output["extract_data"],
            n_jobs=args.n_jobs,
        )


class StandaloneFitNeuroqueryStep(BaseProcessingStep):
    """Fitting NeuroQuery as a standalone command (nqdc fit_neuroquery)."""

    name = _STEP_NAME
    short_description = _STEP_DESCRIPTION

    def edit_argument_parser(self, argument_parser: ArgparseActions) -> None:
        argument_parser.add_argument(
            "vectorized_data_dir",
            help="Directory containing TFIDF features and vocabulary. "
            "It is a directory created by nqdc whose name ends with "
            "'_vectorizedText'. A sibling directory will be created for "
            "the NeuroQuery model.",
        )
        _utils.add_n_jobs_argument(argument_parser)
        argument_parser.description = _HELP

    def run(
        self,
        args: argparse.Namespace,
        previous_steps_output: Mapping[str, Path],
    ) -> Tuple[Path, int]:
        return fit_neuroquery(args.vectorized_data_dir, n_jobs=args.n_jobs)
