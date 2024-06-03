import contextlib
import io
import json
import logging
import shutil
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Type
import numpy as np
import pandas as pd
from joblib import Memory
from matchms import Spectrum
from matchms.filtering import (
    add_losses,
    normalize_intensities,
    reduce_to_number_of_peaks,
    require_minimum_number_of_peaks,
    select_by_mz,
    select_by_relative_intensity,
)
from matchms.similarity.BaseSimilarity import BaseSimilarity
from numba import cuda
from tqdm import tqdm


cache = Memory(
    "cache",
    verbose=0,
).cache
logger = logging.getLogger("cudams")


def argbatch(lst: list, batch_size: int) -> Iterable[tuple[int, int]]:
    """
    Given a list, return batches of indices, of size `batch_size`. It can be used as follows:

    ```
    for bstart, bend in argbatch(references, 2048):
        rbatch = references[bstart:bend]
        # Do something with `rbatch`
    ```
    """
    for i in range(0, len(lst), batch_size):
        yield i, i + batch_size


def mkdir(p: Path, clean=False) -> Path:
    "Modified pathlib mkdir, made a bit convenient to use."
    p = Path(p)
    if clean and p.is_dir() and p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(exist_ok=True, parents=True)
    return p


@contextlib.contextmanager
def mute_stdout():
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

def spectra_peaks_to_tensor(
    spectra: list,
    dtype: str = "float32",
    n_max_peaks: int = None,
    ignore_null_spectra: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Working with GPU requires us to have a fixed shape for mz/int arrays.
    This isn't the case for real-life data, so we have to either pad or trim the spectra so that
    all of them are the same size.

    Returns:
        spectra: [2, len(spectra)] float32
        batch: [len(spectra)] int32
    """
    dynamic_shape = max(len(s.peaks) for s in spectra)
    n_max_peaks = dynamic_shape if n_max_peaks is None else n_max_peaks

    mz = np.empty((len(spectra), n_max_peaks), dtype=dtype)
    int = np.empty((len(spectra), n_max_peaks), dtype=dtype)
    batch = np.empty(len(spectra), dtype=np.int32)
    for i, s in enumerate(spectra):
        if s is not None:
            # .to_numpy creates an unneeded copy - we don't need to do that twice
            spec_len = min(len(s.peaks), n_max_peaks)
            mz[i, :spec_len] = s._peaks.mz[:spec_len]
            int[i, :spec_len] = s._peaks.intensities[:spec_len]
            batch[i] = spec_len
        elif s is None and ignore_null_spectra:
            batch[i] = 0
    spec = np.stack([mz, int], axis=0)
    return spec, batch


def process_spectrum(spectrum: Spectrum) -> Optional[Spectrum]:
    """
    One of the many ways to preprocess the spectrum - we use this by default.
    """
    spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001)
    spectrum = reduce_to_number_of_peaks(spectrum, n_max=1024)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
    return spectrum


def get_ref_spectra_from_df(
    spectra_df,
    limit=None,
    spectrum_processor: callable = process_spectrum,
) -> list[Spectrum]:
    """
    Convenience function that reads a pair of CSV datasets with columns like `pbid`,`precursor_mz`, etc. and
    returns parsed and preprocessed spectra as a list.
    """
    from joblib import Parallel, delayed

    # for index, row in spectra_df.iterrows():
    def fn(index, row):
        pbid = row["pbid"]
        precursor_mz = row["precursor_mz"]
        smiles = row["pb_smiles"]
        inchikey = row["pb_inchikey"]
        mz_array = np.array(json.loads(row["peaks_mz"]))
        intensity_array = np.array(json.loads(row["peaks_intensities"]))
        sp = Spectrum(
            mz=mz_array,
            intensities=intensity_array,
            metadata={
                "id": pbid,
                "precursor_mz": precursor_mz,
                "smiles": smiles,
                "inchikey": inchikey,
            },
        )
        if spectrum_processor is not None:
            sp = spectrum_processor(sp)
        return sp

    if limit is not None:
        spectra_df = spectra_df.head(limit)
    spectra = Parallel(-2)(
        delayed(fn)(index, row)
        for index, row in tqdm(spectra_df.iterrows(), total=len(spectra_df))
    )
    spectra = [s for s in spectra if s is not None]
    return spectra

def download(
    name: Literal[
        "ALL_GNPS.mgf",
        "ALL_GNPS.pickle",
        "GNPS-random-1k.mgf",
        "GNPS-random-10k.mgf",
        "GNPS-LIBRARY.mgf",
        "GNPS-LIBRARY.pickle",
        "GNPS-LIBRARY-default-filter-nmax-2048.pickle",
        "GNPS-LIBRARY-default-filter-nmax-1024.pickle",
        "pesticides.mgf",
        "GNPS-MSMLS.mgf",
        "MASSBANK.mgf",
    ]
) -> str:
    """
    Downloads a set of sample spectra files from https://github.com/PangeAI/cudams/releases/tag/samples-0.1
    Downloaded files are cached, and not re-downloaded after the initial call.
    """
    import pooch

    return pooch.retrieve(
        # TODO: Before we fully migrate we still use old repo for file reference
        # url=f"https://github.com/PangeAI/cudams/releases/download/samples-0.1/{name}",
        url=f"https://github.com/tornikeo/cudams/releases/download/samples-0.1/{name}",
        known_hash=None,
        progressbar=True,
    )

class Timer:
    def __enter__(self):
        self.duration = -time.perf_counter()
        return self

    def __exit__(self, *args):
        self.duration += time.perf_counter()


@cache
def get_correct_scores(
    references: list,
    queries: list,
    similarity_class: Type[BaseSimilarity],
    **similarity_parameters,
) -> np.ndarray:
    """
    MatchMS is quite slow for large number of spectra. To avoid re-calculating same scores with exact same arguments for testing,
    we cache the results on the disk and read them back if everything matches (class, args, all spectra involved).
    """
    return similarity_class(**similarity_parameters).matrix(references, queries)
