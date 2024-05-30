import logging
import warnings
from itertools import product
from pathlib import Path
from typing import List, Literal, Union
import numpy as np
import torch
from matchms import Spectrum
from matchms.filtering.metadata_processing.add_precursor_mz import _convert_precursor_mz
from matchms.similarity.BaseSimilarity import BaseSimilarity
from numba import cuda
from sparsestack import StackedSparseArray
from tqdm import tqdm
from ..utils import argbatch
from .spectrum_similarity_functions import modified_cosine_kernel


logger = logging.getLogger("cudams")


def get_valid_precursor_mz(spectrum):
    """Extract valid precursor_mz from spectrum if possible. If not raise exception."""
    message_precursor_missing = (
        "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
    )
    message_precursor_no_number = "Precursor_mz must be of type int or float. Apply 'add_precursor_mz' filter first."
    message_precursor_below_0 = (
        "Expect precursor to be positive number." "Apply 'require_precursor_mz' first"
    )

    precursor_mz = spectrum.get("precursor_mz", None)
    if not isinstance(precursor_mz, (int, float)):
        logger.warning(message_precursor_no_number)
    precursor_mz = _convert_precursor_mz(precursor_mz)
    assert precursor_mz is not None, message_precursor_missing
    assert precursor_mz > 0, message_precursor_below_0
    return precursor_mz


class CudaModifiedCosine(BaseSimilarity):
    """
    Calculate 'cosine similarity score' between two spectra using CUDA acceleration.

    CudaModifiedCosine calculates the cosine similarity score between two mass spectra
    using CUDA acceleration. The score is calculated by finding the best possible
    matches between peaks of two spectra. It provides a 'greedy' solution for the
    peak assignment problem, aimed at faster performance.

    Parameters:
    -----------
    tolerance : float, optional
        Tolerance for considering peaks as matching, by default 0.1.
    mz_power : float, optional
        Exponent for m/z values in similarity score calculation, by default 0.0.
    intensity_power : float, optional
        Exponent for intensity values in similarity score calculation, by default 1.0.
    batch_size : int, optional
        Batch size for processing spectra, by default 2048.
    n_max_peaks : int, optional
        Maximum number of peaks to consider in each spectrum, by default 1024.
    match_limit : int, optional
        Limit on the number of matches allowed, by default 2048.
    sparse_threshold : float, optional
        Threshold for considering scores in sparse output, by default 0.75.
    verbose : bool, optional
        Verbosity flag, by default False.

    Attributes:
    -----------
    kernel_time : float
        Time taken by the CUDA kernel for computation.
    device : str
        Device used for computation, either 'cuda' or 'cpu'.

    Methods:
    --------
    pair(reference: Spectrum, query: Spectrum) -> float:
        Calculate the cosine similarity score between a reference and a query spectrum.
    matrix(references: List[Spectrum], queries: List[Spectrum], array_type: Literal["numpy", "sparse"] = "numpy", is_symmetric: bool = False) -> np.ndarray:
        Calculate a matrix of similarity scores between reference and query spectra.
    """

    score_datatype = [
        ("score", np.float32),
        ("matches", np.int32),
        ("overflow", np.uint8),
    ]

    def __init__(
        self,
        tolerance: float = 0.1,
        mz_power: float = 0.0,
        intensity_power: float = 1.0,
        batch_size: int = 2048,
        n_max_peaks: int = 1024,
        match_limit: int = 2048,
        sparse_threshold: float = 0.75,
        verbose=False,
    ):
        """
        Initialize CudaModifiedCosine with specified parameters.

        Parameters:
        -----------
        tolerance : float, optional
            Tolerance for considering peaks as matching, by default 0.1.
        mz_power : float, optional
            Exponent for m/z values in similarity score calculation, by default 0.0.
        intensity_power : float, optional
            Exponent for intensity values in similarity score calculation, by default 1.0.
        batch_size : int, optional
            Batch size for processing spectra, by default 2048.
        n_max_peaks : int, optional
            Maximum number of peaks to consider in each spectrum, by default 1024.
        match_limit : int, optional
            Limit on the number of matches allowed, by default 2048.
        sparse_threshold : float, optional
            Threshold for considering scores in sparse output, by default 0.75.
        verbose : bool, optional
            Verbosity flag, by default False.
        """

        # Initialize parameters
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.int_power = intensity_power
        self.batch_size = batch_size
        self.match_limit = match_limit
        self.verbose = verbose
        self.n_max_peaks = n_max_peaks
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sparse_threshold = sparse_threshold

        # Compile kernel function
        self.kernel = modified_cosine_kernel(
            tolerance=self.tolerance,
            mz_power=self.mz_power,
            int_power=self.int_power,
            match_limit=self.match_limit,
            batch_size=self.batch_size,
            n_max_peaks=self.n_max_peaks,
        )

        # Warn if CUDA device is unavailable
        if not cuda.is_available():
            warnings.warn(f"{self.__class__.__name__}: CUDA device seems unavailable.")

    def _spectra_peaks_to_tensor(
        self,
        spectra: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert list of Spectrum objects into numpy arrays.

        Parameters:
        -----------
        spectra : list of Spectrum
            List of Spectrum objects.

        Returns:
        --------
        tuple of np.ndarray
            Tuple containing two numpy arrays: one for m/z values and one for intensities.
        """
        # Determine dynamic shape
        dynamic_shape = max(len(s.peaks) for s in spectra)
        if self.n_max_peaks is None:
            n_max_peaks = dynamic_shape
        else:
            n_max_peaks = self.n_max_peaks

        # Initialize arrays
        dtype = self.score_datatype[0][1]
        mz = np.zeros((len(spectra), n_max_peaks), dtype=dtype)
        int_ = np.zeros((len(spectra), n_max_peaks), dtype=dtype)
        spectra_lens = np.zeros(len(spectra), dtype=np.int32)
        spectra_precursor_mzs = np.zeros(len(spectra), dtype=dtype)

        # Populate arrays
        for i, s in enumerate(spectra):
            if s is not None:
                spec_len = min(len(s.peaks), n_max_peaks)
                mz[i, :spec_len] = s._peaks.mz[:spec_len]
                int_[i, :spec_len] = s._peaks.intensities[:spec_len]
                spectra_lens[i] = spec_len
                spectra_precursor_mzs[i] = get_valid_precursor_mz(s)
        # Stack arrays and return
        stacked_spectra = np.stack([mz, int_], axis=0)
        return stacked_spectra, spectra_lens, spectra_precursor_mzs

    def _get_batches(self, references, queries):
        """
        Generate batches of spectra pairs for processing.

        Parameters:
        -----------
        references : list
            List of reference Spectrum objects.
        queries : list
            List of query Spectrum objects.

        Returns:
        --------
        list of tuple
            List of tuples containing batched input data.
        """
        batches_r = []
        for bstart, bend in argbatch(references, self.batch_size):
            rspec, rlen, rpmz = self._spectra_peaks_to_tensor(references[bstart:bend])
            batches_r.append([rspec, rlen, rpmz, bstart, bend])

        batches_q = []
        for bstart, bend in argbatch(queries, self.batch_size):
            qspec, qlen, qpmz = self._spectra_peaks_to_tensor(queries[bstart:bend])
            batches_q.append([qspec, qlen, qpmz, bstart, bend])

        batched_inputs = tuple(product(batches_r, batches_q))
        return batched_inputs

    def pair(self, reference: Spectrum, query: Spectrum) -> float:
        """
        Calculate the cosine similarity score between a reference and a query spectrum. Used for testing only.

        Parameters:
        -----------
        reference : Spectrum
            Reference spectrum.
        query : Spectrum
            Query spectrum.

        Returns:
        --------
        float
            Cosine similarity score between the reference and query spectra.
        """
        result_mat = self.matrix([reference], [query])
        return np.asarray(result_mat.squeeze(), dtype=self.score_datatype)

    def matrix(
        self,
        references: List[Spectrum],
        queries: List[Spectrum],
        array_type: Literal["numpy", "sparse"] = "numpy",
        is_symmetric: bool = False,
    ) -> Union[np.ndarray, StackedSparseArray]:
        """
        Calculate a matrix of similarity scores between reference and query spectra.

        Parameters:
        -----------
        references : List[Spectrum]
            List of reference Spectrum objects.
        queries : List[Spectrum]
            List of query Spectrum objects.
        array_type : Literal["numpy", "sparse"], optional
            Specify the output array type, by default "numpy".
        is_symmetric : bool, optional
            Set to True when references and queries are identical, by default False.

        Returns:
        --------
        np.ndarray or StackedSparseArray, depending on `array_type` argument.
            Matrix of similarity scores between reference and query spectra.
        """
        # Warn if is_symmetric is passed
        if is_symmetric:
            warnings.warn("is_symmetric is ignored here, it has no effect.")

        # Check if array_type is valid
        assert array_type in [
            "numpy",
            "sparse",
        ], "Invalid array_type. Use 'numpy' or 'sparse'."

        # Initialize batched inputs
        batched_inputs = self._get_batches(references=references, queries=queries)
        R, Q = len(references), len(queries)

        # Initialize result variable based on array_type
        if array_type == "numpy":
            result = torch.empty(3, R, Q, dtype=torch.float32, device=self.device)
        elif array_type == "sparse":
            result = []

        # Iterate over batched inputs
        with torch.no_grad():
            for batch_i in tqdm(range(len(batched_inputs)), disable=not self.verbose):
                # Unpack batched inputs
                (rspec, rlen, rpmz, rstart, rend), (
                    qspec,
                    qlen,
                    qpmz,
                    qstart,
                    qend,
                ) = batched_inputs[batch_i]

                # Create tensor for spectrum lengths
                lens = torch.zeros(2, self.batch_size, dtype=torch.int32)
                lens[0, : len(rlen)] = torch.from_numpy(rlen)
                lens[1, : len(qlen)] = torch.from_numpy(qlen)
                lens = lens.to(self.device)

                # Convert spectra to tensors and move to device
                rspec = torch.from_numpy(rspec).to(self.device)  # 2, R, N
                qspec = torch.from_numpy(qspec).to(self.device)  # 2, Q, M

                # Pre-calculate metadata (includes norm + precursor_mz)
                meta = torch.ones(4, self.batch_size, dtype=torch.float32).to(
                    self.device
                )
                rnorm = (
                    (
                        (
                            rspec[0, :, :] ** self.mz_power
                            * rspec[1, :, :] ** self.int_power
                        )
                        ** 2
                    )
                    .sum(-1)
                    .sqrt()
                )  # R
                qnorm = (
                    (
                        (
                            qspec[0, :, :] ** self.mz_power
                            * qspec[1, :, :] ** self.int_power
                        )
                        ** 2
                    )
                    .sum(-1)
                    .sqrt()
                )  # Q
                meta[0, : len(rnorm)] = rnorm
                meta[1, : len(qnorm)] = qnorm

                meta[2, : len(rnorm)] = torch.from_numpy(rpmz)
                meta[3, : len(qnorm)] = torch.from_numpy(qpmz)

                # Convert tensors to CUDA arrays
                rspec = cuda.as_cuda_array(rspec)
                qspec = cuda.as_cuda_array(qspec)
                lens = cuda.as_cuda_array(lens)
                meta = cuda.as_cuda_array(meta)

                # Initialize output tensor
                out = torch.empty(
                    3,
                    self.batch_size,
                    self.batch_size,
                    dtype=torch.float32,
                    device=self.device,
                )
                out = cuda.as_cuda_array(out)

                self.kernel(rspec, qspec, lens, meta, out)

                # Convert output to tensor
                out = torch.as_tensor(out)

                # Populate result based on array_type
                if array_type == "numpy":
                    result[:, rstart:rend, qstart:qend] = out[
                        :, : len(rlen), : len(qlen)
                    ]
                elif array_type == "sparse":
                    mask = out[0] >= self.sparse_threshold
                    row, col = torch.nonzero(mask, as_tuple=True)
                    rabs = (rstart + row).cpu()
                    qabs = (qstart + col).cpu()
                    score, matches, overflow = out[:, mask].cpu()
                    result.append(
                        dict(
                            rabs=rabs.int().cpu().numpy(),
                            qabs=qabs.int().cpu().numpy(),
                            score=score.float().cpu().numpy(),
                            matches=matches.int().cpu().numpy(),
                            overflow=overflow.bool().cpu().numpy(),
                        )
                    )

            # Return result based on array_type
            if array_type == "numpy":
                return np.rec.fromarrays(
                    result.cpu().numpy(),
                    dtype=self.score_datatype,
                )
            elif array_type == "sparse":
                sp = StackedSparseArray(len(references), len(queries))
                sparse_data = []

                for bunch in tqdm(result, disable=not self.verbose):
                    sparse_data.append(
                        (
                            bunch["rabs"],
                            bunch["qabs"],
                            bunch["score"],
                            bunch["matches"],
                            bunch["overflow"],
                        )
                    )

                if sparse_data:
                    r, q, s, m, o = zip(*sparse_data)
                    sp.add_sparse_data(
                        np.array(r),
                        np.array(q),
                        np.rec.fromarrays(
                            arrayList=[np.array(s), np.array(m), np.array(o)],
                            names=["score", "matches", "overflow"],
                        ),
                        name="sparse",
                    )
                return sp
