import warnings
from itertools import product
from typing import List, Literal, Union
import numpy as np
import torch
from matchms import Spectrum
from matchms.similarity.BaseSimilarity import BaseSimilarity
from numba import cuda
from sparsestack import StackedSparseArray
from tqdm import tqdm
from ..utils import argbatch, get_device
from .spectrum_similarity_functions import cosine_kernel


class CudaCosineGreedy(BaseSimilarity):
    """
    Calculate cosine similarity score between two spectra using CUDA acceleration.

    The score is calculated by finding the best possible matches between peaks of two spectra.
    It provides a 'greedy' solution for the peak assignment problem, aimed at faster performance.

    This implementation is meant to replicate outputs of `matchms.similarity.CosineGreedy`.

    For example:

    >>> import numpy as np
    >>> from matchms import Spectrum
    >>> from simms.similarity import CudaCosineGreedy
    >>> reference = Spectrum(mz=np.array([100, 150, 200.]), intensities=np.array([0.7, 0.2, 0.1]))
    >>> query = Spectrum(mz=np.array([100, 140, 190.]), intensities=np.array([0.4, 0.2, 0.1]))
    >>> cosine_greedy = CudaCosineGreedy(tolerance=0.2)
    >>> score = cosine_greedy.pair(reference, query)
    >>> print(f"Cosine score is {score['score']:.2f} with {score['matches']} matched peaks")
    Cosine score is 0.83 with 1 matched peaks
    """

    score_datatype = [
        ("score", np.float32),
        ("matches", np.int32),
        ("overflow", np.bool_),
    ]

    def __init__(
        self,
        tolerance: float = 0.1,
        mz_power: float = 0.0,
        intensity_power: float = 1.0,
        shift: float = 0,
        batch_size: int = 2048,
        n_max_peaks: int = 1024,
        match_limit: int = 2048,
        sparse_threshold: float = 0.75,
        verbose=False,
    ):
        """
        Initialize CudaCosineGreedy with specified parameters.

        Parameters:
        -----------
        tolerance : float, optional
            Tolerance for considering peaks as matching, by default 0.1.
        mz_power : float, optional
            Exponent for m/z values in similarity score calculation, by default 0.0.
        intensity_power : float, optional
            Exponent for intensity values in similarity score calculation, by default 1.0.
        shift : float, optional
            Value to shift m/z values, by default 0.
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

        # Warn if CUDA device is unavailable
        if not cuda.is_available():
            warnings.warn(f"{self.__class__.__name__}: CUDA device seems unavailable.")

        # Initialize parameters
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.int_power = intensity_power
        self.shift = shift
        self.batch_size = batch_size
        self.match_limit = match_limit
        self.verbose = verbose
        self.n_max_peaks = n_max_peaks
        self.device = get_device()

        assert 0 <= sparse_threshold <= 1, "Sparse threshold has to be greather than 0."

        self.sparse_threshold = sparse_threshold

        # Compile kernel function
        self.kernel = cosine_kernel(
            tolerance=self.tolerance,
            shift=self.shift,
            mz_power=self.mz_power,
            int_power=self.int_power,
            match_limit=self.match_limit,
            batch_size=self.batch_size,
            n_max_peaks=self.n_max_peaks,
            spectra_dtype="float32",
        )

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
        dtype = np.float32
        mz = np.zeros((len(spectra), n_max_peaks), dtype=dtype)
        int = np.zeros((len(spectra), n_max_peaks), dtype=dtype)
        spectra_lens = np.zeros(len(spectra), dtype=np.int32)

        # Populate arrays
        for i, s in enumerate(spectra):
            if s is not None:
                spec_len = min(len(s.peaks), n_max_peaks)
                mz[i, :spec_len] = s._peaks.mz[:spec_len]
                int[i, :spec_len] = s._peaks.intensities[:spec_len]
                spectra_lens[i] = spec_len

        # Stack arrays and return
        stacked_spectra = np.stack([mz, int], axis=0)
        return stacked_spectra, spectra_lens

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
            rspec, rlen = self._spectra_peaks_to_tensor(references[bstart:bend])
            batches_r.append([rspec, rlen, bstart, bend])

        batches_q = []
        for bstart, bend in argbatch(queries, self.batch_size):
            qspec, qlen = self._spectra_peaks_to_tensor(queries[bstart:bend])
            batches_q.append([qspec, qlen, bstart, bend])

        batched_inputs = tuple(product(batches_r, batches_q))
        return batched_inputs

    def pair(self, reference: Spectrum, query: Spectrum) -> float:
        """
        Do not use, unless testing. GPUs work best with a lot of data at the same time.
        Calculates the cosine similarity score between a reference and a query spectrum.

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
            Unused. This unused argument is left is for compatibility reasons.

        Returns:
        --------
        Score : Union[np.ndarray, StackedSparseArray]
            Matrix of similarity scores between reference and query spectra.
            Type of Score depends on on `array_type` argument, with "sparse" array_type
            returning a `sparsestack.StackedSparseArray`
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
                (rspec, rlen, rstart, rend), (
                    qspec,
                    qlen,
                    qstart,
                    qend,
                ) = batched_inputs[batch_i]

                # Tensor holding lengths and norms
                metadata = torch.zeros(
                    4, self.batch_size, dtype=torch.float32, device=self.device
                )

                # Convert spectra to tensors and move to device
                rspec = torch.from_numpy(rspec).to(self.device)  # 2, R, N
                qspec = torch.from_numpy(qspec).to(self.device)  # 2, Q, M

                # Pre-calculate norms
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

                # Create tensor for lengths, and norms
                metadata[0, : len(rlen)] = torch.from_numpy(rlen).to(self.device)
                metadata[1, : len(qlen)] = torch.from_numpy(qlen).to(self.device)
                metadata[2, : len(rnorm)] = rnorm
                metadata[3, : len(qnorm)] = qnorm

                # Initialize output tensor
                out = torch.empty(
                    3,
                    self.batch_size,
                    self.batch_size,
                    dtype=torch.float32,
                    device=self.device,
                )

                # Convert tensors to CUDA arrays
                rspec = cuda.as_cuda_array(rspec)
                qspec = cuda.as_cuda_array(qspec)
                metadata = cuda.as_cuda_array(metadata)
                out = cuda.as_cuda_array(out)

                # Run GPU kernel
                self.kernel(rspec, qspec, metadata, out)

                # Convert output to tensor
                out = torch.as_tensor(out)

                out = out[:, : len(rlen), : len(qlen)]

                # Populate result based on array_type
                if array_type == "numpy":
                    result[:, rstart:rend, qstart:qend] = out

                elif array_type == "sparse":
                    mask = out[0] >= self.sparse_threshold
                    if mask.any():
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
                for bunch in result:
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
                    r = np.concatenate(r)
                    q = np.concatenate(q)
                    sp.add_sparse_data(
                        r,
                        q,
                        np.rec.fromarrays(
                            arrayList=[
                                np.concatenate(s),
                                np.concatenate(m),
                                np.concatenate(o),
                            ],
                            names=["score", "matches", "overflow"],
                        ),
                        name="sparse",
                    )
                return sp
