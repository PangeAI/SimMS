"""Spectra as padded peak arrays, an alternative to lists of matchms ``Spectrum`` objects.

The similarity kernels work on fixed-width arrays of m/z and intensities. ``matrix()`` builds them from
``Spectrum`` objects one spectrum at a time; for millions of spectra that conversion, and creating the
``Spectrum`` objects in the first place, can take longer than the GPU computation. ``PeakArrays`` holds the
arrays directly, so data that is already in arrays (Parquet, HDF5, NumPy) can go straight to the GPU.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class PeakArrays:
    """A batch of spectra as zero-padded arrays.

    Attributes
    ----------
    mz : np.ndarray
        ``(n_spectra, width)`` m/z values. Each row holds its spectrum's peaks in ascending m/z, then zeros.
    intensities : np.ndarray
        ``(n_spectra, width)`` intensities, aligned with ``mz``.
    lengths : np.ndarray
        ``(n_spectra,)`` number of peaks in each row.
    precursor_mz : np.ndarray, optional
        ``(n_spectra,)`` precursor m/z, needed by ``CudaModifiedCosine``.

    >>> import numpy as np
    >>> from simms.peaks import PeakArrays
    >>> peaks = PeakArrays.from_peak_lists(
    ...     mz=[np.array([100.0, 150.0]), np.array([120.0, 130.0, 140.0])],
    ...     intensities=[np.array([0.5, 1.0]), np.array([1.0, 0.2, 0.3])],
    ... )
    >>> peaks.mz.shape, peaks.lengths.tolist()
    ((2, 3), [2, 3])
    """

    mz: np.ndarray
    intensities: np.ndarray
    lengths: np.ndarray
    precursor_mz: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.mz.ndim != 2 or self.mz.shape != self.intensities.shape:
            raise ValueError("mz and intensities must be 2-D arrays of the same shape.")
        if self.lengths.shape != (self.mz.shape[0],):
            raise ValueError("lengths must have one entry per spectrum.")
        if self.lengths.size and (self.lengths.min() < 0 or self.lengths.max() > self.mz.shape[1]):
            raise ValueError("lengths must lie between 0 and the array width.")
        if self.precursor_mz is not None and self.precursor_mz.shape != (self.mz.shape[0],):
            raise ValueError("precursor_mz must have one entry per spectrum.")

    def __len__(self) -> int:
        return self.mz.shape[0]

    def __getitem__(self, index: slice) -> "PeakArrays":
        if not isinstance(index, slice):
            raise TypeError("PeakArrays supports slicing only, e.g. peaks[0:2048].")
        return PeakArrays(
            mz=self.mz[index],
            intensities=self.intensities[index],
            lengths=self.lengths[index],
            precursor_mz=None if self.precursor_mz is None else self.precursor_mz[index],
        )

    @property
    def width(self) -> int:
        """The number of peak slots per spectrum."""
        return self.mz.shape[1]

    def to_width(self, width: int, dtype=np.float32) -> "tuple[np.ndarray, np.ndarray]":
        """Stacked ``(2, n_spectra, width)`` m/z and intensities, and lengths capped at ``width``.

        Rows longer than ``width`` keep their first ``width`` peaks, as ``matrix()`` does for ``Spectrum``
        objects with more than ``n_max_peaks`` peaks.
        """
        stacked = np.zeros((2, len(self), width), dtype=dtype)
        keep = min(width, self.width)
        stacked[0, :, :keep] = self.mz[:, :keep]
        stacked[1, :, :keep] = self.intensities[:, :keep]
        return stacked, np.minimum(self.lengths, width).astype(np.int32)

    @classmethod
    def from_peak_lists(
        cls,
        mz: Sequence[np.ndarray],
        intensities: Sequence[np.ndarray],
        precursor_mz: Optional[Sequence[float]] = None,
        n_max_peaks: Optional[int] = None,
        dtype=np.float64,
    ) -> "PeakArrays":
        """Pad per-spectrum peak arrays (each in ascending m/z) into a ``PeakArrays``.

        With ``n_max_peaks``, each spectrum keeps its first ``n_max_peaks`` peaks.

        ``dtype`` is the storage type of m/z and intensities. The float64 default keeps the values matchms
        ``Spectrum`` objects hold, so both similarity classes give exactly the scores they give for ``Spectrum``
        input. float32 halves the memory and is exact for ``CudaCosineGreedy``, which computes in float32, but
        ``CudaModifiedCosine`` computes in float64 by default, and rounded m/z can flip matches that lie near the
        tolerance. Precursor m/z is always stored as float64.
        """
        if len(mz) != len(intensities):
            raise ValueError("mz and intensities must hold the same number of spectra.")
        lengths = np.fromiter((len(m) for m in mz), dtype=np.int64, count=len(mz))
        if n_max_peaks is not None:
            lengths = np.minimum(lengths, n_max_peaks)
        width = int(lengths.max(initial=0))
        mz_out = np.zeros((len(mz), width), dtype=dtype)
        int_out = np.zeros((len(mz), width), dtype=dtype)
        for i, (m, x) in enumerate(zip(mz, intensities)):
            n = lengths[i]
            mz_out[i, :n] = m[:n]
            int_out[i, :n] = x[:n]
        return cls(
            mz=mz_out,
            intensities=int_out,
            lengths=lengths.astype(np.int32),
            precursor_mz=None if precursor_mz is None else np.asarray(precursor_mz, dtype=np.float64),
        )

    @classmethod
    def from_spectra(cls, spectra, n_max_peaks: Optional[int] = None, with_precursor_mz: bool = False, dtype=np.float64) -> "PeakArrays":
        """Convert matchms ``Spectrum`` objects once, so that repeated ``matrix()`` calls skip the conversion."""
        precursor = None
        if with_precursor_mz:
            from .similarity.CudaModifiedCosine import get_valid_precursor_mz

            precursor = [get_valid_precursor_mz(s) for s in spectra]
        return cls.from_peak_lists(
            mz=[s.peaks.mz for s in spectra],
            intensities=[s.peaks.intensities for s in spectra],
            precursor_mz=precursor,
            n_max_peaks=n_max_peaks,
            dtype=dtype,
        )
