import numpy as np
import pytest
import torch
from matchms import Spectrum
from simms.peaks import PeakArrays
from simms.similarity import CudaCosineGreedy, CudaModifiedCosine


def random_spectra(n, rng, max_peaks=12):
    spectra = []
    for _ in range(n):
        k = int(rng.integers(1, max_peaks + 1))
        mz = np.sort(rng.choice(np.arange(100.0, 160.0, 0.5), size=k, replace=False))
        spectra.append(
            Spectrum(
                mz=mz,
                intensities=rng.uniform(0.1, 1.0, size=k),
                metadata={"precursor_mz": float(rng.uniform(300.0, 310.0))},
            )
        )
    return spectra


def test_from_peak_lists_pads_and_truncates():
    peaks = PeakArrays.from_peak_lists(
        mz=[np.array([100.0, 150.0, 200.0]), np.array([120.0])],
        intensities=[np.array([1.0, 0.5, 0.2]), np.array([0.7])],
        precursor_mz=[500.0, 600.0],
        n_max_peaks=2,
    )
    assert peaks.mz.tolist() == [[100.0, 150.0], [120.0, 0.0]]
    assert peaks.intensities[1].tolist() == pytest.approx([0.7, 0.0])
    assert peaks.lengths.tolist() == [2, 1]
    assert peaks.precursor_mz.tolist() == [500.0, 600.0]
    assert len(peaks[1:2]) == 1
    assert peaks[1:2].precursor_mz.tolist() == [600.0]


def test_from_spectra_keeps_peaks():
    spectra = random_spectra(5, np.random.default_rng(0))
    peaks = PeakArrays.from_spectra(spectra, with_precursor_mz=True)
    for i, s in enumerate(spectra):
        n = len(s.peaks)
        assert peaks.lengths[i] == n
        np.testing.assert_allclose(peaks.mz[i, :n], s.peaks.mz, rtol=1e-6)
        np.testing.assert_allclose(peaks.precursor_mz[i], s.get("precursor_mz"), rtol=1e-6)


def test_rejects_inconsistent_arrays():
    with pytest.raises(ValueError):
        PeakArrays(mz=np.zeros((2, 3)), intensities=np.zeros((2, 4)), lengths=np.zeros(2, int))
    with pytest.raises(ValueError):
        PeakArrays(mz=np.zeros((2, 3)), intensities=np.zeros((2, 3)), lengths=np.array([1, 5]))


@pytest.mark.parametrize("n_max_peaks", [16, 6])
def test_cosine_greedy_peak_arrays_match_spectra(n_max_peaks):
    rng = np.random.default_rng(1)
    references, queries = random_spectra(11, rng), random_spectra(7, rng)
    similarity = CudaCosineGreedy(tolerance=0.2, batch_size=8, n_max_peaks=n_max_peaks)
    expected = similarity.matrix(references, queries)
    got = similarity.matrix(PeakArrays.from_spectra(references), PeakArrays.from_spectra(queries))
    np.testing.assert_allclose(got["score"], expected["score"], rtol=1e-6)
    np.testing.assert_array_equal(got["matches"], expected["matches"])


def test_modified_cosine_peak_arrays_match_spectra():
    rng = np.random.default_rng(2)
    references, queries = random_spectra(9, rng), random_spectra(6, rng)
    similarity = CudaModifiedCosine(tolerance=0.2, batch_size=8, n_max_peaks=16)
    expected = similarity.matrix(references, queries)
    got = similarity.matrix(
        PeakArrays.from_spectra(references, with_precursor_mz=True),
        PeakArrays.from_spectra(queries, with_precursor_mz=True),
    )
    np.testing.assert_allclose(got["score"], expected["score"], rtol=1e-6)
    np.testing.assert_array_equal(got["matches"], expected["matches"])


def test_modified_cosine_default_storage_keeps_edge_matches():
    # A query peak just inside the tolerance in float64 that float32 rounding pushes outside it.
    tolerance = 0.1
    ref_mz, query_mz = next(
        (r, r + tolerance - eps)
        for r in 1000.0 + np.arange(200) * 1.3e-5
        for eps in np.linspace(1e-7, 5e-5, 50)
        if float(np.float32(r + tolerance - eps)) - float(np.float32(r)) > tolerance
    )
    reference = Spectrum(mz=np.array([ref_mz, 1100.0]), intensities=np.array([1.0, 0.5]), metadata={"precursor_mz": 1500.0})
    query = Spectrum(mz=np.array([query_mz, 1100.0]), intensities=np.array([1.0, 0.5]), metadata={"precursor_mz": 1500.0})
    similarity = CudaModifiedCosine(tolerance=tolerance, batch_size=4, n_max_peaks=4)
    expected = similarity.matrix([reference], [query])
    assert expected["matches"][0, 0] == 2
    got = similarity.matrix(
        PeakArrays.from_spectra([reference], with_precursor_mz=True),
        PeakArrays.from_spectra([query], with_precursor_mz=True),
    )
    np.testing.assert_array_equal(got["matches"], expected["matches"])
    np.testing.assert_allclose(got["score"], expected["score"], rtol=1e-12)
    rounded = similarity.matrix(
        PeakArrays.from_spectra([reference], with_precursor_mz=True, dtype=np.float32),
        PeakArrays.from_spectra([query], with_precursor_mz=True, dtype=np.float32),
    )
    assert rounded["matches"][0, 0] == 1


def test_from_peak_lists_stores_float64_by_default():
    peaks = PeakArrays.from_peak_lists([np.array([100.0])], [np.array([1.0])], precursor_mz=[500.123456789])
    assert peaks.mz.dtype == np.float64 and peaks.intensities.dtype == np.float64
    assert peaks.precursor_mz.dtype == np.float64
    small = PeakArrays.from_peak_lists([np.array([100.0])], [np.array([1.0])], precursor_mz=[500.123456789], dtype=np.float32)
    assert small.mz.dtype == np.float32 and small.precursor_mz.dtype == np.float64


def test_modified_cosine_needs_precursor_mz():
    spectra = random_spectra(3, np.random.default_rng(3))
    similarity = CudaModifiedCosine(batch_size=8, n_max_peaks=16)
    with pytest.raises(ValueError, match="precursor_mz"):
        similarity.matrix(PeakArrays.from_spectra(spectra), PeakArrays.from_spectra(spectra))


@pytest.mark.parametrize("similarity_class", [CudaCosineGreedy, CudaModifiedCosine])
def test_torch_output_matches_numpy(similarity_class):
    rng = np.random.default_rng(4)
    references, queries = random_spectra(10, rng), random_spectra(5, rng)
    similarity = similarity_class(tolerance=0.2, batch_size=8, n_max_peaks=16)
    expected = similarity.matrix(references, queries)
    got = similarity.matrix(references, queries, array_type="torch")
    assert isinstance(got, torch.Tensor)
    assert tuple(got.shape) == (3, 10, 5)
    np.testing.assert_allclose(got[0].cpu().numpy(), expected["score"], rtol=1e-6)
    np.testing.assert_array_equal(got[1].cpu().numpy(), expected["matches"])
