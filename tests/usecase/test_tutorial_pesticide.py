from pathlib import Path
import numpy as np
from matchms import calculate_scores
from matchms.filtering import (
    default_filters,
    normalize_intensities,
    reduce_to_number_of_peaks,
)
from matchms.importing import load_from_mgf
from matchms.similarity import CosineGreedy
from numba import cuda
from simms.similarity import CudaCosineGreedy
from simms.utils import download


def test_tutorial_pesticide():
    pest_file = download("pesticides.mgf")
    assert isinstance(pest_file, str), "Don't use strings for downloader"

    file = list(load_from_mgf(str(pest_file)))
    # Apply filters to clean and enhance each spectrum
    spectrums = []

    for spectrum in file:
        # Apply default filter to standardize ion mode, correct charge and more.
        # Default filter is fully explained at https://matchms.readthedocs.io/en/latest/api/matchms.filtering.html .
        spectrum = default_filters(spectrum)
        # Scale peak intensities to maximum of 1
        spectrum = normalize_intensities(spectrum)
        spectrums.append(spectrum)

    scores = calculate_scores(
        references=spectrums,
        queries=spectrums,
        similarity_function=CosineGreedy(),
        is_symmetric=True,
    )

    # Matchms allows to get the best matches for any query using scores_by_query
    query = spectrums[15]  # just an example
    best_matches = scores.scores_by_query(query, "CosineGreedy_score", sort=True)

    # It is necessary to make sure that the number of peaks are reasonable

    MAX_PEAKS = 1024

    def process_spectrum(spectrum: np.ndarray) -> np.ndarray:
        # spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
        # spectrum = normalize_intensities(spectrum)
        # spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001)
        # spectrum = reduce_to_number_of_peaks(spectrum, n_max=1000)
        spectrum = reduce_to_number_of_peaks(spectrum, n_max=MAX_PEAKS)
        # spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
        return spectrum

    f_specs = [process_spectrum(s) for s in spectrums]
    scores_cu = calculate_scores(
        references=f_specs,
        queries=f_specs,
        similarity_function=CudaCosineGreedy(batch_size=512),
    )
    # This computed all-vs-all similarity scores, the array of which can be accessed as scores.scores
    print(f"Size of matrix of computed similarities: {scores_cu.scores.shape}")

    # Matchms allows to get the best matches for any query using scores_by_query
    query = spectrums[15]  # just an example
    best_matches_cu = scores_cu.scores_by_query(
        query, "CudaCosineGreedy_score", sort=True
    )

    # Print the calculated scores_cu for each spectrum pair
    for reference, (score, matches, overflow) in best_matches_cu[:10]:
        # Ignore scores_cu between same spectrum
        if reference != query:
            print(f"Reference scan id: {reference.metadata['scans']}")
            print(f"Query scan id: {query.metadata['scans']}")
            print(f"Score: {score:.4f}")
            print(f"Number of matching peaks: {matches}")
            print(f"Did GPU overflow at this pair: {overflow}")
            print("----------------------------")

    for a, b in zip(best_matches[:20], best_matches_cu[:20]):
        reference, (score_a, matches_a) = a
        reference, (score_b, matches_b, overflow_b) = b
        if reference != query:
            # If we didn't overflow
            if not overflow_b:
                assert np.isclose(score_a, score_b), ("score error", score_a, score_b)
                assert np.isclose(matches_a, matches_b), (
                    "match error",
                    matches_a,
                    matches_b,
                )
            # If overflow, score must be leq
            else:
                assert score_a >= score_b
                assert matches_a >= matches_b
