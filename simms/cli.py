import click
import os
from pathlib import Path
from typing import List, Literal, Optional
from matchms import Spectrum
from matchms.filtering import (
    normalize_intensities,
    reduce_to_number_of_peaks,
    select_by_mz,
    select_by_relative_intensity,
)
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from matchms import calculate_scores
from matchms.importing import load_spectra
from simms.similarity import CudaCosineGreedy, CudaFingerprintSimilarity, CudaModifiedCosine

def process_spectrum(spectrum: Spectrum) -> Optional[Spectrum]:
    """
    One of the many ways to preprocess the spectrum - we use this by default.
    """
    spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001)
    spectrum = reduce_to_number_of_peaks(spectrum, n_max=1024)
    return spectrum

def preprocess_spectra(spectra: List[Spectrum]) -> List[Spectrum]:
    spectra = [s for s in (process_spectrum(s) for s in spectra) if s is not None]
    return spectra

@click.command()
@click.option('--references', type=click.Path(exists=True), required=True, help='Path to the reference MGF file.')
@click.option('--queries', type=click.Path(exists=True), required=True, help='Path to the query MGF file.')
@click.option('--output_file', type=click.Path(), default='scores.pickle', help='Path to save the output .pickle file.')
@click.option('--tolerance', default=0.1, help='Tolerance for matching peaks.')
@click.option('--mz_power', default=0.0, help='Power of mz in similarity calculation.')
@click.option('--intensity_power', default=1.0, help='Power of intensity in similarity calculation.')
@click.option('--shift', default=0, help='Shift in mz for similarity calculation.')
@click.option('--batch_size', default=2048, help='Batch size for GPU processing.')
@click.option('--n_max_peaks', default=1024, help='Maximum number of peaks in spectrum.')
@click.option('--match_limit', default=2048, help='Maximum number of matches to consider.')
@click.option('--array_type', default='numpy', type=click.Choice(['sparse', 'numpy']), help='Type of output array. For very large spectra, sparse is recommended.')
@click.option('--sparse_threshold', default=0.75, help='Threshold for sparse array.')
@click.option('--method', default='CudaCosineGreedy', type=click.Choice(['CudaCosineGreedy', 'CudaFingerprintSimilarity', 'CudaModifiedCosine']), help='Similarity method to use.')
@click.option('--visualize', is_flag=True, help='Generate visualization images.')
@click.option('--preprocess', is_flag=True, help='Apply preprocessing to spectra.')
def main(references, queries, output_file, tolerance, mz_power, intensity_power, shift, batch_size, n_max_peaks, match_limit, array_type, sparse_threshold, method, visualize, preprocess):
    print = click.echo

    # Ensure output_file has a .pickle extension
    if not output_file.endswith('.pickle'):
        click.echo("Output file must have a .pickle extension.")
        return

    references = Path(references)
    queries = Path(queries)

    # Check if the reference and query files exist
    if not references.exists():
        click.echo(f"Reference file does not exist: {references}")
        return
    if not references.exists():
        click.echo(f"Query file does not exist: {references}")
        return

    # Load and optionally preprocess spectra
    refs = list(load_spectra(str(references)))
    ques = list(load_spectra(str(queries)))

    if preprocess:
        refs = preprocess_spectra(refs)
        ques = preprocess_spectra(ques)

    # Adjust batch size if necessary
    if batch_size > max(len(refs), len(ques)):
        batch_size = max(len(refs), len(ques))

    # Select the similarity method
    similarity_functions = {
        "CudaCosineGreedy": CudaCosineGreedy,
        "CudaFingerprintSimilarity": CudaFingerprintSimilarity,
        "CudaModifiedCosine": CudaModifiedCosine
    }
    similarity_function = similarity_functions[method](
        tolerance=tolerance,
        mz_power=mz_power,
        intensity_power=intensity_power,
        shift=shift,
        batch_size=batch_size,
        n_max_peaks=n_max_peaks,
        match_limit=match_limit,
        sparse_threshold=sparse_threshold,
    )

    # Calculate scores
    scores_obj = calculate_scores(refs, ques, similarity_function=similarity_function, array_type=array_type)

    scores_obj.to_pickle(output_file)

    # Optionally generate visualization
    if visualize:
        score_vis = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
        scores = scores_obj.to_array()
        
        axs[0].imshow(scores[f"{method}_score"])
        axs[0].set_title("Score")
        
        axs[1].imshow(scores[f"{method}_matches"])
        axs[1].set_title("Matches")
        
        plt.suptitle("Score and Matches")
        plt.savefig(score_vis.name)
        print(f"Visualization saved at: {score_vis.name}")

    print(f"Scores saved at: {output_file}")


if __name__ == "__main__":
    main()
