"""
Gradio UI for running Cosine Greedy without writing code.
"""

import gradio as gr
import torch
import os
from pathlib import Path
from matchms import Spectrum
from typing import List, Optional, Literal
# os.system("nvidia-smi")
# print("TORCH_CUDA", torch.cuda.is_available())

def preprocess_spectra(spectra: List[Spectrum]) -> Spectrum:
    from matchms.filtering import select_by_intensity, \
        normalize_intensities, \
        select_by_relative_intensity, \
        reduce_to_number_of_peaks, \
        select_by_mz, \
        require_minimum_number_of_peaks
    
    def process_spectrum(spectrum: Spectrum) -> Optional[Spectrum]:
        """
        One of the many ways to preprocess the spectrum - we use this by default.
        """
        spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
        spectrum = normalize_intensities(spectrum)
        spectrum = select_by_relative_intensity(spectrum, intensity_from=0.001)
        spectrum = reduce_to_number_of_peaks(spectrum, n_max=1024)
        return spectrum
    
    spectra = list(process_spectrum(s) for s in spectra) # Some might be None
    return spectra

def run(r_filepath:Path, q_filepath:Path,
        tolerance: float = 0.1,
        mz_power: float = 0.0,
        intensity_power: float = 1.0,
        shift: float = 0,
        batch_size: int = 2048,
        n_max_peaks: int = 1024,
        match_limit: int = 2048,
        array_type: Literal['sparse','numpy'] = "numpy",
        sparse_threshold: float = .75):
    print('\n>>>>', r_filepath, q_filepath, array_type, '\n')
    # debug = os.getenv('CUDAMS_DEBUG') == '1'
    # if debug:
    #     r_filepath = Path('tests/data/pesticides.mgf')
    #     q_filepath = Path('tests/data/pesticides.mgf')

    assert r_filepath is not None, "Reference file is missing."
    assert q_filepath is not None, "Query file is missing."
    import tempfile
    import numpy as np
    from cudams.similarity import CudaCosineGreedy
    from matchms.importing import load_from_mgf
    from matchms import calculate_scores
    import matplotlib.pyplot as plt

    refs = preprocess_spectra(list(load_from_mgf(str(r_filepath))))
    ques = preprocess_spectra(list(load_from_mgf(str(q_filepath))))

    # If we have small spectra, don't make a huge batch
    if batch_size > max(len(refs), len(ques)):
         batch_size = max(len(refs), len(ques))

    scores_obj = calculate_scores(
        refs, ques, 
        similarity_function=CudaCosineGreedy(
            tolerance=tolerance,
            mz_power=mz_power,
            intensity_power=intensity_power,
            shift=shift,
            batch_size=batch_size,
            n_max_peaks=n_max_peaks,
            match_limit=match_limit,
            sparse_threshold=sparse_threshold
        ),
        array_type=array_type
    )

    score_vis = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)

    fig, axs = plt.subplots(1, 2,
                            figsize=(10, 5), 
                            dpi=150)
    
    scores = scores_obj.to_array()
    ax = axs[0]
    ax.imshow(scores['CudaCosineGreedy_score'])

    ax = axs[1]
    ax.imshow(scores['CudaCosineGreedy_matches'])

    plt.suptitle("Score and matches")
    plt.savefig(score_vis.name)

    score = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
    np.savez(score.name, scores=scores)


    import pickle
    pickle_ = tempfile.NamedTemporaryFile(suffix='.pickle', delete=False)

    Path(pickle_.name).write_bytes(pickle.dumps(scores_obj))
    return score.name,  score_vis.name, pickle_.name

with gr.Blocks() as demo:
    gr.Markdown("Run Cuda Cosine Greedy on your MGF files.")
    with gr.Row():
        refs = gr.File(label="Upload REFERENCES.mgf",
                       interactive=True,
                               value='tests/data/pesticides.mgf')
        ques = gr.File(label="Upload QUERIES.mgf",
                       interactive=True,
                               value='tests/data/pesticides.mgf')
    with gr.Row():
            tolerance = gr.Slider(minimum=0, maximum=1, value=0.1, label="Tolerance")
            mz_power = gr.Slider(minimum=0, maximum=2, value=0.0, label="mz_power")
            intensity_power = gr.Slider(minimum=0, maximum=2, value=1.0, label="Intensity Power")
            shift = gr.Slider(minimum=-10, maximum=10, value=0, label="Shift")
    with gr.Row():
            batch_size = gr.Number(value=2048, label="Batch Size", info='How many spectra to process pairwise, in one step. Limited by GPU size, default works well for the T4 GPU.')
            n_max_peaks = gr.Number(value=1024, label="Maximum Number of Peaks", 
                                    info="Some spectra are too large to fit on GPU,"
                                        "so we have to trim them to only use the first "
                                        "n_max_peaks number of peaks.")
            match_limit = gr.Number(value=2048, label="Match Limit", 
                                    info="Two very similar spectra of size N and M can have N * M matches, before filtering."
                                         "This doesn't fit on GPU, so we stop accumulating more matches once we have at most match_limit number of them."
                                         "In practice, a value of 2048 gives more than 99.99% accuracy on GNPS")
    with gr.Row():
            array_type = gr.Radio(['numpy', 'sparse'], value='numpy', type='value',
                                     label='How to handle outputs - if sparse, everything with score less than sparse_threshold will be discarded. If `numpy`, we disable sparse behaviour.')
            sparse_threshold = gr.Slider(minimum=0, maximum=1, value=0.75, label="Sparse Threshold",
                                         info="For very large results, when comparing, more than 10k x 10k, the output dense score matrix can grow too large for RAM."
                                            "While most of the scores aren't useful (near zero). This argument discards all scores less than sparse_threshold, and returns "
                                            "results as a SparseStack format."
                                            )
    with gr.Row():
        score_vis = gr.Image()

    with gr.Row():
        out_npz = gr.File(label="Download similarity matrix (.npz format)", 
                      interactive=False)
        out_pickle = gr.File(label="Download full `Scores` object (.pickle format)", 
                      interactive=False)
    btn = gr.Button("Run")
    btn.click(fn=run, inputs=[refs, ques, tolerance, mz_power, intensity_power, shift, 
                              batch_size, n_max_peaks, match_limit, 
                              array_type, sparse_threshold], outputs=[out_npz, score_vis, out_pickle])

if __name__ == "__main__":
    demo.launch(debug=True)