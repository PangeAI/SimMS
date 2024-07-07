import pytest
import numpy as np
from click.testing import CliRunner
from simms.cli import main  # Import the main function from your CLI script
from matchms.importing import scores_from_pickle, load_spectra
from matchms.similarity import CosineGreedy
from matchms import calculate_scores

# You might need to adjust the import path based on where `main` is defined

def test_run_with_valid_files():
    runner = CliRunner()
    result = runner.invoke(main, [
        '--references','tests/data/pesticides.mgf',  # Provide a valid path to a reference file
        '--queries','tests/data/pesticides.mgf',  # Provide a valid path to a query file
        '--output_file','data/tmp.pickle',
        '--tolerance', '0.1',
        '--mz_power', '1',
        '--intensity_power', '1',
        '--batch_size', '256',
        '--n_max_peaks', '512',
        '--match_limit', '1024',
        '--array_type', 'numpy',
        '--sparse_threshold', '0.5',
        '--method', 'CudaCosineGreedy'
    ])

    assert result.exit_code == 0
    assert "Scores saved at:" in result.output

    # Load the scores from the output file
    scores_np = scores_from_pickle('data/tmp.pickle').to_array()

    # Load spectra from files
    refs = list(load_spectra('tests/data/pesticides.mgf'))
    ques = refs

    # Compute scores using matchms.similarity.CosineGreedy
    similarity_function = CosineGreedy(
        tolerance=0.1,
        mz_power=1,
        intensity_power=1,
    )
    scores_direct = calculate_scores(refs, ques, similarity_function=similarity_function, array_type="numpy").to_array()

    # Compare shapes
    scores_np = scores_np['CudaCosineGreedy_score']
    scores_direct = scores_direct['CosineGreedy_score']

    assert scores_np.shape == scores_direct.shape
    # Compare content (allowing for numerical tolerance if necessary)
    np.testing.assert_allclose(scores_np, scores_direct, atol=1e-3)