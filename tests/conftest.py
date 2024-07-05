import warnings
import pytest
from joblib import Memory


memory = Memory(location="cache")


def pytest_configure(config):
    config.addinivalue_line("markers", "performance")


@pytest.fixture(autouse=True, scope="session")
def warn_on_no_cuda():
    import os
    import numba

    if not numba.cuda.is_available():
        warnings.warn(
            "CUDA was unavailable - consider using `NUMBA_ENABLE_CUDASIM=1 pytest <same args, if any>` to simulate having GPU and cudatoolkit for testing purposes"
        )
    yield


@pytest.fixture(autouse=True, scope="session")
def ignore_warnings():
    import os
    os.environ["NUMBA_DISABLE_PERFORMANCE_WARNINGS"] = "1"
    yield


@pytest.fixture(scope="session")
@memory.cache(verbose=0)
def gnps():
    from cudams.utils import download
    from matchms.importing import load_from_mgf

    spectra = tuple(load_from_mgf(download("GNPS-random-10k.mgf")))
    return spectra

@pytest.fixture(scope="session")
@memory.cache
def gnps_with_fingerprint(gnps: list):
    from matchms.filtering import add_fingerprint

    spectra = [add_fingerprint(s) for s in gnps]
    return spectra
