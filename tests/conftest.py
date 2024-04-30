import warnings
import pytest
from joblib import Memory

memory = Memory(location="cache")

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "performance"
    )

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


@pytest.fixture(autouse=True)
def patch_harmonize_values(monkeypatch):
    """
    Necessary until https://github.com/matchms/matchms/pull/605 gets merged
    """
    from matchms import Metadata
    from matchms.filtering.metadata_processing.add_precursor_mz import \
        _add_precursor_mz_metadata
    from matchms.filtering.metadata_processing.add_retention import (
        _add_retention, _retention_index_keys, _retention_time_keys)
    from matchms.filtering.metadata_processing.interpret_pepmass import \
        _interpret_pepmass_metadata
    from matchms.filtering.metadata_processing.make_charge_int import \
        _convert_charge_to_int

    def harmonize_values(self):
        """Runs default harmonization of metadata.

        This includes harmonizing entries for ionmode, retention time and index,
        charge, as well as the removal of invalid entried ("", "NA", "N/A", "NaN").
        """
        metadata_filtered = _interpret_pepmass_metadata(self.data)
        metadata_filtered = _add_precursor_mz_metadata(metadata_filtered)

        if metadata_filtered.get("ionmode"):
            metadata_filtered["ionmode"] = self.get("ionmode").lower()

        if metadata_filtered.get("retention_time"):
            metadata_filtered = _add_retention(
                metadata_filtered, "retention_time", _retention_time_keys
            )

        if metadata_filtered.get("retention_index"):
            metadata_filtered = _add_retention(
                metadata_filtered, "retention_index", _retention_index_keys
            )

        if metadata_filtered.get("parent"):
            metadata_filtered["parent"] = float(metadata_filtered.get("parent"))

        charge = metadata_filtered.get("charge")
        charge_int = _convert_charge_to_int(charge)
        if not isinstance(charge, int) and charge_int is not None:
            metadata_filtered["charge"] = charge_int

        invalid_entries = ["", "NA", "N/A", "NaN"]

        metadata_filtered_ = {}
        # Necessary to check not isinstance(..., str), since some values are arrays, and `not in`
        # operator results in iterable, that has an ambiguous truth value
        for k, v in metadata_filtered.items():
            if not isinstance(v, str) or v not in invalid_entries:
                metadata_filtered_[k] = v
        self.data = metadata_filtered_

    monkeypatch.setattr(Metadata, "harmonize_values", harmonize_values)
    yield


@pytest.fixture(scope='session')
@memory.cache
def gnps():
    import pickle
    from cudams.utils import download
    spectra = pickle.load(open(download("GNPS-LIBRARY.pickle"), "rb"))
    return spectra

@pytest.fixture(scope='session')
@memory.cache
def gnps_with_fingerprint(gnps: list):
    from matchms.filtering import add_fingerprint
    spectra = [add_fingerprint(s) for s in gnps]
    return spectra
