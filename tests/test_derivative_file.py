import tempfile
from pathlib import Path

import packaging.version
import pytest
from numpy.testing import assert_array_equal

pytest.importorskip("enterprise.derivative_file", reason="HDF5 not available")
import enterprise.pulsar  # noqa: E402
from enterprise.derivative_file import FilePulsar, format_version  # noqa: E402
from tests.enterprise_test_data import datadir as datadir_str  # noqa: E402

datadir = Path(datadir_str)


@pytest.fixture(scope="module", params=["tempo2", "pint"])
def psr_roundtrip(request):
    timing_package = request.param
    with tempfile.TemporaryDirectory(prefix="enterprise-testing-") as d:
        h5path = Path(d) / "test.hdf5"
        if timing_package == "pint":
            if enterprise.pulsar.pint is None:
                pytest.skip(reason="PINT is not available")
            psr = enterprise.pulsar.Pulsar(
                str(datadir / "B1855+09_NANOGrav_9yv1.gls.par"),
                str(datadir / "B1855+09_NANOGrav_9yv1.tim"),
                timing_package="pint",
                drop_pintpsr=False,
            )
        if timing_package == "tempo2":
            if enterprise.pulsar.t2 is None:
                pytest.skip(reason="TEMPO2 is not available")
            psr = enterprise.pulsar.Pulsar(
                str(datadir / "B1855+09_NANOGrav_9yv1.gls.par"),
                str(datadir / "B1855+09_NANOGrav_9yv1.tim"),
                timing_package="tempo2",
                drop_t2pulsar=False,
            )
        psr.to_hdf5(h5path)
        new_psr = FilePulsar(h5path)
        yield psr, new_psr


@pytest.mark.parametrize("attribute", ["name", "theta", "phi", "dm", "dmx"])
def test_basic_attributes_unchanged(psr_roundtrip, attribute):
    psr, new_psr = psr_roundtrip
    assert getattr(new_psr, attribute) == getattr(psr, attribute)


@pytest.mark.parametrize("attribute", ["pos", "residuals", "Mmat", "toas", "stoas"])
def test_vector_attributes_unchanged(psr_roundtrip, attribute):
    psr, new_psr = psr_roundtrip
    assert_array_equal(getattr(new_psr, attribute), getattr(psr, attribute))


def test_flags_unchanged(psr_roundtrip):
    psr, new_psr = psr_roundtrip
    assert set(psr.flags.keys()) == set(new_psr.flags.keys())
    for k, v1 in psr.flags.items():
        v2 = new_psr.flags[k]
        assert_array_equal(v2, v1)


def test_format_version_set(psr_roundtrip):
    psr, new_psr = psr_roundtrip
    assert hasattr(new_psr, "format_version")
    assert new_psr.format_version == format_version
    assert isinstance(
        packaging.version.parse(new_psr.format_version),
        packaging.version.Version,
    )


def test_par_tim_files_preserved(psr_roundtrip):
    psr, new_psr = psr_roundtrip
    assert hasattr(new_psr, "parfile")
    assert new_psr.parfile == psr.parfile
    assert hasattr(new_psr, "timfile")
    assert new_psr.timfile == psr.timfile
    assert "PSR" in new_psr.parfile
    assert new_psr.name in new_psr.parfile
    assert "FORMAT 1" in new_psr.timfile
