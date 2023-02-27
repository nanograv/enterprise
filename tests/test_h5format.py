from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("h5py", reason="h5py not available")
import h5py  # noqa: E402
import numpy as np  # noqa: E402

from enterprise.h5format import H5Format, H5Entry, H5ConstantEntry, MissingAttribute, MissingName  # noqa: E402


@dataclass
class Thing:
    an_entry: Any = None


@pytest.fixture
def simple_format() -> H5Format:
    return H5Format(
        description_intro="""\
            # Simple testing format

            This is an HDF5 "format" designed to exercise the basic tools.
            """,
        description_finale="""\
            ## Final notes

            You can add some ending comments here.
            """,
        format_name="simple_format",
        format_version="1.0.0",
    )


def test_description_intro(simple_format: H5Format):
    assert simple_format.description.startswith("# Simple testing format")


def test_description_finale(simple_format: H5Format):
    assert "\n## Final notes" in simple_format.description


def test_description_includes_entry(simple_format: H5Format):
    simple_format.add_entry(H5Entry(name="an_entry", description="This is a sample entry."))
    assert "an_entry" in simple_format.description


def test_string_entry_description():
    e = H5Entry(
        name="the_name",
        description="""\
        Quick testing entry.
        """,
    )
    f = StringIO()
    e.write_description(f)
    d = f.getvalue()
    print(d)
    assert d.startswith("* `the_name` ")
    assert "attribute" in d
    assert "\n    Quick" in d
    assert d.endswith("\n")


def test_string_entry_description_dataset():
    e = H5Entry(
        name="the_name",
        description="""\
        Quick testing entry.
        """,
        use_dataset=True,
    )
    f = StringIO()
    e.write_description(f)
    d = f.getvalue()
    print(d)
    assert "dataset" in d


def test_write_succeeds(tmp_path: Path, simple_format: H5Format):
    h5path = tmp_path / "test.hdf5"

    simple_format.add_entry(
        H5Entry(
            name="an_entry",
            description="This is a sample entry.",
        )
    )

    thing = Thing()
    thing.an_entry = "fish"
    simple_format.save_to_hdf5(h5path, thing)


@pytest.mark.parametrize(
    "value",
    [
        "fish",
        7,
        7.0,
        np.float16(2.3),
        np.float32(2.5),
        np.float64(1.7),
    ],
)
def test_write_read_scalar(
    tmp_path: Path,
    simple_format: H5Format,
    value: Any,
):
    h5path = tmp_path / "test.hdf5"
    simple_format.add_entry(
        H5Entry(
            name="an_entry",
            description="This is a sample entry.",
        )
    )

    thing = Thing()
    thing.an_entry = value
    simple_format.save_to_hdf5(h5path, thing)

    recovered_thing = Thing()
    simple_format.load_from_hdf5(h5path, recovered_thing)

    assert recovered_thing.an_entry == thing.an_entry


@pytest.mark.parametrize(
    "value",
    [
        "fish",
        7,
        7.0,
        np.float16(2.3),
        np.float32(2.5),
        np.float64(1.7),
    ],
)
def test_write_read_scalar_dataset(
    tmp_path: Path,
    simple_format: H5Format,
    value: Any,
):
    h5path = tmp_path / "test.hdf5"
    simple_format.add_entry(
        H5Entry(
            name="an_entry",
            use_dataset=True,
            description="This is a sample entry.",
        )
    )

    thing = Thing()
    thing.an_entry = value
    simple_format.save_to_hdf5(h5path, thing)

    recovered_thing = Thing()
    simple_format.load_from_hdf5(h5path, recovered_thing)

    assert recovered_thing.an_entry == thing.an_entry


def test_write_read_str(
    tmp_path: Path,
    simple_format: H5Format,
):
    value = "a string"
    h5path = tmp_path / "test.hdf5"
    simple_format.add_entry(
        H5Entry(
            name="an_entry",
            description="This is a sample entry.",
        )
    )

    thing = Thing()
    thing.an_entry = value
    simple_format.save_to_hdf5(h5path, thing)

    recovered_thing = Thing()
    simple_format.load_from_hdf5(h5path, recovered_thing)

    assert isinstance(recovered_thing.an_entry, str)
    assert recovered_thing.an_entry == thing.an_entry


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        ["cod", "plaice"],
        [1.7, 2.3, 0.1],
        np.zeros((2, 3, 4)),
        np.array(["cod", "plaice"]),
    ],
)
def test_write_read_vector(
    tmp_path: Path,
    simple_format: H5Format,
    value: Any,
):
    h5path = tmp_path / "test.hdf5"
    simple_format.add_entry(
        H5Entry(
            name="an_entry",
            description="This is a sample entry.",
        )
    )

    thing = Thing()
    thing.an_entry = value
    simple_format.save_to_hdf5(h5path, thing)

    recovered_thing = Thing()
    simple_format.load_from_hdf5(h5path, recovered_thing)

    assert np.array_equal(recovered_thing.an_entry, thing.an_entry)
    if isinstance(recovered_thing.an_entry, np.ndarray) and isinstance(thing.an_entry, np.ndarray):
        assert recovered_thing.an_entry.dtype == thing.an_entry.dtype


@pytest.mark.parametrize(
    "value",
    [
        {1: 2, 3: 4},
        dict(fish=1, fowl=2),
        dict(fish="cod", fowl="pheasant"),
    ],
)
def test_write_read_dict_raises(tmp_path: Path, simple_format: H5Format, value: Any):
    h5path = tmp_path / "test.hdf5"
    simple_format.add_entry(
        H5Entry(
            name="an_entry",
            description="This is a sample entry.",
        )
    )

    thing = Thing()
    thing.an_entry = value
    with pytest.raises(TypeError) as e:
        simple_format.save_to_hdf5(h5path, thing)
    assert "dict" in str(e.value)


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        [1.7, 2.3, 0.1],
        np.zeros((2, 3, 4)),
        ["english", "français", "nederlandse"],
        np.array(["cod", "plaice"]),
    ],
)
def test_write_read_vector_dataset(
    tmp_path: Path,
    simple_format: H5Format,
    value: Any,
):
    h5path = tmp_path / "test.hdf5"
    simple_format.add_entry(
        H5Entry(
            name="an_entry",
            description="This is a sample entry.",
            use_dataset=True,
        )
    )

    thing = Thing()
    thing.an_entry = value
    simple_format.save_to_hdf5(h5path, thing)

    recovered_thing = Thing()
    simple_format.load_from_hdf5(h5path, recovered_thing)

    assert np.array_equal(recovered_thing.an_entry, thing.an_entry)
    assert isinstance(recovered_thing.an_entry, np.ndarray)


@pytest.mark.parametrize(
    "value",
    [
        "fish",
        "fish\n",
        "fish\nfowl",
        "fish\nfowl\n",
        """
        Long message with random whitespace and some français.
        """,
    ],
)
def test_write_read_string_dataset(
    tmp_path: Path,
    simple_format: H5Format,
    value: Any,
):
    h5path = tmp_path / "test.hdf5"
    simple_format.add_entry(
        H5Entry(
            name="an_entry",
            description="This is a sample entry.",
            use_dataset=True,
        )
    )

    thing = Thing()
    thing.an_entry = value
    simple_format.save_to_hdf5(h5path, thing)

    recovered_thing = Thing()
    simple_format.load_from_hdf5(h5path, recovered_thing)

    assert isinstance(recovered_thing.an_entry, str)
    assert recovered_thing.an_entry == thing.an_entry


@pytest.mark.parametrize(
    "value",
    [
        dict(fish=1, fowl=2),
        dict(fish="cod", fowl="pheasant"),
        dict(fish=[1, 2, 3], fowl=dict(hare=1, hounds="dogs")),
        dict(fish=np.array([1, 2, 3]), fowl=dict(hare=1, hounds="dogs")),
        dict(fish=np.array(["a", "b", "c"]), fowl=dict(hare=1, hounds="dogs")),
    ],
)
def test_write_read_dict_dataset(tmp_path: Path, simple_format: H5Format, value: Any):
    h5path = tmp_path / "test.hdf5"
    simple_format.add_entry(
        H5Entry(
            name="an_entry",
            description="This is a sample entry.",
            use_dataset=True,
        )
    )

    thing = Thing()
    thing.an_entry = value
    simple_format.save_to_hdf5(h5path, thing)

    recovered_thing = Thing()
    simple_format.load_from_hdf5(h5path, recovered_thing)

    def compare_dicts(d1, d2):
        assert set(d1.keys()) == set(d2.keys())
        for k, v1 in d1.items():
            v2 = d2[k]
            if isinstance(v1, dict):
                assert isinstance(v2, dict)
                compare_dicts(v1, v2)
            elif isinstance(v1, np.ndarray):
                assert np.array_equal(v1, v2)
            else:
                assert v1 == v2

    compare_dicts(recovered_thing.an_entry, thing.an_entry)


def test_write_missing(simple_format, tmp_path):
    h5path = tmp_path / "test.hdf5"
    thing = Thing()
    simple_format.add_entry(H5Entry(name="another_entry", description="", required=False))
    simple_format.save_to_hdf5(h5path, thing)
    # Should not raise an exception


def test_write_missing_required(simple_format, tmp_path):
    h5path = tmp_path / "test.hdf5"
    thing = Thing()
    simple_format.add_entry(H5Entry(name="another_entry", description=""))
    with pytest.raises(MissingAttribute) as e:
        simple_format.save_to_hdf5(h5path, thing)
    assert "another_entry" in str(e.value)


def test_read_missing(simple_format, tmp_path):
    h5path = tmp_path / "test.hdf5"
    thing = Thing()
    simple_format.save_to_hdf5(h5path, thing)
    simple_format.add_entry(H5Entry(name="another_entry", description="", required=False))
    another_thing = Thing()
    simple_format.load_from_hdf5(h5path, another_thing)
    assert not hasattr(another_thing, "another_entry")


def test_read_missing_required(simple_format, tmp_path):
    h5path = tmp_path / "test.hdf5"
    thing = Thing()
    simple_format.save_to_hdf5(h5path, thing)
    simple_format.add_entry(H5Entry(name="another_entry", description=""))
    another_thing = Thing()
    with pytest.raises(MissingName) as e:
        simple_format.load_from_hdf5(h5path, another_thing)
    assert "another_entry" in str(e.value)


def test_no_stray_entries(simple_format, tmp_path):
    h5path = tmp_path / "test.hdf5"
    simple_format.add_entry(
        H5Entry(
            name="an_entry",
            description="This is a sample entry.",
        )
    )

    thing = Thing()
    thing.an_entry = "a value"
    simple_format.save_to_hdf5(h5path, thing)

    with h5py.File(h5path, "r") as f:
        names = set(simple_format.all_names)
        attr_names = set(f.attrs.keys())
        data_group_names = set(f.keys())
        assert not attr_names.intersection(data_group_names)
        assert names == attr_names.union(data_group_names)


@pytest.mark.parametrize("value", [17, "fish"])
def test_constant_write(simple_format, tmp_path, value):
    h5path = tmp_path / "test.hdf5"
    simple_format.add_entry(
        H5ConstantEntry(
            name="a_constant_entry",
            description="This is a constant entry.",
            value=value,
        )
    )
    assert "constant" in simple_format.description
    assert str(value) in simple_format.description

    thing = Thing()

    simple_format.save_to_hdf5(h5path, thing)

    another_thing = Thing()
    simple_format.load_from_hdf5(h5path, another_thing)

    assert hasattr(another_thing, "a_constant_entry")
    assert another_thing.a_constant_entry == value


def test_warn_format_name(simple_format, tmp_path, caplog):
    h5path = tmp_path / "test.hdf5"

    thing = Thing()
    simple_format.save_to_hdf5(h5path, thing)

    other_format = H5Format(
        description_intro="",
        description_finale="",
        format_name="other_format",
        format_version="1.0.0",
    )

    another_thing = Thing()
    other_format.load_from_hdf5(h5path, another_thing)

    print(caplog.records)
    assert "different formats" in caplog.records[0].msg
    assert "simple_format" in caplog.records[0].msg
    assert "other_format" in caplog.records[0].msg


def test_warn_major_format(simple_format, tmp_path, caplog):
    h5path = tmp_path / "test.hdf5"

    thing = Thing()
    simple_format.save_to_hdf5(h5path, thing)

    other_format = H5Format(
        description_intro="",
        description_finale="",
        format_name="simple_format",
        format_version="2.0.0",
    )

    another_thing = Thing()
    other_format.load_from_hdf5(h5path, another_thing)

    print(caplog.records)
    assert "major version" in caplog.records[0].msg
    assert "1.0.0" in caplog.records[0].msg
    assert "2.0.0" in caplog.records[0].msg


def test_no_warn_minor_format_newer(simple_format, tmp_path, caplog):
    h5path = tmp_path / "test.hdf5"

    thing = Thing()

    other_format = H5Format(
        description_intro="",
        description_finale="",
        format_name="simple_format",
        format_version="1.1.0",
    )

    another_thing = Thing()

    simple_format.save_to_hdf5(h5path, thing)
    other_format.load_from_hdf5(h5path, another_thing)

    print(caplog.records)
    assert not caplog.records


def test_warn_minor_format_older(simple_format, tmp_path, caplog):
    h5path = tmp_path / "test.hdf5"

    thing = Thing()

    other_format = H5Format(
        description_intro="",
        description_finale="",
        format_name="simple_format",
        format_version="1.1.0",
    )

    another_thing = Thing()

    other_format.save_to_hdf5(h5path, thing)
    simple_format.load_from_hdf5(h5path, another_thing)

    print(caplog.records)
    assert "minor version" in caplog.records[0].msg
    assert "1.0.0" in caplog.records[0].msg
    assert "1.1.0" in caplog.records[0].msg


def test_constant_extra_tags():
    f = StringIO()
    e = H5ConstantEntry(
        name="fish",
        description="One of a number of aquatic species",
        value="cod",
    )
    e.write_description(f, extra_tags=["definition"])

    assert "fish" in f.getvalue()
    assert "aquatic" in f.getvalue()
    assert "definition" in f.getvalue()
    assert "cod" in f.getvalue()


@pytest.mark.parametrize(
    "extras",
    [
        dict(),
        dict(units="s"),
        dict(fish="carp", fowl=17.3),
    ],
)
def test_extra_attributes_dataset(simple_format, tmp_path, extras):
    h5path = tmp_path / "test.hdf5"
    simple_format.add_entry(
        H5Entry(
            name="an_entry",
            description="This is a sample entry.",
            use_dataset=True,
            extra_attributes=extras,
        )
    )

    thing = Thing()
    thing.an_entry = 17
    simple_format.save_to_hdf5(h5path, thing)

    with h5py.File(h5path, "r") as f:
        attrs = dict(f["an_entry"].attrs)
        assert "sample" in attrs.pop("Description")
        assert attrs == extras
