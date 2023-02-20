import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from textwrap import indent, dedent
from typing import Optional, Callable, List, IO, Union, Any

import h5py
import numpy as np
import packaging.version

logger = logging.getLogger(__name__)


class MissingAttribute(ValueError):
    pass


class MissingName(ValueError):
    pass


@dataclass
class H5Entry:
    name: str
    description: str
    required: bool = True
    use_dataset: bool = False
    attribute: Optional[str] = None
    write: Optional[Callable] = None
    read: Optional[Callable] = None
    extra_attributes: Optional[dict] = None

    def write_to_hdf5(self, h5file: h5py.File, thing):
        attribute = self.name if self.attribute is None else self.attribute
        if not hasattr(thing, attribute):
            if self.required:
                raise MissingAttribute(f"Attribute {attribute} needed for HDF5 {self.name} missing from {thing}")
            logger.debug(f"Not writing {self.name} because attribute {attribute} is missing")
            return
        if self.write is not None:
            return self.write(h5file, self.name, thing, attribute)
        self._write_value_to_hdf5(h5file, getattr(thing, attribute))

    def _write_value_to_hdf5(self, h5file: h5py.File, value: Any):
        if self.use_dataset:
            if isinstance(value, dict):
                write_dict_to_hdf5(h5file, self.name, value)
            elif isinstance(value, str):
                write_string_to_hdf5_dataset(h5file, self.name, value)
            else:
                write_array_to_hdf5_dataset(h5file, self.name, np.asarray(value))
            if self.extra_attributes is not None:
                h5file[self.name].attrs.update(self.extra_attributes)
        else:
            try:
                h5file.attrs[self.name] = value
            except TypeError as e:
                raise TypeError(f"Invalid type for storage in an attribute: {type(value)}") from e
            if self.extra_attributes is not None:
                raise ValueError(f"Cannot apply extra attributes to attribute: {self.extra_attributes}")

    def read_from_hdf5(self, h5file: h5py.File, thing):
        attribute = self.name if self.attribute is None else self.attribute
        if self.name not in (h5file if self.use_dataset else h5file.attrs):
            if self.required:
                raise MissingName(f"Entry {self.name} missing from HDF5")
            logger.debug(f"Not reading {attribute} because attribute {self.name} is missing")
            return
        if self.read is not None:
            return self.read(h5file, self.name, thing, attribute)
        try:
            if self.use_dataset:
                value = h5file[self.name]
                if isinstance(value, h5py.Group):
                    value = read_dict_from_hdf5(value)
                else:
                    value = decode_array_dataset_if_necessary(value)
            else:
                value = h5file.attrs[self.name]
        except KeyError:
            if self.required:
                raise
            return
        else:
            setattr(thing, attribute, value)

    def write_description(self, f: IO[str], extra_tags: Optional[List[str]] = None):
        tags = ["dataset" if self.use_dataset else "attribute"]
        if not self.required:
            tags.append("optional")
        if self.extra_attributes is not None and "units" in self.extra_attributes:
            tags.append(f"units={self.extra_attributes['units']}")
        if extra_tags is not None:
            tags.extend(extra_tags)
        # End with two spaces to arrange for a Markdown line break
        print(f"* `{self.name}` ({', '.join(tags)})  ", file=f)
        print(indent(dedent(self.description).strip(), 4 * " "), file=f)


@dataclass
class H5ConstantEntry(H5Entry):
    value: Any = None

    def write_to_hdf5(self, h5file: h5py.File, thing):
        self._write_value_to_hdf5(h5file, self.value)

    def write_description(self, f: IO[str], extra_tags: Optional[List[str]] = None):
        tags = [f'constant value="{self.value}"']
        if extra_tags is not None:
            tags.extend(extra_tags)
        super().write_description(f, extra_tags=tags)


def write_array_to_hdf5_dataset(h5group: h5py.Group, name: str, value: np.ndarray):
    if value.dtype.kind == "U":
        logger.debug(f"Encoding {name} {value} as dataset in utf-8")
        encoded = True
        value = np.char.encode(value, "utf-8")
    else:
        encoded = False
    if value.shape == ():
        d = h5group.create_dataset(
            name,
            data=value,
            track_order=True,
        )
    else:
        d = h5group.create_dataset(
            name,
            data=value,
            compression="gzip",
            compression_opts=9,
            shuffle=True,
            track_order=True,
        )
    if encoded:
        logger.debug(f"Recording attributes for {name}")
        d.attrs["lines"] = False
        d.attrs["coding"] = "utf-8"
        d.attrs["type"] = "str"


def write_string_to_hdf5_dataset(h5group: h5py.Group, name: str, value: str):
    value_as_array = np.array([s.encode("utf-8") for s in value.split("\n")])
    logger.debug(f"converted {repr(value)} to {repr(value_as_array)}")
    write_array_to_hdf5_dataset(h5group, name, value_as_array)
    h5group[name].attrs["lines"] = True
    h5group[name].attrs["coding"] = "utf-8"
    h5group[name].attrs["type"] = "str"


def decode_array_dataset_if_necessary(dataset: h5py.Dataset) -> Any:
    lines = dataset.attrs.get("lines", False)
    type_ = dataset.attrs.get("type", "")
    coding = dataset.attrs.get("coding", "")

    logger.debug(f"Decoding {dataset} with {type_=} {coding=} {lines=}")
    if type_ == "str":
        if lines:
            return "\n".join([s.decode(coding) for s in dataset])
        return np.char.decode(dataset, coding)
    return np.array(dataset)


def write_dict_to_hdf5(h5group: h5py.Group, name: str, d: dict):
    g = h5group.create_group(name, track_order=True)
    for k, v in d.items():
        if isinstance(v, dict):
            write_dict_to_hdf5(g, k, v)
        elif isinstance(v, np.ndarray):
            write_array_to_hdf5_dataset(g, k, v)
        else:
            g.attrs[k] = v


def read_dict_from_hdf5(h5group: h5py.Group) -> dict:
    r = dict(h5group.attrs)
    for k, v in h5group.items():
        r[k] = read_dict_from_hdf5(v) if isinstance(v, h5py.Group) else decode_array_dataset_if_necessary(v)
    return r


class H5Format:
    def __init__(
        self,
        description_intro: str,
        entries: Optional[List[H5Entry]] = None,
        description_finale: str = "",
        format_name: Optional[str] = None,
        format_version: Optional[str] = None,
    ):
        self.description_intro = dedent(description_intro)
        self.description_finale = dedent(description_finale)
        self.format_name = format_name
        self.format_version = format_version
        self.entries: List[H5Entry] = []
        if self.format_name is not None:
            self.entries.append(
                H5ConstantEntry(
                    name="format_name",
                    required=False,
                    description="The name of this particular HDF5 format.",
                    value=self.format_name,
                )
            )
        if self.format_version is not None:
            if not isinstance(packaging.version.parse(self.format_version), packaging.version.Version):
                raise ValueError(f"Unable to parse format_version {self.format_version}")
            self.entries.append(
                H5ConstantEntry(
                    name="format_version",
                    required=False,
                    description="""\
                        Version number indicating the compatibility of this file with
                        other readers of this format.
                        """,
                    value=self.format_version,
                )
            )
        if entries is not None:
            self.entries.extend(entries)

    def add_entry(self, entry: H5Entry):
        logger.debug(f"Added entry {entry.name} ({entry.attribute})")
        self.entries.append(entry)

    def write_description(self, f: IO[str]):
        f.write(self.description_intro)
        for entry in self.entries:
            entry.write_description(f)
        f.write("\n")
        f.write(self.description_finale)

    def save_to_hdf5(self, h5: Union[Path, str, h5py.Group], thing):
        if not isinstance(h5, h5py.Group):
            # This requests the the file preserve insertion order;
            # it may not work on read: https://github.com/h5py/h5py/issues/1577
            with h5py.File(h5, "w", track_order=True) as f:
                return self.save_to_hdf5(f, thing)
        h5.attrs["README"] = self.description
        for entry in self.entries:
            entry.write_to_hdf5(h5, thing)

    def load_from_hdf5(self, h5: Union[Path, str, h5py.Group], thing):
        if not isinstance(h5, h5py.Group):
            with h5py.File(h5, "r") as f:
                return self.load_from_hdf5(f, thing)
        for entry in self.entries:
            entry.read_from_hdf5(h5, thing)
        if self.format_name is not None and getattr(thing, "format_name") != self.format_name:
            logger.warning(
                f"Apparently different formats: file is in format "
                f"{getattr(thing, 'format_name')} while this reader expects "
                f"format {self.format_name}."
            )
        if self.format_version is not None and hasattr(thing, "format_version"):
            s = packaging.version.parse(self.format_version)
            o = packaging.version.parse(thing.format_version)
            if s.major != o.major:
                logger.warning(
                    f"Incompatible major versions for the format ({self.format_version} and {thing.format_version})"
                )
            if s.minor < o.minor:
                logger.warning(
                    f"File has newer format minor version than reader "
                    f"({self.format_version} and {thing.format_version})"
                )

    @property
    def description(self) -> str:
        f = StringIO()
        self.write_description(f)
        return f.getvalue()

    @property
    def all_names(self) -> List[str]:
        return ["README"] + [e.name for e in self.entries]
