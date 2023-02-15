import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from textwrap import indent, dedent
from typing import Optional, Callable, List, IO, Union

import h5py
import numpy as np

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

    def write_to_hdf5(self, h5file: h5py.File, thing):
        attribute = self.name if self.attribute is None else self.attribute
        if self.write is not None:
            return self.write(h5file, self.name, thing, attribute)
        if not hasattr(thing, attribute):
            if self.required:
                raise MissingAttribute(f"Attribute {attribute} needed for HDF5 {self.name} missing from {thing}")
            logger.debug(f"Not writing {self.name} because attribute {attribute} is missing")
            return
        value = getattr(thing, attribute)
        if self.use_dataset:
            if isinstance(value, dict):
                write_dict_to_hdf5(h5file, self.name, value)
            else:
                value = np.asarray(value)
                if value.dtype.kind == "U":
                    value = np.char.encode(value, "utf-8")
                h5file.create_dataset(
                    self.name,
                    data=value,
                    compression="gzip",
                    compression_opts=9,
                )
        else:
            try:
                h5file.attrs[self.name] = value
            except TypeError as e:
                raise TypeError(f"Invalid type for storage in an attribute: {type(getattr(thing,attribute))}") from e

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
                    print(self.name, type(value))
                    value = np.array(value)
                    print(value.dtype)
                    if value.dtype.kind == "S":
                        value = np.char.decode(value, "utf-8")
            else:
                value = h5file.attrs[self.name]
        except KeyError:
            if self.required:
                raise
            return
        else:
            setattr(thing, attribute, value)

    def write_description(self, f: IO[str]):
        tags = ["dataset" if self.use_dataset else "attribute"]
        if not self.required:
            tags.append("optional")
        print(f"* `{self.name}` ({', '.join(tags)})", file=f)
        print(indent(dedent(self.description).strip(), 4 * " "), file=f)


def write_dict_to_hdf5(h5group: h5py.Group, name: str, d: dict):
    g = h5group.create_group(name)
    for k, v in d.items():
        if isinstance(v, dict):
            write_dict_to_hdf5(g, k, v)
        elif isinstance(v, np.ndarray):
            g.create_dataset(k, data=v, compression="gzip", compression_level=9)
        else:
            g.attrs[k] = v


def read_dict_from_hdf5(h5group: h5py.Group) -> dict:
    r = dict(h5group.attrs)
    for k, v in h5group.items():
        r[k] = read_dict_from_hdf5(v) if isinstance(v, h5py.Group) else np.array(v)
    return r


class H5Format:
    def __init__(self, description_intro: str, entries: Optional[List[H5Entry]] = None, description_finale: str = ""):
        self.description_intro = dedent(description_intro)
        self.description_finale = dedent(description_finale)
        self.entries = [] if entries is None else entries

    def add_entry(self, entry: H5Entry):
        logger.debug(f"Added entry {entry.name} ({entry.attribute})")
        self.entries.append(entry)

    def write_description(self, f: IO[str]):
        f.write(self.description_intro)
        for entry in self.entries:
            entry.write_description(f)
        f.write("\n")
        f.write(self.description_finale)

    def save_to_hdf5(self, h5path: Union[Path, str], thing):
        with h5py.File(h5path, "w") as f:
            f.attrs["description"] = self.description
            for entry in self.entries:
                entry.write_to_hdf5(f, thing)

    def load_from_hdf5(self, h5path: Union[Path, str], thing):
        with h5py.File(h5path, "r") as f:
            for entry in self.entries:
                entry.read_from_hdf5(f, thing)

    @property
    def description(self) -> str:
        f = StringIO()
        self.write_description(f)
        return f.getvalue()