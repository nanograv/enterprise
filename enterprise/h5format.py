"""Tools for defining a file format for storing Python information in HDF5.

This module is intended to be used declaratively: you define a file format
by specifying the attributes of a Python object, their names in HDF5, and
some other information about how they should be read and written. This
format is then an H5Format object, which can be queried for a detailed
description of the file format, suitable for inclusion in a README file.
The H5Format object also has methods to allow you to write the information
from a suitable Python object out into an HDF5 file in the described
format, and to allow you to read the information from a suitable HDF5
format into a Python object.

These formats are aimed at relatively small HDF5 files, where the data
is loaded into numpy arrays rather than being read on the fly from the
file (using HDF5 indexing).

"""
import logging
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from textwrap import dedent, indent, shorten
from typing import IO, Any, Callable, List, Optional, Union

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
    """An entry in an HDF5 file.

    This object describes an entry in an HDF5 file - that is,
    an instance of this class can specify that a certain attribute
    of a Python object should be written out to an HDF5 dataset
    or attribute, and how.

    There is a derived class, H5ConstantEntry, suitable for
    entries that write a particular value to the file format
    regardless of the contents of the Python object (for
    example, a format name).

    Parameters
    ----------
    name : str
        The name to be used for this entry in the HDF5 file.
        This may include spaces and need not be a valid
        Python identifier.
    description : str
        A brief description of the file format. This will
        be included in the part of the file format description
        corresponding to this entry. Indentation will be
        removed (by textwrap.dedent).
    required : bool, optional
        If True, raise an exception if this entry is not present
        in the Python object on write, or in the HDF5 file on
        read. If False, missing entries are simply not
        written.
    use_dataset : bool, optional
        If True, save the data corresponding to this entry
        in an HDF5 dataset. This is appropriate for any large
        object; the data will be compressed on write. If
        False, save the data in an attribute of the HDF5
        file. This should only be used for small items,
        often ones describing the file as a whole; strings
        in particular are more easily handled here.
    attribute : str, optional
        If supplied, this is the attribute on the Python
        object that should contain the entry. If not
        supplied, the attribute will be assumed to
        be the same as the HDF5 name.
    write : callable, optional
        If supplied, the callable is called as
        write(h5file, name, thing, attribute),
        where h5file is an open HDF5 file object.
        The callable should normally extract the
        named attribute from the provided thing
        and save it to the HDF5 file, but it may,
        for example, look up other attributes and
        save their data as attributes of a dataset.
        If this returns a dataset, a decription
        of the entry will be attached as an
        attribute.
    read : callable, optional
        If supplied, the callable is called as
        read(h5file, name, thing, attribute),
        where h5file is an open HDF5 file object.
        This callable should normally extract
        the information from the HDF5 dataset or
        attribute specified by name and save it
        as the named attribute of thing. It may
        extract auxiliary information and do
        save it in thing as well.
    extra_attributes : dict, optional
        If provided, this should be a dictionary of
        additional things to save as attributes
        of the created dataset. It is an error
        to supply this when use_dataset is False.
        Note that extra attributes supplied this
        way must be the same for all files in this
        format; if you want to store attributes
        that depend on the thing being saved,
        you will need to use read and write, above.
    """

    name: str
    description: str
    required: bool = True
    use_dataset: bool = False
    attribute: Optional[str] = None
    write: Optional[Callable] = None
    read: Optional[Callable] = None
    extra_attributes: Optional[dict] = None

    def write_to_hdf5(self, h5file: h5py.File, thing):
        """Write this entry to an HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            An open HDF5 file to write to.
        thing : object
            The relevant data is extracted from this object.
        """
        attribute = self.name if self.attribute is None else self.attribute
        if not hasattr(thing, attribute):
            if self.required:
                raise MissingAttribute(f"Attribute {attribute} needed for HDF5 {self.name} missing from {thing}")
            logger.debug(f"Not writing {self.name} because attribute {attribute} is missing")
            return
        if self.write is not None:
            n = self.write(h5file, self.name, thing, attribute)
            if n is not None:
                n.attrs["Description"] = shorten(self.description, 256)
            return
        self._write_value_to_hdf5(h5file, getattr(thing, attribute))

    def _write_value_to_hdf5(self, h5file: h5py.File, value: Any):
        if self.use_dataset:
            if isinstance(value, dict):
                new_item = write_dict_to_hdf5(h5file, self.name, value)
            elif isinstance(value, str):
                new_item = write_string_to_hdf5_dataset(h5file, self.name, value)
            else:
                new_item = write_array_to_hdf5_dataset(h5file, self.name, np.asarray(value))
            new_item.attrs["Description"] = shorten(self.description, 256)
            if self.extra_attributes is not None:
                new_item.attrs.update(self.extra_attributes)
        else:
            if isinstance(value, np.ndarray) and value.dtype.kind == "U":
                if len(value.shape) != 1:
                    raise ValueError("Cannot store multidimentsional string arrays in HDF5 attributes: {value.shape}")
                value = [str(v) for v in value]
            try:
                h5file.attrs[self.name] = value
            except TypeError as e:
                raise TypeError(f"Invalid type for storage in an attribute: {type(value)}") from e
            if self.extra_attributes is not None:
                raise ValueError(f"Cannot apply extra attributes to attribute: {self.extra_attributes}")

    def read_from_hdf5(self, h5file: h5py.File, thing):
        """Read this entry from an HDF5 file.

        Parameters
        ----------
        h5file : h5py.File
            An open HDF5 file to read the value from.
        thing : object
            Attributes of this object are set to the values extracted
            from the HDF5 file.        logger.debug(f"Creating non-string dataset {name}")

        """
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
                if isinstance(value, np.ndarray) and value.dtype == "O" and len(value.shape) == 1:
                    value = list(value)
        except KeyError:
            if self.required:
                raise
            return
        else:
            setattr(thing, attribute, value)

    def write_description(self, f: IO[str], extra_tags: Optional[List[str]] = None):
        """Write out a description of this entry.

        This will give the name and properties of this entry in
        a format suitable for inclusion in the overall description
        of the file format. In particular each entry produces one
        item in a Markdown list.

        Parameters
        ----------
        f : file
            An open file to write the description to.
        extra_tags : list of str, optional
            If provided, each item in the list will be
            added to the list of tags on the first line
            of the description.
        """
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
    """An H5Entry that always stores the same value.

    This kind of entry is constructed with a constant value
    that is stored in every file in this format. This
    is suitable for format names, version numbers, or
    other metadata about the format itself.

    Upon read, the loaded data is stored into the indicated
    attribute of the Python object, and not checked against
    the constant value; users who want to do validation
    should check the values themselves.
    """

    value: Any = None

    def write_to_hdf5(self, h5file: h5py.File, thing):
        self._write_value_to_hdf5(h5file, self.value)

    def write_description(self, f: IO[str], extra_tags: Optional[List[str]] = None):
        tags = [f"constant value={repr(self.value)}"]
        if extra_tags is not None:
            tags.extend(extra_tags)
        super().write_description(f, extra_tags=tags)


def write_array_to_hdf5_dataset(h5group: h5py.Group, name: str, value: np.ndarray) -> h5py.Dataset:
    """Write an array to an HDF5 dataset.

    For numeric arrays, this just creates a dataset and saves the value.

    For strings, the string is split at newline characters, the lines
    are encoded in UTF-8, and the result is written as a (fixed-length)
    byte array. An attribute "lines" is set to True to indicate that
    this contains the lines of a single string. Attributes
    "coding" and "type" are set to "utf-8" and "str' respectively.

    For arrays of strings, the strings are converted to UTF-8 (but not
    further split) and stored as a fixed-length byte array. Attributes
    "coding" and "type" are set to "utf-8" and "str' respectively.

    The dataset is constructed with gzip compression and the shuffle filter,
    unless the value is a simple scalar.
    """
    if value.dtype.kind == "U":
        logger.debug(f"Encoding {name} {value} as dataset in utf-8")
        encoded = True
        value = np.char.encode(value, "utf-8")
    else:
        encoded = False
    if value.shape == ():
        logger.debug(f"Creating string dataset {name}")
        d = h5group.create_dataset(
            name,
            data=value,
            track_order=True,
        )
    else:
        logger.debug(f"Creating non-string dataset {name}")
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
    return d


def write_string_to_hdf5_dataset(h5group: h5py.Group, name: str, value: str) -> h5py.Dataset:
    """Write a string to an HDF5 dataset.

    This is for writing single, long, Python strings; it splits them at
    newlines and then saves them as a byte array.
    """
    value_as_array = np.array([s.encode("utf-8") for s in value.split("\n")])
    logger.debug(f"converted {repr(value)} to {repr(value_as_array)}")
    d = write_array_to_hdf5_dataset(h5group, name, value_as_array)
    d.attrs["lines"] = True
    d.attrs["coding"] = "utf-8"
    d.attrs["type"] = "str"
    return d


def decode_array_dataset_if_necessary(dataset: h5py.Dataset) -> Any:
    """Extract the value from a dataset.

    This exists to extract values from an HDF5 dataset, reversing any
    encoding or line splitting that has been done. It relies on the
    attributes "lines", "type", and "coding" of the dataset to
    determine what decoding is appropriate.
    """
    lines = dataset.attrs.get("lines", False)
    type_ = dataset.attrs.get("type", "")
    coding = dataset.attrs.get("coding", "")

    logger.debug(f"Decoding {dataset} with type={type_} coding={coding} lines={lines}")
    if type_ == "str":
        if lines:
            return "\n".join([s.decode(coding) for s in dataset])
        return np.char.decode(dataset, coding)
    return np.array(dataset)


def write_dict_to_hdf5(h5group: h5py.Group, name: str, d: dict) -> h5py.Group:
    """Write a dictionary to an HDF5 file (or group).

    Because HDF5 doesn't have a native notion of a dictionary,
    this uses HDF5 groups to store the data. Thus the keys
    must be strings. Contents of the dictionary are stored
    as attributes if scalars, datasets if arrays, and sub-groups
    if they are dicts.
    """
    g = h5group.create_group(name, track_order=True)
    for k, v in d.items():
        if k == "Description":
            raise ValueError("The value `Description` is special to h5format and cannot be stored in dictionaries.")
        if isinstance(v, dict):
            write_dict_to_hdf5(g, k, v)
        elif isinstance(v, np.ndarray):
            write_array_to_hdf5_dataset(g, k, v)
        else:
            g.attrs[k] = v
    return g


def read_dict_from_hdf5(h5group: h5py.Group) -> dict:
    """Read a dictionary from an HDF5 file (or group).

    This converts HDF5 groups into dictionaries, saving
    their attributes and datasets as values in the dictionary,
    and sub-groups as further dictionaries.
    """
    r = dict(h5group.attrs)
    r.pop("Description", None)
    for k, v in h5group.items():
        r[k] = read_dict_from_hdf5(v) if isinstance(v, h5py.Group) else decode_array_dataset_if_necessary(v)
    return r


class H5Format:
    """An HDF5-based file format.

    These objects exist to describe, write, and read HDF5-based
    file formats. Normally an H5Format is constructed by a
    library, and users can use this object to read and write
    library objects; they can also obtain a Markdown description
    of the format by querying the object.

    Parameters
    ----------
    description_intro : str
        Some text to supply as an introduction. This will be
        run through textwrap.dedent to clean up leading indentation.
    entries : list of H5Entry
        The entries in the file format.
    description_finale : str, optional
        A final bit of text to add to the description.
    format_name : str, optional
        A distinctive name for this file format. This name will
        be checked against files read and a warning will be issued
        if there is a mismatch.
    format_version : str, optional
        A version string defining the current version of the file
        format. This version string should be of the form 1.2.3,
        as a form of semantic versioning. A warning will be issued
        if a file being read is not guaranteed to be readable
        by the rules of semantic versioning (major versions are
        not guaranteed any compatibility, newer minor versions
        should be able to read older minor versions but not
        necessarily vice versa).
    """

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
        """Add a new entry to the end of the list in the format."""
        logger.debug(f"Added entry {entry.name} ({entry.attribute})")
        self.entries.append(entry)

    def write_description(self, f: IO[str]):
        """Write the format description to an open file."""
        f.write(self.description_intro)
        for entry in self.entries:
            entry.write_description(f)
        f.write("\n")
        f.write(self.description_finale)

    def save_to_hdf5(self, h5: Union[Path, str, h5py.Group], thing):
        """Save the information in thing to HDF5.

        Parameters
        ----------
        h5 : str or Path or h5py.File or h5py.Group
            Where to write the file. If a string or path, that is opened
            as an HDF5 file for writing; otherwise an open HDF5 file
            should be provided. It is possible to write to a sub-group
            of an existing file, although the reader will need to
            be provided the appropriate sub-group to work with.
        thing : object
            A python object containing the data to be written out,
            extracting data from its attributes as specified by the
            entries in this file format.
        """
        if not isinstance(h5, h5py.Group):
            # This requests the the file preserve insertion order;
            # it may not work on read: https://github.com/h5py/h5py/issues/1577
            with h5py.File(h5, "w", track_order=True) as f:
                return self.save_to_hdf5(f, thing)
        H5ConstantEntry(
            name="README", description="A text description of the format.", use_dataset=True, value=self.description
        ).write_to_hdf5(h5, thing)
        for entry in self.entries:
            entry.write_to_hdf5(h5, thing)

    def load_from_hdf5(self, h5: Union[Path, str, h5py.Group], thing):
        """Load data from an HDF5 file into an existing Python object.

        This method will emit warnings if the file does not appear to
        be in a compatible format, or raise an exception if it cannot
        be read as an HDF5 file at all.

        Parameters
        ----------
        h5 : str or Path or h5py.File or h5py.Group
            Where to read the data. If a string or path, that is opened
            as an HDF5 file for reading; otherwise an open HDF5 file
            should be provided. It is possible to read from a sub-group
            of an existing file.
        thing : object
            A Python object to write the data to. Note that the Python
            type "object" is unsuitable as it cannot accept additional
            attributes, but most Python objects are appropriate here.
        """
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
        """A (long) string describing the file format."""
        f = StringIO()
        self.write_description(f)
        return f.getvalue()

    @property
    def all_names(self) -> List[str]:
        """All names that will be written to the HDF5 file.

        This can be used to look for other information, not used by this format,
        that has been stored in the HDF5 file.
        """
        return ["README"] + [e.name for e in self.entries]
