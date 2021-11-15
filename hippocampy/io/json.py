import os
import json


def read_json(json_path, field=None):
    """
    return a dict with the content of a json file
    """

    # to avoid some bugs
    json_path = os.path.expanduser(json_path)
    # check path exist and if this is a directory
    assert os.path.exists(json_path), print("%s \n does not exist" % json_path)

    # read the file
    with open(json_path) as f:
        data = f.read()

    if field is None:
        return json.loads(data)
    else:
        jf = json.loads(data)

        if isinstance(field, str):
            assert field in jf.keys(), f"Field {field} not fount in json file"
            return jf[field]
        else:

            assert [f in jf.keys() for f in field], "Field not fount in json file"
            return {f: jf[f] for f in field}


def write_json(json_path: str, d: dict, *, overwrite: bool = False):
    """
    Wrapper for json.dump to write a dictionary to a json file.

    Parameters
    ----------
    json_path : str
        path of the json file
    d : dict
        dictionary to write
    overwrite : bool, optional
        if we should overwrite existing files or not, by default False

    """

    json_path = os.path.expanduser(json_path)
    # check path exist and if this is a directory
    if os.path.exists(json_path):
        raise FileExistsError(
            f"File {json_path} already exist, delete this file of use overwrite option"
        )
    if not type(d) == dict:
        raise ValueError(f" The variable {d} should be a dict")
    with open(json_path, "wb") as fio:
        json.dump(d, json_path)

