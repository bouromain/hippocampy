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
