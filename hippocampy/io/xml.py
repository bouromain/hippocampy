import xml.etree.ElementTree as ET
import os


def read_XML_field(fpath, element, attribute=None):
    """
    This function read an element attribute in an xlm file 

    Parameters
    ----------
    fpath: str
        path of the xlm file
    element: str
        element of interest
    attribute: str
        attribute to read

    Returns
    -------
    out: dict
        dictionary {attributes: values}

    Example
    -------
    executing this function on a file containing the following content
    and searching for the element 'box' and attribute 'framerate'
    <rootelement>
        <box channel='1' framerate=23>
        </box>
        <otherbox  port='abc'>
        </otherbox>
        <box channel='123' framerate=36>
        </box>
    <rootelement>
    
    will return:
    {framerate: 23, framerate:36}
    """
    if not os.path.exists(fpath):
        raise FileNotFoundError

    tree = ET.parse(fpath)
    root = tree.getroot()

    if not element.startswith(".//"):
        element = ".//" + element

    element_list = root.findall(element)

    assert len(element_list) > 1, "Element %s not found in xml file" % (element)

    if attribute is None:
        # if no attribute is given return them all with their corresponding value
        return [{k: v for k, v in el.attrib.items()} for el in element_list]
    else:
        # if the attribute is specified only return this one
        return [el.attrib.get(attribute) for el in element_list]
