import json
from pathlib import Path
import networkx as nx
from lxml import etree

XMI_NS_URI = "http://www.omg.org/XMI"
ID_ATTRS = [f"{{{XMI_NS_URI}}}id", "id"]

def iter_capella_elements(xml_path: Path):
    """
    Yield each end-event element from a Capella XML file, clearing parsed tree
    to cap memory use.
    """
    for _, elem in etree.iterparse(str(xml_path), events=("end",)):
        yield elem
        elem.clear()
        parent = elem.getparent()
        if parent is not None:
            while elem.getprevious() is not None:
                del parent[0]

def get_node_id(elem: etree._Element) -> str:
    """
    Return first non-empty XMI/plain `id` attribute, or None.
    """
    for attr in ID_ATTRS:
        node_id = elem.get(attr)
        if node_id:
            return node_id
    return None

def build_network(xml_path: Path) -> nx.DiGraph:
    """
    Build a directed graph of containment from a .capella XML.
    Nodes carry tag, name, file, line, description.
    Edges are parent→child with type='contains'.
    """
    G = nx.DiGraph()
    for elem in iter_capella_elements(xml_path):
        node_id = get_node_id(elem)
        if not node_id:
            continue
        G.add_node(
            node_id,
            tag         = etree.QName(elem.tag).localname,
            name        = elem.get("name", ""),
            file        = str(xml_path.resolve()),
            line        = elem.sourceline,
            description = elem.get("description", "")
        )
        parent = elem.getparent()
        parent_id = get_node_id(parent) if parent is not None else None
        if parent_id:
            G.add_edge(parent_id, node_id, type="contains")
    return G

def save_network_json(G: nx.DiGraph, out_path: Path):
    """
    Dump graph to node-link JSON for debugging/inspection.
    """
    data = nx.readwrite.json_graph.node_link_data(G)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
