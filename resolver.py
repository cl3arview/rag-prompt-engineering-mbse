import re
from pathlib import Path
from rapidfuzz import process, fuzz
import networkx as nx

XMI_NS_URI = "http://www.omg.org/XMI"
ID_ATTRS = [f"{{{XMI_NS_URI}}}id", "id"]
TAG_RE = re.compile(r"\[S[a-f0-9]{6}\]", re.I)

def build_name_index(G: nx.DiGraph) -> dict[str, list[str]]:
    """
    Build a mapping from lowercase `name` → list of node_ids.
    """
    idx: dict[str, list[str]] = {}
    for nid, attrs in G.nodes(data=True):
        key = attrs["name"].lower()
        idx.setdefault(key, []).append(nid)
    return idx

def fuzzy_candidates(query: str, choices: dict[str, str], top_k: int = 5, score_cutoff: int = 80):
    """
    Yield (node_id, score) for names closest to `query` via RapidFuzz.
    """
    for nid, score, _ in process.extract(
        query, choices, scorer=fuzz.token_set_ratio, limit=top_k
    ):
        if score >= score_cutoff:
            yield nid, score

def resolve_entity(entity: str, name_index: dict[str, list[str]], choices: dict[str, str], *, fuzzy: bool = True) -> list[str]:
    """
    Resolve an entity string to Capella node IDs.
      1. Case-insensitive exact match
      2. Fuzzy match (optional)
    """
    key = entity.lower()
    if key in name_index:
        return name_index[key]
    if fuzzy:
        return [nid for nid, _ in fuzzy_candidates(entity, choices)]
    return []

def slice_xml(node_attrs: dict, context_lines: int = 0) -> str:
    """
    Return the raw XML block for a Capella element, reading only needed lines.
    """
    path = Path(node_attrs["file"])
    lines = path.read_text(encoding="utf-8").splitlines()
    start = max(node_attrs["line"] - 1 - context_lines, 0)
    tag = node_attrs["tag"]
    open_t, close_t = f"<{tag}", f"</{tag}>"
    depth = 0
    end = None
    for i, ln in enumerate(lines[node_attrs["line"] - 1:], start=node_attrs["line"]):
        if open_t in ln:
            depth += ln.count(open_t)
        if close_t in ln:
            depth -= ln.count(close_t)
            if depth <= 0:
                end = i
                break
    end = end if end is not None else len(lines) - 1
    snippet = "\n".join(lines[start : end + 1])
    return snippet

def extract_tags(text: str) -> list[str]:
    """
    Return unique [Sxxxxx] tags in order of first appearance.
    """
    seen, ordered = set(), []
    for match in TAG_RE.findall(text):
        tag = match.strip("[]")
        if tag not in seen:
            ordered.append(tag)
            seen.add(tag)
    return ordered

def slice_relevant_xml(nid: str, G: nx.DiGraph, max_len: int = 600) -> str:
    """
    Minify and skip layout-only tags for prompt use, truncate to max_len.
    """
    raw = slice_xml(G.nodes[nid])
    # skip if purely layout metadata
    if any(t in raw for t in ("ownedDiagrams", "layoutData", "filters")):
        return ""
    xml = re.sub(r"\s+", " ", raw).strip()
    return xml[:max_len] + ("…" if len(xml) > max_len else "")
