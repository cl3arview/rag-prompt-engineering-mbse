import uuid
import numpy as np
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from resolver import resolve_entity, extract_tags, slice_relevant_xml

class EntityList(BaseModel):
    entities: List[str] = Field(..., description="exact names as they appear in Capella")

class QA(BaseModel):
    question: str
    answer:   str
    sources:  List[str]

class QASet(BaseModel):
    simple_fact:        QA
    simple_conditional: QA
    comparison:         QA
    interpretative:     QA
    multi_answer:       QA
    aggregation:        QA
    multi_hop:          QA
    heavy_post:         QA
    erroneous:          QA
    summary:            QA

def setup_entity_extractor(
    api_key: str,
    api_base: str
) -> Tuple[ChatOpenAI, ChatPromptTemplate, PydanticOutputParser]:
    """
    Initialize LLM, prompt templates, and parser for entity extraction.
    """
    llm = ChatOpenAI(
        model_name      = "google/gemini-2.0-flash-001",
        openai_api_key  = api_key,
        openai_api_base = api_base,
    )
    parser = PydanticOutputParser(pydantic_object=EntityList)
    raw_inst = parser.get_format_instructions()
    safe_inst = raw_inst.replace("{", "{{").replace("}", "}}")
    system_tmpl = SystemMessagePromptTemplate.from_template(
        "You are an assistant that extracts Capella element names from user queries "
        "and returns them **only** as JSON matching this schema:\n\n" + safe_inst
    )
    user_tmpl = HumanMessagePromptTemplate.from_template("{user_query}")
    prompt    = ChatPromptTemplate.from_messages([system_tmpl, user_tmpl])
    return llm, prompt, parser

def setup_qa_llm(
    api_key: str,
    api_base: str
) -> Tuple[ChatOpenAI, PydanticOutputParser]:
    """
    Initialize LLM and parser for generating the QA set.
    """
    llm = ChatOpenAI(
        model_name      = "google/gemini-2.5-pro-preview",
        openai_api_key  = api_key,
        openai_api_base = api_base,
    )
    parser = PydanticOutputParser(pydantic_object=QASet)
    return llm, parser

def generate_qa_set(
    user_query: str,
    G: Any,
    vectordb: Chroma,
    qa_llm: ChatOpenAI,
    qa_parser: PydanticOutputParser,
    extract_llm: ChatOpenAI,
    extract_prompt: ChatPromptTemplate,
    extract_parser: PydanticOutputParser,
    name_index: Dict[str, List[str]],
    choices: Dict[str, str],
    k_pdf: int = 5
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run the full QA pipeline for a single user_query.
    Returns (qa_set_dict, src_map).
    """
    # 1) Extract entities
    chain = extract_prompt | extract_llm | extract_parser
    entities = chain.invoke({"user_query": user_query}).entities

    # 2) Resolve IDs
    resolved = {e: resolve_entity(e, name_index, choices) for e in entities}
    flat_ids = [nid for ids in resolved.values() for nid in ids]

    # 3) Rank by embedding similarity
    embeddings = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-30m-english")
    q_vec    = embeddings.embed_query(user_query)
    descs    = [G.nodes[n]["description"] or G.nodes[n]["name"] for n in flat_ids]
    doc_vecs = embeddings.embed_documents(descs)
    scores   = np.dot(doc_vecs, q_vec)
    top_nids = [flat_ids[i] for i in np.argsort(scores)[-8:][::-1]]

    # 4) Build Capella snippet blocks & src_map
    capella_blocks, src_map = [], {}
    for nid in top_nids:
        xml = slice_relevant_xml(nid, G)
        if not xml:
            continue
        sid = f"S{uuid.uuid4().hex[:6]}"
        src_map[sid] = {
            "kind":     "capella",
            "id":       nid,
            "tag_name": G.nodes[nid]["tag"],
            "name":     G.nodes[nid]["name"],
            "snippet":  xml,
        }
        capella_blocks.append(f"[{sid}] ({G.nodes[nid]['tag']}) id={nid}\n```xml\n{xml}\n```")

    # 5) PDF retrieval
    pdf_chunks = vectordb.similarity_search(user_query, k=k_pdf)
    pdf_blocks = []
    for ch in pdf_chunks:
        sid = f"S{uuid.uuid4().hex[:6]}"
        page = ch.metadata.get("page", "?")
        src_map[sid] = {"kind": "pdf", "page": page, "snippet": ch.page_content}
        pdf_blocks.append(f"[{sid}] (page {page})\n{ch.page_content}")

    # 6) Assemble and invoke QA prompt
    CATEGORY_DESC = """
1. simple_fact          : a single factual answer.
2. simple_conditional   : answer depends on an 'if' condition.
3. comparison           : compare / evaluate two items.
4. interpretative       : requires interpretation of intent / rationale.
5. multi_answer         : expects a set/list of items.
6. aggregation          : numeric or textual aggregation.
7. multi_hop            : needs reasoning over ≥2 facts.
8. heavy_post           : answer needs transformation (e.g., unit conversion).
9. erroneous            : user premise wrong; correct it politely.
10. summary             : produce a concise summary.
"""
    safe_schema = qa_parser.get_format_instructions().replace("{", "{{").replace("}", "}}")
    sys_msg = (
        "You are an aerospace-domain assistant. Prefer PDF snippets for facts; "
        "use Capella XML only as supplementary context **and do NOT quote or "
        "leak any XML content in your output**.\n\n"
        "Generate TEN Q-A pairs that follow the JSON schema below. "
        "Every answer must cite at least one [Sxxxxx] token.\n\n"
        "Category definitions:\n" + CATEGORY_DESC +
        "\n\nSchema:\n" + safe_schema
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("human",
         "## Documents\n" + "\n\n".join(pdf_blocks) +
         "\n\n## Capella\n" + "\n\n".join(capella_blocks) +
         "\n\n## Original question\n" + user_query)
    ])
    qa_set: QASet = (qa_prompt | qa_llm | qa_parser).invoke({})

    return qa_set.dict(), src_map

def resolve_all_sources(qa_set: Dict[str, Any], src_map: Dict[str, Any], G: Any) -> Dict[str, List[Dict[str, Any]]]:
    """
    Given qa_set and src_map, return resolved source info per QA category.
    """
    from .resolver import resolve_tag

    resolved = {}
    for cat, qa in qa_set.items():
        tags = extract_tags(qa["answer"])
        resolved[cat] = [resolve_tag(t, src_map, G) for t in tags]
    return resolved
