"""
MBSE Use Case QA Pipeline CLI

This script orchestrates an end-to-end question-answer generation pipeline
specifically tailored for Model-Based Software Engineering (MBSE) use cases.

It performs the following steps:
  1) Change working directory to the script's folder (project root).
  2) Ensure a top-level `results/` directory exists under the project root.
  3) Prepare the output directory for QA results (`--outdir`).
  4) Load environment variables from a `.env` file (if provided).
  5) Read API credentials for LLM services from the environment.
  6) Load a Capella XML model into a NetworkX graph (the MBSE model) and
     optionally export it as JSON.
  7) Load and split a PDF specification into text chunks.
  8) Build a Chroma vector index over those chunks.
  9) Initialize LLMs for entity extraction and QA generation.
 10) Load seed questions defining key MBSE use-case scenarios.
 11) Batch-generate question-answer sets to validate and document the use cases.
 12) Write the QA results to a timestamped JSON file under `--outdir`.

Usage:
    python cli.py --capella <model.capella> \
                  --spec    <requirements.pdf> \
                  --questions <usecase_questions.json> \
                  --outdir  <results_dir> [--dotenv .env] \
                  [--graph_json_out <graph_out>] [--k_pdf <int>]

Required arguments:
  --capella         Path to the Capella XML model file.
  --spec            Path to the PDF requirements/specification.
  --questions       Path to the JSON file with seed use-case questions.
  --outdir          Directory for writing QA results (created if needed).

Optional arguments:
  --dotenv          Path to a `.env` file for environment variables.
  --graph_json_out  Directory or file path for dumping the MBSE graph JSON.
  --k_pdf           Number of PDF chunks to retrieve per query (default: 5).

Environment Variables:
  OPENAI_API_KEY or OPENROUTER_API_KEY     API key for the LLM service.
  OPENAI_API_BASE or OPENROUTER_BASE_URL   Base URL for the LLM API.
"""

import os
import json
import datetime as dt
import argparse
from pathlib import Path
from dotenv import load_dotenv
import networkx as nx

from graph_builder import build_network, save_network_json
from vector_index import load_and_split, build_vector_index
from resolver import build_name_index
from qa_generator import (
    setup_entity_extractor,
    setup_qa_llm,
    generate_qa_set
)

def run(args):
    """
    Execute the MBSE use-case QA pipeline using the provided arguments.
    """
    # 1) Change working directory to project root (where this script lives)
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)

    # 2) Ensure top-level 'results/' under project root exists
    (project_root / "results").mkdir(parents=True, exist_ok=True)

    # 3) Prepare output directory (can override via --outdir)
    outdir_path = Path(args.outdir)
    if not outdir_path.is_absolute():
        outdir_path = project_root / outdir_path
    outdir_path.mkdir(parents=True, exist_ok=True)

    # 4) Load .env if provided
    if args.dotenv:
        load_dotenv(args.dotenv)

    # 5) Read credentials from environment
    api_key  = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENROUTER_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if not api_key or not api_base:
        raise RuntimeError("API key/base not found in environment")

    # 6) Build Capella graph (and optionally save network JSON)
    print("Building Capella network…")
    G = build_network(Path(args.capella))
    print(f"  → {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    if args.graph_json_out:
        graph_out = Path(args.graph_json_out)
        if not graph_out.is_absolute():
            graph_out = project_root / graph_out
        if (graph_out.exists() and graph_out.is_dir()) or str(graph_out).endswith(os.sep):
            graph_out.mkdir(parents=True, exist_ok=True)
            graph_file = graph_out / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_network.json"
        else:
            graph_file = graph_out
            graph_file.parent.mkdir(parents=True, exist_ok=True)
        save_network_json(G, graph_file)
        print(f"  → Network JSON written to {graph_file.resolve()}")

    # 7) Load & split PDF, then build vector index
    print("Loading and splitting PDF…")
    chunks = load_and_split(Path(args.spec))
    print("Building Chroma vector index…")
    vectordb = build_vector_index(chunks, persist_dir=str(outdir_path / "chroma"))

    # 8) Initialize LLMs with env credentials /  TODO: better document te params
    extract_llm, extract_prompt, extract_parser = setup_entity_extractor(
        api_key=api_key, api_base=api_base
    )

    # Agent responsible for generating the question/answer/source truple
    qa_llm, qa_parser = setup_qa_llm(
        api_key=api_key, api_base=api_base
    )

    # 9) Load seed questions JSON
    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # 10) Batch run QA generation
    print("\nStarting batch run…")
    name_idx = build_name_index(G)
    choices  = {nid: data["name"] for nid, data in G.nodes(data=True)}
    records  = []
    for q in questions:
        try:
            qa_set, _ = generate_qa_set(
                q, G, vectordb,
                qa_llm, qa_parser,
                extract_llm, extract_prompt, extract_parser,
                name_idx, choices,
                k_pdf=args.k_pdf
            )
            records.append({"question": q, "qa": qa_set})
            print(f"  ✓ {q}")
        except Exception as e:
            records.append({"question": q, "error": str(e)})
            print(f"  ✗ {q}: {e}")

    # 11) Write QA results JSON
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file  = outdir_path / f"qa_results_{timestamp}.json"
    out_file.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Batch results written to:", out_file.resolve())

def main():
    """
    Parse command-line arguments and invoke the MBSE QA pipeline.
    """
    parser = argparse.ArgumentParser(description="MBSE QA pipeline")
    parser.add_argument("run", nargs="?", help="Run the pipeline")
    parser.add_argument("--capella",        required=True, help=".capella XML file")
    parser.add_argument("--spec",           required=True, help="PDF spec file")
    parser.add_argument("--questions",      required=True, help="Seed questions JSON")
    parser.add_argument(
        "--outdir", "-o",
        default="./results/",
        help="Output directory for QA results (relative to project root if not absolute)"
    )
    parser.add_argument(
        "--dotenv", "-e",
        default="./.env",
        help="Path to .env file (optional)"
    )
    parser.add_argument(
        "--graph_json_out", "-g",
        default="./results/",
        help=(
            "Where to dump graph JSON. "
            "If this names a directory (or ends with '/'), "
            "we'll emit <timestamp>_network.json into it; "
            "otherwise we write exactly to this file path."
        )
    )
    parser.add_argument(
        "--k_pdf", "-k",
        type=int,
        default=5,
        help="Number of PDF chunks to retrieve per query"
    )
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
