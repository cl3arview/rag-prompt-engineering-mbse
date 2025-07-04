# MBSE Use-Case QA Pipeline

This repository provides an end-to-end CLI for automatically generating structured question–answer sets to validate and document Model-Based Software Engineering (MBSE) use cases. It integrates Capella model parsing, PDF specification indexing, and large-language-model–driven QA generation.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Prerequisites](#prerequisites)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Repository Structure](#repository-structure)  
7. [Module Descriptions](#module-descriptions)  
8. [Output](#output)  
9. [Environment Variables](#environment-variables)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## Overview

The pipeline orchestrates the following high-level steps:
1. **Graph Construction**  
   Parse a Capella `.capella` XML model into a NetworkX directed graph of containment relationships.

2. **Specification Indexing**  
   Load a PDF requirements document, split it into overlapping text chunks, and build a Chroma vector index for semantic retrieval.

3. **Seed Question Loading**  
   Read a JSON file of user-defined “seed” questions that describe key MBSE use-case scenarios.

4. **QA Generation**  
   For each seed question:
   - Extract Capella entities via an entity-extraction LLM.
   - Resolve entities to graph nodes (exact and fuzzy matching).
   - Retrieve top relevant PDF chunks and Capella snippets.
   - Generate ten categorized Q-A pairs (simple fact, conditional, comparison, etc.) via a second LLM.
   - Enforce citation of PDF/Capella snippets by [Sxxxxx] tokens.

5. **Results Export**  
   Write a timestamped JSON file of questions and QA sets under the specified output directory.

---

## Features

- **Automated entity extraction** from Capella model with a Pydantic-backed LLM parser.
- **Semantic PDF retrieval** using HuggingFace embeddings and Chroma.
- **Structured multi-category QA** generation following a strict schema.
- **JSON-based outputs** with embedded snippet citations for traceability.
- **Configurable** via command-line arguments and `.env` for API credentials.

---

## Prerequisites

- Python 3.9 or newer  
- Access to an LLM provider (OpenAI or OpenRouter compatible)  
- Capella XML model file (`*.capella`)  
- PDF specification document (`*.pdf`)  
- Seed questions JSON (`usecase_questions.json`)  

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/cl3arview/rag-prompt-engineering-mbse
   cd rag-prompt-engineering-mbse
   ```

2. **Create the Conda environment from the YAML file**  
   ```bash
   conda env create -n mbseqa -f requirements.yaml
   ```

3. **Activate the environment**  
   ```bash
   conda activate mbseqa
   ```

---

## Usage

```bash
python cli.py   --capella     path/to/model.capella   --spec        path/to/requirements.pdf   --questions   path/to/usecase_questions.json   --outdir      results/   [--dotenv     .env]   [--graph_json_out  results/graph/]   [--k_pdf      5]
```

- `--capella`  Path to the Capella XML model.  
- `--spec`   Path to the PDF requirements/specification.  
- `--questions` Path to the JSON file containing seed use-case questions.  
- `--outdir`  Directory to write QA results (created if missing).  
- `--dotenv`  Optional path to a `.env` file for API credentials.  
- `--graph_json_out`  
  - If a directory: dumps `<timestamp>_network.json` inside it.  
  - If a file: writes directly to that path.  
- `--k_pdf`   Number of PDF chunks to retrieve per query (default: 5).  

---

## Repository Structure

```
.
├── __init__.py
├── cli.py
├── graph_builder.py
├── vector_index.py
├── qa_generator.py
├── resolver.py
├── results/                # Default output directory
├── usecase_questions.json  # Example seed questions
├── requirements.yaml
└── README.md
```

---

## Module Descriptions

### `cli.py`  
Orchestrates the full pipeline: directory setup, environment loading, graph build, vector index creation, LLM initialization, batch QA generation, and result writing.

### `graph_builder.py`  
Parses Capella XML via `lxml.iterparse`, builds a `networkx.DiGraph` of containment edges, and optionally serializes to node-link JSON.

### `vector_index.py`  
Loads PDF pages via `PyPDFium2Loader`, splits text into overlapping chunks, and builds/persists a Chroma index using HuggingFace embeddings.

### `qa_generator.py`  
- **Entity extraction**: prompts a ChatOpenAI model to output a JSON list of Capella element names.  
- **QA generation**: for each question, retrieves PDF & XML contexts, then prompts a second LLM to produce a ten-category `QASet` with snippet citations.

### `resolver.py`  
- Fuzzy/exact name matching to map entity strings to graph node IDs.  
- Utilities to slice and minify XML snippets for prompt inclusion.  
- Tag‐extraction for `[Sxxxxx]` citation tokens in generated answers.

---

## Output

All results are written as a UTF-8, indented JSON array in:

```
<outdir>/qa_results_<YYYYMMDD_HHMMSS>.json
```

Each record contains:
- `question`: the original seed question.
- `qa`: a `QASet` object with ten QA items (keys: `simple_fact`, `comparison`, …, `summary`).
- Or, on error, an `error` field with the exception message.

If `--graph_json_out` is set, a snapshot of the Capella graph is also saved for inspection.

---

## Environment Variables

Set one of the following pairs in your shell or via a `.env` file:

```dotenv
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1

# — or —

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://your.openrouter.endpoint
```

