import os
import io
import sys
import json
import shutil
import logging
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import gradio as gr

from cli import run as run_pipeline



def _ensure_env():
    """
    Verify that at least one API key/base is present.
    """
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENROUTER_BASE_URL")
    if not key:
        raise RuntimeError("Missing OPENAI_API_KEY or OPENROUTER_API_KEY in environment")
    return key, base


def _save_upload(uploaded: str | Path | io.IOBase, dest: Path) -> Path:
    """
    Copy an uploaded file (either a path or a file-like object) into `dest`,
    preserving the original filename.
    """
    if isinstance(uploaded, (str, Path)):
        src = Path(uploaded)
        dst = dest / src.name
        shutil.copy(src, dst)
        return dst

    # file-like object
    filename = Path(uploaded.name).name
    dst = dest / filename
    with open(dst, "wb") as f:
        f.write(uploaded.read())
    return dst


def _run_pipeline(
    capella_file,
    spec_file,
    questions_file,
    k_pdf: int,
    graph_json_out: Optional[str],
) -> tuple[pd.DataFrame, str, str]:
    """
    Invoke the MBSE QA CLI pipeline, capture logs, and return:
      1) a DataFrame of results,
      2) the raw pipeline stdout/stderr,
      3) the path to the JSON output.
    """
    _ensure_env()

    # 1) Prepare a temp working dir
    workdir = Path("gradio_tmp")
    workdir.mkdir(exist_ok=True)

    # 2) Copy uploads into workdir
    capella_path   = _save_upload(capella_file, workdir)
    spec_path      = _save_upload(spec_file, workdir)
    questions_path = _save_upload(questions_file, workdir)

    # 3) Build CLI args namespace
    args = argparse.Namespace(
        capella=str(capella_path),
        spec=str(spec_path),
        questions=str(questions_path),
        outdir=str(workdir / "results"),
        dotenv=None,
        graph_json_out=(graph_json_out or None),
        k_pdf=k_pdf
    )

    # 4) Capture stdout+stderr
    buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buf, buf
    try:
        run_pipeline(args)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    log_text = buf.getvalue()

    # 5) Find latest JSON
    results_dir = Path(args.outdir)
    json_files  = sorted(results_dir.glob("qa_results_*.json"))
    if not json_files:
        raise RuntimeError("No output JSON found in results dir")
    latest_json = json_files[-1]
    records     = json.loads(latest_json.read_text(encoding="utf-8"))

    # 6) Build DataFrame
    rows = []
    for rec in records:
        rows.append({
            "Question": rec.get("question",""),
            "Error":    rec.get("error",""),
            "Q-A Set":  json.dumps(rec.get("qa",{}), ensure_ascii=False, indent=2)
        })
    df = pd.DataFrame(rows)
    return df, log_text, str(latest_json)



with gr.Blocks(title="MBSE Use Case QA Pipeline") as demo:
    gr.Markdown("## MBSE Use-Case Q&A Generator")

    with gr.Row():
        capella_in = gr.File(
            label="Upload Capella .capella XML",
            file_types=[".capella", ".xml"],
            type="filepath"
        )
        spec_in = gr.File(
            label="Upload Requirements PDF",
            file_types=[".pdf"],
            type="filepath"
        )
        questions_in = gr.File(
            label="Upload Seed Questions JSON",
            file_types=[".json"],
            type="filepath"
        )

    with gr.Accordion("Advanced Settings", open=False):
        k_pdf     = gr.Slider(1, 20, value=5, step=1, label="PDF Chunks per Query (k_pdf)")
        graph_out = gr.Textbox(placeholder="Optional path to dump graph JSON", label="Graph JSON Output")

    run_btn = gr.Button("Run QA Pipeline", variant="primary")
    status  = gr.Textbox(value="", interactive=False, label="Status")

    with gr.Tabs():
        with gr.TabItem("Results Table"):
            result_df     = gr.Dataframe(headers=["Question","Error","Q-A Set"], interactive=False)
            download_json = gr.File(label="Download Full JSON")
        with gr.TabItem("Logs"):
            log_output = gr.Textbox(lines=20, interactive=False, label="Pipeline Logs")

    def _on_run(capella, spec, questions, k, gout):
        # Run the pipeline
        df, logs, out_json = _run_pipeline(capella, spec, questions, k, gout)
        # Return all four outputs in the same order as `outputs=[...]`
        return "Done!", df, logs, out_json

    run_btn.click(
        fn=_on_run,
        inputs=[capella_in, spec_in, questions_in, k_pdf, graph_out],
        outputs=[status, result_df, log_output, download_json]
    )

if __name__ == "__main__":
    demo.queue().launch()
