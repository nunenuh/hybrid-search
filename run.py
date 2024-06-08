import os
import sys

module_path = os.path.join(os.getcwd(), "src")
if module_path not in sys.path:
    sys.path.append(module_path)

import gradio as gr
import pandas as pd

from hybrid_search.evaluation import (calculate_confusion_metrics,
                                      calculate_evaluation_metrics,
                                      generate_labels)
from hybrid_search.mapper import DictMapper
from hybrid_search.search import HybridSearch

# Add src to sys.path


def build_mapping(file_path="data/finance_template_map.xlsx"):
    sheet_name = "Income Statement"
    dmap = DictMapper(file_path, sheet_name)

    target_column = "C"
    base_source_columns = ["F", "G"]
    test_source_columns = ["H", "I", "J"]

    base_mapping = dmap.create_mapping_dict(base_source_columns, target_column)
    test_mapping = dmap.create_mapping_dict(test_source_columns, target_column)

    return base_mapping, test_mapping


def build_engine(base_mapping):
    sbert_model_name = "uonyeka/bge-base-financial-matryoshka"
    engine = HybridSearch(base_mapping, transformer_model=sbert_model_name)

    return engine


def hybrid_search_and_evaluate(search, top_n, bm25_weight, transformer_weight):
    results = engine.hybrid_search(
        search,
        top_n=int(top_n),
        bm25_weight=bm25_weight,
        transformer_weight=transformer_weight,
    )
    return pd.DataFrame(results)


def generate_evaluation():
    true_labels, predicted_labels = generate_labels(test_mapping, engine.hybrid_search)
    precision, recall, f1, accuracy = calculate_evaluation_metrics(
        true_labels, predicted_labels
    )
    tp, fp, fn, tn, _ = calculate_confusion_metrics(true_labels, predicted_labels)

    evaluation_df = pd.DataFrame(
        {
            "Metric": [
                "True Positive",
                "False Positive",
                "False Negative",
                "True Negative",
                "Precision",
                "Recall",
                "F1 Score",
                "Accuracy",
            ],
            "Value": [
                tp.sum(),
                fp.sum(),
                fn.sum(),
                tn.sum(),
                precision,
                recall,
                f1,
                accuracy,
            ],
        }
    )

    return evaluation_df


def initialize(file, progress=gr.Progress(track_tqdm=True)):
    progress(0, "Initializing...")
    try:
        file_path = file.name
        global engine, test_mapping, base_mapping
        progress(0.25, "Initialization mapping from file...")
        base_mapping, test_mapping = build_mapping(file_path)
        progress(0.75, "Initialization search engine...")
        engine = build_engine(base_mapping)
        progress(1, "Initialization complete")
        return (
            gr.update(
                value="File uploaded and engine initialized successfully.", visible=True
            ),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    except Exception as e:
        progress(0.0, "Initialization failed")
        return (
            gr.update(value=f"Initialization failed: {str(e)}", visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        )


def main():
    with gr.Blocks() as iface:
        with gr.Column() as col:
            gr.Markdown(
                "<h1 style='text-align: center;'>Hybrid Search and Evaluation</h1>"
            )
            gr.Markdown(
                "<p style='text-align: center;'>Upload a file to initialize the search engine, then enter a query to perform hybrid search and see the top results.</p>"
            )

            with gr.Column(visible=True) as upload_row:
                upload = gr.File(
                    label="Upload Excel File", file_types=[".xlsx"], scale=1
                )
                init_status = gr.Textbox(
                    label="Initialization Status",
                    interactive=False,
                    visible=True,
                    scale=1,
                )
                upload_button = gr.Button("Upload and Initialize", scale=1)
                gr.Markdown("")

            with gr.Column(visible=False) as search_col:
                search_input = gr.Textbox(
                    label="Search",
                    placeholder="Enter your query here",
                    show_label=False,
                )
                submit_button = gr.Button("Submit")

                with gr.Row():
                    top_n = gr.Number(value=5, label="Top N", precision=0)
                    bm25_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="BM25 Weight",
                    )
                    transformer_weight = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Transformer Weight",
                    )

                result_output = gr.Dataframe(
                    headers=["similar_name", "account_name", "scores"],
                    datatype=["str", "str", "number"],
                )
                evaluation_button = gr.Button("Generate Evaluation")
                evaluation_output = gr.Dataframe(
                    headers=["Metric", "Value"],
                    datatype=["str", "number"],
                    label="Evaluation Results",
                )

                submit_button.click(
                    hybrid_search_and_evaluate,
                    inputs=[search_input, top_n, bm25_weight, transformer_weight],
                    outputs=result_output,
                )
                evaluation_button.click(generate_evaluation, outputs=evaluation_output)

            upload_button.click(
                initialize, inputs=upload, outputs=[init_status, upload_row, search_col]
            )

    iface.launch()


if __name__ == "__main__":
    main()
