import os
import sys

module_path = os.path.join(os.getcwd(), "src")
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse

# Initialize the HybridSearch engine (ensure this is properly initialized)
# engine = HybridSearch(base_mapping, transformer_model="uonyeka/bge-base-financial-matryoshka")
import gradio as gr
import pandas as pd

from hybrid_search.mapper import DictMapper
from hybrid_search.search import HybridSearch

from hybrid_search.search import HybridSearch, FAISSManager
from hybrid_search.evaluation import (
    calculate_evaluation_metrics, 
    calculate_confusion_metrics,
    generate_labels, 
    evaluate_search_accuracy
)


def build_engine(file_path=None):
    if file_path is None:
        file_path = "data/finance_template_map.xlsx"
    sheet_name = "Income Statement"
    dmap = DictMapper(file_path, sheet_name)

    target_column = "C"
    base_source_columns = ["F", "G"]
    test_source_columns = ["H", "I", "J"]

    base_mapping = dmap.create_mapping_dict(base_source_columns, target_column)
    test_mapping = dmap.create_mapping_dict(test_source_columns, target_column)

    sbert_model_name = "uonyeka/bge-base-financial-matryoshka"
    engine = HybridSearch(base_mapping, transformer_model=sbert_model_name)
    
    return engine, base_mapping, test_mapping

engine, base_mapping, test_mapping = build_engine()

# Function for Gradio interface
def hybrid_search_and_evaluate(search):
    results = engine.hybrid_search(search, top_n=5)
    df_results = pd.DataFrame(results)
    return df_results

def generate_evaluation():
    true_labels, predicted_labels = generate_labels(
        test_mapping, 
        engine.hybrid_search
    )
    precision, recall, f1, accuracy = calculate_evaluation_metrics(
        true_labels, 
        predicted_labels
    )

    tp, fp, fn, tn, labels = calculate_confusion_metrics(
        true_labels, 
        predicted_labels
    )
    
    text_output = (
        "Confusion Matrix Value:\n"
        f"{'-'*25}\n"
        f"True Positive   : {tp.sum()}\n"
        f"False Positive  : {fp.sum()}\n"
        f"False Negative  : {fn.sum()}\n"
        f"True Negative   : {tn.sum()}\n"
        "\n"
        "Evaluation Metrics:\n"
        f"{'-'*20}\n"
        f"{'Precision':<10} {precision:.4f}\n"
        f"{'Recall':<10} {recall:.4f}\n"
        f"{'F1 Score':<10} {f1:.4f}\n"
        f"{'Accuracy':<10} {accuracy:.4f}"
    )
    return text_output

# Main function to parse arguments and run appropriate interface
def main():
 # Create Gradio interface with centered layout
    with gr.Blocks() as iface:
        with gr.Column():
            gr.Markdown("<h1 style='text-align: center;'>Hybrid Search and Evaluation</h1>")
            gr.Markdown("<p style='text-align: center;'>Enter a query to perform hybrid search and see the top 5 results.</p>")
            search_input = gr.Textbox(label="Search", placeholder="Enter your query here", show_label=False)
            submit_button = gr.Button("Submit")
            result_output = gr.Dataframe(headers=["similar_name", "account_name", "scores"], datatype=["str", "str", "number"])
            evaluation_button = gr.Button("Generate Evaluation")
            evaluation_output = gr.Textbox(label="Evaluation Results", placeholder="Evaluation metrics will be displayed here.", show_label=False)

        submit_button.click(hybrid_search_and_evaluate, inputs=search_input, outputs=result_output)
        evaluation_button.click(generate_evaluation, outputs=evaluation_output)

    iface.launch()

if __name__ == "__main__":
    main()