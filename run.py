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
# Function for Gradio interface
def hybrid_search_and_evaluate(search, top_n, bm25_weight, transformer_weight):
    results = engine.hybrid_search(search, top_n=int(top_n), bm25_weight=bm25_weight, transformer_weight=transformer_weight)
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
    
    evaluation_df = pd.DataFrame({
        "Metric": ["True Positive", "False Positive", "False Negative", "True Negative", "Precision", "Recall", "F1 Score", "Accuracy"],
        "Value": [tp.sum(), fp.sum(), fn.sum(), tn.sum(), precision, recall, f1, accuracy]
    })
    
    return evaluation_df

# Main function to parse arguments and run appropriate interface
def main():
    with gr.Blocks() as iface:
        with gr.Column():
            gr.Markdown("<h1 style='text-align: center;'>Hybrid Search and Evaluation</h1>")
            gr.Markdown("<p style='text-align: center;'>Enter a query to perform hybrid search and see the top 5 results.</p>")
            search_input = gr.Textbox(label="Search", placeholder="Enter your query here", show_label=False)
            submit_button = gr.Button("Submit")
            
            with gr.Row():
                top_n = gr.Number(value=5, label="Top N")
                bm25_weight = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, step=0.1, label="BM25 Weight")
                transformer_weight = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, label="Transformer Weight")

            result_output = gr.Dataframe(headers=["similar_name", "account_name", "scores"], datatype=["str", "str", "number"])
            evaluation_button = gr.Button("Generate Evaluation")
            evaluation_output = gr.Dataframe(headers=["Metric", "Value"], datatype=["str", "number"], label="Evaluation Results")

            submit_button.click(hybrid_search_and_evaluate, inputs=[search_input, top_n, bm25_weight, transformer_weight], outputs=result_output)
            evaluation_button.click(generate_evaluation, outputs=evaluation_output)

    iface.launch()

if __name__ == "__main__":
    main()