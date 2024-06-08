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


# Function for Gradio interface
def hybrid_search_and_evaluate(search):
    results = engine.hybrid_search(search, top_n=5)
    df_results = pd.DataFrame(results)
    return df_results


# Main function to parse arguments and run appropriate interface
def main():
    parser = argparse.ArgumentParser(description="Hybrid Search and Evaluation")
    parser.add_argument(
        "--cli", type=str, help="Run in CLI mode with the provided search query"
    )
    args = parser.parse_args()

    if args.cli:
        run_cli(args.cli)
    else:
        # Create Gradio interface
        iface = gr.Interface(
            fn=hybrid_search_and_evaluate,
            inputs="text",
            outputs="dataframe",
            title="Hybrid Search and Evaluation",
            description="Enter a query to perform hybrid search and see the top 5 results.",
        )
        iface.launch()


if __name__ == "__main__":
    main()
