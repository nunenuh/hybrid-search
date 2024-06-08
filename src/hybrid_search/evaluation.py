import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

def generate_labels(test_mapping_dict, search_method):
    """
    Generates true and predicted labels using the provided search method.

    Args:
        test_mapping_dict (dict): Dictionary containing test data with keys as queries and values as true labels.
        search_method (function): Search method to generate predicted labels.

    Returns:
        tuple: Lists of true labels and predicted labels.
    """
    true_labels = []
    predicted_labels = []
    for key, true_value in test_mapping_dict.items():
        results = search_method(query=key, top_n=1)
        predicted_value = results[0]['account_name'] if results else "Unmapped"
        true_labels.append(true_value)
        predicted_labels.append(predicted_value)
    return true_labels, predicted_labels

def evaluate_search_accuracy(test_mapping_dict, search_engine, search_method):
    """
    Evaluates search accuracy and returns detailed results as a DataFrame.

    Args:
        test_mapping_dict (dict): Dictionary containing test data.
        search_engine (object): Search engine instance.
        search_method (function): Search method to use.

    Returns:
        tuple: DataFrame with detailed results and accuracy percentage.
    """
    results = []
    correct = 0
    for key, true_value in test_mapping_dict.items():
        search_results = search_method(key, top_n=1)
        predicted_value = search_results[0]['account_name'] if search_results else "Unmapped"
        is_correct = predicted_value == true_value
        if is_correct:
            correct += 1
        results.append({
            "Key": key,
            "Predicted": predicted_value,
            "Ground Truth": true_value,
            "Correct": is_correct,
            "Score": f"{search_results[0]['scores']:.4f}" if search_results else "N/A",
        })

    accuracy = correct / len(test_mapping_dict) * 100
    return pd.DataFrame(results), accuracy

def calculate_confusion_metrics(true_labels, predicted_labels):
    """
    Calculates confusion metrics.

    Args:
        true_labels (list): List of true labels.
        predicted_labels (list): List of predicted labels.

    Returns:
        tuple: Confusion matrix components and labels.
    """
    labels = list(set(true_labels + predicted_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)

    true_positive = np.diag(cm)
    false_positive = cm.sum(axis=0) - true_positive
    false_negative = cm.sum(axis=1) - true_positive
    true_negative = cm.sum() - (false_positive + false_negative + true_positive)

    return true_positive, false_positive, false_negative, true_negative, labels

def calculate_evaluation_metrics(true_labels, predicted_labels):
    """
    Calculates evaluation metrics.

    Args:
        true_labels (list): List of true labels.
        predicted_labels (list): List of predicted labels.

    Returns:
        tuple: Precision, recall, F1 score, and accuracy.
    """
    precision = precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return precision, recall, f1, accuracy

def print_evaluation_metrics(true_labels, predicted_labels):
    """
    Prints evaluation metrics.

    Args:
        true_labels (list): List of true labels.
        predicted_labels (list): List of predicted labels.
    """
    precision, recall, f1, accuracy = calculate_evaluation_metrics(true_labels, predicted_labels)

    tp, fp, fn, tn, labels = calculate_confusion_metrics(true_labels, predicted_labels)

    print(f"Confusion Matrix Value:")
    print(f"{'-'*25}")
    print(f"True Positive   : {tp.sum()}")
    print(f"False Positive  : {fp.sum()}")
    print(f"False Negative  : {fn.sum()}")
    print(f"True Negative   : {tn.sum()}")
    print("\n")
    print(f"Evaluation Metrics:")
    print(f"{'-'*20}")
    print(f"{'Precision':<10} {precision:.4f}")
    print(f"{'Recall':<10} {recall:.4f}")
    print(f"{'F1 Score':<10} {f1:.4f}")
    print(f"{'Accuracy':<10} {accuracy:.4f}")

# # Example usage
# true_labels, predicted_labels = generate_labels(mapping_test, engine.hybrid_search)
# print_evaluation_metrics(true_labels, predicted_labels)
