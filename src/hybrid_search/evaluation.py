import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


def generate_labels(test_mapping_dict, search_engine, search_method):
    true_labels = []
    predicted_labels = []
    for key, true_value in test_mapping_dict.items():
        results = search_method(query=key, top_n=1)
        predicted_value = results[0][2] if results else "Unmapped"
        true_labels.append(true_value)
        predicted_labels.append(predicted_value)
    return true_labels, predicted_labels


def calculate_confusion_metrics(true_labels, predicted_labels):
    labels = list(set(true_labels + predicted_labels))
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    
    true_positive = np.diag(cm)
    false_positive = cm.sum(axis=0) - true_positive
    false_negative = cm.sum(axis=1) - true_positive
    true_negative = cm.sum() - (false_positive + false_negative + true_positive)
    
    return true_positive, false_positive, false_negative, true_negative, labels


def calculate_evaluation_metrics(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    accuracy = accuracy_score(true_labels, predicted_labels)
    return precision, recall, f1, accuracy


def print_evaluation_metrics(true_labels, predicted_labels):
    precision, recall, f1, accuracy = calculate_evaluation_metrics(
        true_labels,
        predicted_labels
    )

    tp, fp, fn, tn, labels = calculate_confusion_metrics(
        true_labels,
        predicted_labels
    )

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
