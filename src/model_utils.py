import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from eli5.lime import TextExplainer


def evaluate(y_true, y_pred, print_cm=True, labels=None):
    """
    Evaluate the results of a model using actual and predicted values.
    
    Arguments:
    y_true : actual values
    y_pred : predicted values
    print_cm : whether to print confusion matrix
    
    Returns:
    tuple : tuple of f1 and accracy scores
    """
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cmd = ConfusionMatrixDisplay(cm, display_labels=labels)

    if print_cm:
        fig, ax = plt.subplots(figsize=(16,16))
        plt.title('Confusion matrix\n', fontsize=20)
        cmd.plot(ax=ax, xticks_rotation='vertical')
        
    return (f1_score(y_true, y_pred, average='macro'),
            accuracy_score(y_true, y_pred))


def find_doc_on_theme(theme, target_names, target_labels, text_data):
    """
    Find text document in all documents on a specific theme.
    
    Arguments:
    theme : theme of the text document
    target_names : list of all themes of the documents
    target_labels : labels of themes
    text_data: text data
    
    Returns:
    doc : first occurence of text on theme in text data list
    """
    
    theme_index = list.index(target_names, theme)
    theme_doc_index = np.where(target_labels == theme_index)[0][0]
    doc = text_data[theme_doc_index]
    return doc


def show_text_explanation(doc, predict_proba, target_names):
    """
    Explain and show predictions.
    
    Arguments:
    doc : text to explain
    predict_proba : black-box probabilistic classification function
    target_names : list of all themes of the documents
    
    Returns:
        show positive and negative words contribution
    """
    
    te = TextExplainer(random_state=42)
    te.fit(doc, predict_proba)
    print(f'\nMetrics:{te.metrics_}\n')
    return te.show_prediction(target_names=target_names)