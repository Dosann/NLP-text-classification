import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


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