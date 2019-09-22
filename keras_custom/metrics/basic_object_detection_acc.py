from utils.compute_basic_accuracy import compute_basic_precision

def basic_accuracy(y_true, y_pred):
    return compute_basic_precision(y_pred, y_true)