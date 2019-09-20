import numpy as np
from typing import Dict, List


class EvaluationResults:
    """
    Aggregates and stores evaluation results
      for average precision metric.

    Metrics stored:
     - AP for each class
     - Recalls for each class
     - Interpolated precisions for each class
     - Number of TP for each class
     - Number of FP for each class
     - Mean average precision (AP averaged across all classes)
    """

    mAP:                            float
    AP_per_class:                   Dict[str, float]
    recalls_per_class:              Dict[str, List[float]]
    interp_precisions_per_class:    Dict[str, List[float]]
    TP_per_class:                   Dict[str, int]
    FP_per_class:                   Dict[str, int]

    def __init__(
        self,
        *,
        AP_per_class:                Dict[str, float],
        recalls_per_class:           Dict[str, List[float]],
        interp_precisions_per_class: Dict[str, List[float]],
        TP_per_class:                Dict[str, int],
        FP_per_class:                Dict[str, int]
    ):
        self.mAP = np.sum(list(AP_per_class.values())) / len(AP_per_class)

        self.AP_per_class = dict(AP_per_class)
        self.recalls_per_class = dict(recalls_per_class)
        self.interp_precisions_per_class = dict(interp_precisions_per_class)
        self.TP_per_class = TP_per_class
        self.FP_per_class = FP_per_class
