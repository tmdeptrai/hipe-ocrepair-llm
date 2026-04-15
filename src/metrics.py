import numpy as np
from jiwer import cer, wer
from typing import List, Dict, Union, Tuple

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculates Character Error Rate (CER)."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    return cer(reference, hypothesis)

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculates Word Error Rate (WER)."""
    if not reference:
        return 1.0 if hypothesis else 0.0
    return wer(reference, hypothesis)

def get_stats(data: List[float]) -> Dict[str, float]:
    """Computes basic statistics for a list of error rates."""
    if not data:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "count": len(data)
    }

def cohens_d(old_errors: List[float], new_errors: List[float]) -> float:
    """
    Computes Cohen's d effect size between two sets of error rates.
    d = (mean1 - mean2) / pooled_std
    """
    n1, n2 = len(old_errors), len(new_errors)
    if n1 == 0 or n2 == 0:
        return 0.0
    
    mu1, mu2 = np.mean(old_errors), np.mean(new_errors)
    var1, var2 = np.var(old_errors, ddof=1), np.var(new_errors, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
        
    return (mu1 - mu2) / pooled_std


def relative_error_reduction(old_val: float, new_val: float) -> float:
    """Calculates the percentage reduction in error (e.g., CER/WER reduction)."""
    if old_val == 0:
        return 0.0
    return (old_val - new_val) / old_val

def get_comparative_report(
    references: List[str], 
    ocr_hyps: List[str], 
    model_hyps: List[str]
) -> Dict[str, Union[Dict, float]]:
    """
    Generates a full comparative report between Original OCR and Model Corrected text.
    """
    ocr_cer_list = [calculate_cer(r, h) for r, h in zip(references, ocr_hyps)]
    model_cer_list = [calculate_cer(r, h) for r, h in zip(references, model_hyps)]
    
    ocr_stats = get_stats(ocr_cer_list)
    model_stats = get_stats(model_cer_list)
    
    return {
        "ocr_stats": ocr_stats,
        "model_stats": model_stats,
        "relative_cer_reduction": relative_error_reduction(ocr_stats['mean'], model_stats['mean']),
        "cohens_d": cohens_d(ocr_cer_list, model_cer_list),
    }

if __name__ == "__main__":
    # Small test case
    gts = ["This is a test.", "Another clean sentence."]
    ocrs = ["Thas is a tast.", "Anothr clen sentnce."]
    models = ["That is a test.", "Another clean sentence."]
    
    report = get_comparative_report(gts, ocrs, models)
    print("Comparative Report (Test):")
    for k, v in report.items():
        print(f"{k}: {v}")
