import numpy as np
from jiwer import cer, wer
from typing import List, Dict, Union

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
    Generates a full comparative report between Original OCR and Model Corrected text,
    tracking both Character Error Rate (CER) and Word Error Rate (WER).
    """
    # Calculate CER
    ocr_cer_list = [calculate_cer(r, h) for r, h in zip(references, ocr_hyps)]
    model_cer_list = [calculate_cer(r, h) for r, h in zip(references, model_hyps)]
    
    # Calculate WER
    ocr_wer_list = [calculate_wer(r, h) for r, h in zip(references, ocr_hyps)]
    model_wer_list = [calculate_wer(r, h) for r, h in zip(references, model_hyps)]
    
    ocr_cer_stats = get_stats(ocr_cer_list)
    model_cer_stats = get_stats(model_cer_list)
    
    ocr_wer_stats = get_stats(ocr_wer_list)
    model_wer_stats = get_stats(model_wer_list)
    
    return {
        "ocr_cer_stats": ocr_cer_stats,
        "model_cer_stats": model_cer_stats,
        "relative_cer_reduction": relative_error_reduction(ocr_cer_stats['mean'], model_cer_stats['mean']),
        "ocr_wer_stats": ocr_wer_stats,
        "model_wer_stats": model_wer_stats,
        "relative_wer_reduction": relative_error_reduction(ocr_wer_stats['mean'], model_wer_stats['mean']),
    }

if __name__ == "__main__":
    # Small test case
    gts = ["This is a test.", "Another clean sentence."]
    ocrs = ["Thas is a tast.", "Anothr clen sentnce."]
    models = ["That is a test.", "Another clean sentence."]
    
    report = get_comparative_report(gts, ocrs, models)
    print("Comparative Report (Test):")
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"{k}: Mean={v['mean']:.4f}, Std={v['std']:.4f}")
        else:
            print(f"{k}: {v*100:.2f}%")