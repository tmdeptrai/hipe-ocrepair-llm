import difflib
import re
from collections import Counter

def get_char_similarity(s1, s2):
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def is_garbage(text):
    if not text: return False
    # Ratio of non-alphanumeric characters
    noise_chars = re.findall(r'[^a-zA-Z0-9\s]', text)
    return len(noise_chars) / len(text) > 0.5

def classify_ocr_error(tag, ocr_chunk, gt_chunk):
    ocr_len = len(ocr_chunk)
    gt_len = len(gt_chunk)
    sim = get_char_similarity(ocr_chunk, gt_chunk)

    # 1. Hallucination
    # If the OCR is producing something completely different from the GT
    if tag in ['replace', 'insert']:
        # Only classify as hallucination if it's more than just a single noisy char
        is_garbage_noise = is_garbage(ocr_chunk) and ocr_len > 1
        is_expansion = gt_len > 0 and ocr_len > gt_len * 2 and ocr_len > 5
        is_total_mismatch = sim < 0.2 and ocr_len > 3

        if is_garbage_noise or is_expansion or is_total_mismatch:
            return "hallucination"

    # 2. Structural Checks (Spaces)
    if ' ' in ocr_chunk and ' ' not in gt_chunk:
        return "split"
    if ' ' in gt_chunk and ' ' not in ocr_chunk:
        return "merge"

    # 3. Pure Deletions/Insertions
    if tag == 'delete':
        return "deletion"
    if tag == 'insert':
        return "insertion"

    # 4. Replacements (Substitution vs Merge vs Deletion)
    if tag == 'replace':
        if ocr_len == 0: return "deletion"
        if gt_len == 0: return "insertion"
        
        # If lengths differ and they are similar, it's a character merge
        if ocr_len < gt_len and sim > 0.4:
            return "merge"
        return "substitution"

    return f"misc_{tag}"

def analyze_document_errors(ocr_text, gt_text):
    """
    Aligns doc and returns a counter of error types.
    """
    matcher = difflib.SequenceMatcher(None, ocr_text, gt_text)
    error_counts = Counter()
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        
        ocr_chunk = ocr_text[i1:i2]
        gt_chunk = gt_text[j1:j2]
        
        error_type = classify_ocr_error(tag, ocr_chunk, gt_chunk)
        error_counts[error_type] += 1
        
    return dict(error_counts)

if __name__ == "__main__":
    # Test cases based on the new scheme
    test_cases = [
        ("Condition", "Condit1on", "substitution"),
        ("Training", "Traning", "merge"),
        ("dent", "den", "deletion"), 
        ("Total", "Tot ' al", "split"),
        ("Museum", "Mu seum", "split"),
        ("clear", "dear", "merge"),
        ("word1 word2", "word1word2", "merge"),
        ("", "||_~||", "hallucination"), # pure noise
        ("a", "hallucinated long string", "hallucination"), # expansion
        ("simple", "totallydifferent", "hallucination"), # low similarity
        ("", "!", "insertion"), # short noise insertion
        ("!", "", "deletion"), # short noise deletion
    ]

    print(f"{'GT':<15} | {'OCR':<25} | {'Expected':<15} | {'Result'}")
    print("-" * 80)

    all_passed = True
    for gt, ocr, expected in test_cases:
        # Determine tag for classification
        if gt and ocr: tag = 'replace'
        elif ocr: tag = 'insert'
        else: tag = 'delete'

        result = classify_ocr_error(tag, ocr, gt)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        if result != expected: all_passed = False

        print(f"{gt:<15} | {ocr:<25} | {expected:<15} | {status}")

    if not all_passed:
        exit(1)
    else:
        print("\nAll tests passed!")
