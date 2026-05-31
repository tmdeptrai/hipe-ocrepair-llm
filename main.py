import difflib
def extract_aligned_chunks(ocr_text, gt_text, target_words=300, window_size=100):
    """
    Ensure the splits happen at exact matching anchor points, prioritizing 
    sentence boundaries (punctuation) within a flexible window.
    """
    ocr_words = str(ocr_text).split()
    gt_words = str(gt_text).split()
    
    if len(ocr_words) < target_words and len(gt_words) < target_words:
        return [{"ocr_text": str(ocr_text), "ground_truth": str(gt_text)}]
        
    matcher = difflib.SequenceMatcher(None, ocr_words, gt_words)
    matching_blocks = matcher.get_matching_blocks()
    
    anchors = []
    for block in matching_blocks:
        ocr_start, gt_start, match_length = block
        if match_length == 0: continue
        for offset in range(match_length + 1):
            anchors.append((ocr_start + offset, gt_start + offset))
            
    anchors = sorted(list(set(anchors)))
    
    chunks = []
    last_ocr_idx = 0
    last_gt_idx = 0
    
    i = 0
    while i < len(anchors):
        ocr_idx, gt_idx = anchors[i]
        current_chunk_size = ocr_idx - last_ocr_idx
        
        if current_chunk_size >= target_words:
            best_anchor_idx = i
            for j in range(i, len(anchors)):
                l_ocr, l_gt = anchors[j]
                if (l_ocr - last_ocr_idx) > (target_words + window_size):
                    break
                if l_ocr > 0:
                    prev_word = ocr_words[l_ocr - 1]
                    if prev_word.endswith(('.', '!', '?', ';', ':')):
                        best_anchor_idx = j
                        break
            
            s_ocr, s_gt = anchors[best_anchor_idx]
            chunk_ocr = " ".join(ocr_words[last_ocr_idx:s_ocr])
            chunk_gt = " ".join(gt_words[last_gt_idx:s_gt])
            
            if chunk_ocr.strip() or chunk_gt.strip():
                chunks.append({"ocr_text": chunk_ocr, "ground_truth": chunk_gt})
                
            last_ocr_idx, last_gt_idx = s_ocr, s_gt
            while i < len(anchors) and anchors[i][0] < last_ocr_idx:
                i += 1
            continue
            
        i += 1
            
    final_ocr = " ".join(ocr_words[last_ocr_idx:])
    final_gt = " ".join(gt_words[last_gt_idx:])
    if final_ocr.strip() or final_gt.strip():
        chunks.append({"ocr_text": final_ocr, "ground_truth": final_gt})
        
    return chunks

def main():
    print(extract_aligned_chunks("Thes is a historical docummt in English. Céci estun docüment historique en francais! Dies ïstein historisces Dokûment in deutsder Sprache.",
                                 "This is a historical document in English. Ceci est un document historique en français! Dies ist ein historisches Dokument in deutscher Sprache?",
                                 6,
                                 20))


if __name__ == "__main__":
    main()
