## BART-BASE:

```json
{"document_id":"BEHI-1875-09-25-a-p0012_par7","chunk_idx":0,"dataset":"impresso-snippets","ocr_text":"It has fillen under our notice that a well-known democratic friend, who\nhas laboured earnestly and long in behalf of human freedom, has fallen by\nthe wayside. He has a large family, and cannot, without assistance, free\nhimself from th. diffliulties by whieh he is beset. As a matter of duty, a\nfew friends have already rallied round him, but it is beyond their power to\ndo all that is required; they are, therefore, compelled to make an appeal to\nfriends. We are assured that this is a thoroughly genuine and deserving\ncase. All remittances and communications to be addressed to the secretary,\nMt. J. B. Leno, 50, Bookseller's-row, Strand, W.C.","ground_truth":"It has fallen under our notice that a well-known democratic friend, who\nhas laboured earnestly and long in behalf of human freedom, has fallen by\nthe wayside. He has a large family, and cannot, without assistance, free\nhimself from the difficulties by which he is beset. As a matter of duty, a\nfew friends have already rallied round him, but it is beyond their power to\ndo all that is required; they are, therefore, compelled to make an appeal to\nfriends. We are assured that this is a thoroughly genuine and deserving\ncase. All remittances and communications to be addressed to the secretary,\nMr. J. B. Leno, 50, Bookseller's-row, Strand, W.C.","model_output":"It has come under our notice that a well-known democratic friend, who\nhas laboured earnestly and long in behalf of human freedom, has fallen by\nthe wayside. He has a large family, and cannot, without assistance, free\nhimself from the diffliulties by which he is beset. As a matter of duty, a\nfew friends have already rallied round him, but it is beyond their power to\ndo all that is required; they are, therefore, compelled to make an appeal to\nfriends. We are assured that this is a thoroughly genuine and deserving\ncase. All remittances and communications to be addressed to the secretary,\nMt. J. B. Leno, 50, Bookseller's‐row, Strand, W.C."}
```

OCR: It has fillen under our notice

Ground Truth: It has fallen under our notice

Model Output: It has come under our notice

My explanation: The phrase "come under our notice" is more common in English literature than "fallen under our notice." The model decided to rewrite our sentence to make it sound more natural.

```json
{"document_id":"ALST-1835-05-05-a-p0003_par21","chunk_idx":0,"dataset":"impresso-snippets","ocr_text":"THE LATE Mn. HERER.—In addilion to the\nnotice lately given in the English newspapers about the\nfamous bibliophile, It. Lieber, Esq., you will see in the\ns' Dagblad van Gravenhage, of this morning that the state-\nment does by no means appear exaggerated here, as that\ngentleman has been known, during his residence in this\nplace, to spend his time at the Royal Library, daily, from\nseven in the morning till seven in the evening, comparing\nand compiling ancient and rare works, without taking any\nmeals in the course of the twelve hours.—Letter from the\nHague.","ground_truth":"THE LATE MR. HEBER.—In addition to the\nnotice lately given in the English newspapers about the\nfamous bibliophile, R. Heber, Esq., you will see in the\ns' Dagblad van Gravenhage, of this morning that the state¬\nment does by no means appear exaggerated here, as that\ngentleman has been known, during his residence in this\nplace, to spend his time at the Royal Library, daily, from\nseven in the morning till seven in the evening, comparing\nand compiling ancient and rare works, without taking any\nmeals in the course of the twelve hours.—Letter from the\nHague.","model_output":"THE LATE MRS. HERER.—In addilion to the\nnotice lately given in the English newspapers about the\nfamous bibliophile, H. Lieber, Esq., you will see in the\ns' Dagblad van Gravenhage, of this morning that the state¬\nment does by no means appear exaggerated here, as that\ngentleman has been known, during his residence in this\nplace, to spend his time at the Royal Library, daily, from\nseven in the morning till seven in the evening, comparing\nand compiling ancient and rare works, without taking any\nmeals in the course of the twelve hours.—Letter from the\nHague."}

```

OCR: THE LATE Mn. HERER

Ground Truth: THE LATE MR. HEBER

Model Output: THE LATE MRS. HERER

Explanation: `HERER` is not a common English name in its vocabulary. The model relied on statistical guesswork. It saw THE LATE [Title] [Unknown Word], and it guessed MRS. instead of `MR..` It completely ignored the fact that `Mn.` is only one character away from `MR..`



## BART-LARGE:
[Data sample](model_eval_logs/model_bart-base-ocr-icdar2017_eval_results.parquet)

### 1. The "Paraphrase" Hallucinations

bart-large aggressively replaced perfectly valid historical words with modern synonyms, completely ignoring the characters it was fed:

GT: `...to slay the innocent.` -> Output: `...to kill the innocent.`

GT: `...debtor to his imagination..`. -> Output: `...indebted to his Intelligence...`

GT: `...record the successes...` -> Output: `...record the results...`

The model saw the OCR text, understood the meaning of the sentence, and then generated what it calculated to be the most statistically probable modern English phrasing. It completely discarded the lexical constraints. To a metric like Character Error Rate (CER), replacing "slay" with "kill" is a massive penalty.


### 2. Contextual Over-Correction
The model is looking at the wider context of the paragraph and forcing concepts to match, even if the characters don't align.

GT: `...by the bumps on her husband's head...`

Output: `...by the bumps on her husband's face...`

GT: R. is truly in a deplorable situation...

Output: T. R. is truly in a deplorable situation... (It hallucinated the "T." here because "T. R." is mentioned much later in the text).

### 3. Complete Surrender to Noise
When the OCR is so corrupted that bart-large cannot instantly guess the semantic meaning, its massive attention mechanism collapses, and it hallucinates total nonsense rather than attempting a character-by-character repair:

GT: `...nor to expiscate an invitation from him...`

Output: `...nor to explicate an invitation in any hint...`

(It didn't know the rare Scottish word "expiscate", so it invented a grammatically acceptable, but completely wrong, English phrase).