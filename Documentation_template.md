# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** AmaZoneIn  
**Team Members:** Dakkshesh Kaul, Anushka Chaudhari, Komal Jadhav, Chetan Chavan  
**Submission Date:** 27/09/2026

---

## 1. Executive Summary
*Our approach models business entity resolution as a strict, decoupled two-stage Information Retrieval (IR) pipeline to overcome massive memory (OOM) bottlenecks. We engineered a lightweight weighted blocking engine to cheaply isolate high-probability candidate pairs, followed by a 13-feature LightGBM binary classifier. By processing the dataset in synchronized lockstep chunks, we successfully scored the entire 9.97 million target test rows safely within consumer hardware limits (16 GB RAM).*

---

## 2. Methodology

### 2.1 Problem Analysis
*Exploratory analysis showed substantial noise between Source 1 and the two target sources. True matches may differ because of legal-suffix variations, punctuation, word-order changes, transliteration, abbreviations, DBA/trade names, and incomplete or differently formatted addresses.*

*On the frozen validation split, exact normalized-name blocking recovered only about 25.66% of true pairs, showing that name variations were a major source of missed links. Among true matches, approximately 74.34% had the same country but different normalized names. Exact normalized-address equality was also relatively weak: only about 8.27% of true pairs had the same normalized address.*

*The target data also contains missing business names and addresses, so the pipeline avoids relying on any single field. Country is used as a blocking dimension, while the method remains string-based and does not use external business information.*

### 2.2 Solution Strategy
*We use a staged retrieval-and-ranking architecture:*

1. Normalize and canonicalize business names and addresses.

2. Build candidate sets using multiple complementary blocking signals.

3. Retain at most 100 candidates for each Source 1 entity using a priority heap.

4. Compute 13 pairwise features for the retained candidates.

5. Score candidates using a LightGBM binary classifier.

6. Apply a probability threshold of 0.60 selected using the frozen validation split.

7. Write both the final matches and the exact candidate set used for inference.

**Approach Type:** Blocking + Supervised Classifier / Hybrid Information Retrieval  
**Core Innovation:** A memory-efficient multi-signal blocking pipeline that combines exact, canonicalized, rare-token, and rare-character-gram evidence before LightGBM ranking.

---

## 3. Candidate Generation (Blocking)
*To reduce the comparison space to a manageable candidate set without exceeding memory limits, we utilized a chunked Max-Heap architecture.*

- **Blocking keys used:** Exact Normalized Name, Exact Strong Core Name (suffix stripped and alphabetically sorted), Exact Strong Address, Rare Name/Address Tokens (Frequency <= 100), and Top-1 Rare 3-Character Grams.
- **Candidate pairs generated:** Strictly capped at the Top-100 candidates per Source 1 entity.
- **How you ensured true matches were not lost:** We used a weighted blocking score during the blocking scan. Exact core matches and other strong signals received larger weights, while rare-token and rare 3-gram evidence contributed additional scores to rank candidates in the priority heap before expensive RapidFuzz feature computation.

---

## 4. Matching Model

**Features used:**
- Name features: 
1. RapidFuzz token_set_ratio
2. RapidFuzz token_sort_ratio
3. RapidFuzz ratio
4. RapidFuzz WRatio
5. Name length ratio
6. Name digit Jaccard similarity
- Address features: 
1. RapidFuzz token_set_ratio
2. RapidFuzz token_sort_ratio
3. RapidFuzz ratio
4. RapidFuzz WRatio
5. Address length ratio
6. Address digit Jaccard similarity

- Other: 
1. Blocking score

**Model type:** LightGBM binary classifier  
**Threshold selection method:** Threshold sweep on the frozen validation split using macro F_0.5. The selected final threshold was 0.60.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** 0.825948
- **Common false positives (wrong merges):** False positives were concentrated in cases where businesses shared generic or highly frequent name/address components within the same country. These cases can produce several plausible candidates with similar string-similarity features. Singleton Source 1 entities are particularly sensitive to even one incorrect predicted match because a false merge converts a correctly empty prediction into an incorrect one under the macro metric.
- **Common false negatives (missed matches):** The main source of missed links was strong variation between records belonging to the same business. Examples include DBA/trade names, transliteration differences, reordered words, abbreviated or expanded legal terms, typographical differences, and incomplete or differently formatted addresses. Validation analysis showed that many true matches did not share an exact normalized name, making retrieval rather than classification an important limiting factor.

---

## 6. Conclusion
*The final solution uses a memory-efficient blocking-plus-classification architecture tailored to millions of records and a precision-heavy macro F_0.5 objective. Combining multiple retrieval signals with a 13-feature LightGBM matcher improved robustness to noisy business names and addresses while keeping candidate sets small enough for commodity hardware.*

---

## Appendix

### A. Code Artefacts
*The complete runnable implementation is provided under:*

`code/business_entity_resolution/`

*The source files are stored in `src/`, the trained LightGBM model is stored in `models/`, and the environment is specified by* *`requirements.txt`.*
*The primary end-to-end inference entry point is the V21 test-set submission script, which generates:*
1. output/matching_results.tsv
2. output/candidate_pairs.tsv

*The pipeline reads the three test TSV files, performs global target profiling, processes Source 1 in 200,000-row batches, generates and ranks candidates, computes the 13 matching features, applies the LightGBM model at threshold 0.60, and streams the results to disk.*

### B. Additional Results
*Selected validation findings that guided the final design:*
1. Exact normalized name + country blocking recovered about 25.66% of true pairs.
2. About 74.34% of validation true pairs had the same country but different normalized names.
3. Exact normalized address equality occurred for about 8.27% of true pairs.
4. The V16 LightGBM matcher achieved a best frozen-validation macro F_0.5 of 0.831439 at threshold 0.60.
5. The corresponding V21 final test submission achieved a public leaderboard score of 0.826.

---