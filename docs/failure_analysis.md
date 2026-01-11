# Failure Analysis

This document describes where the model fails, edge cases, and limitations.

---

## Examples Where the Model Fails

### Example 1: Satire and Opinion Pieces

**Input:**
> "The president's latest decision is absolutely ridiculous and everyone knows it. This is the worst policy in American history."

**Model Output:** FAKE (90% confidence)

**Problem:** This could be legitimate opinion journalism, not fake news.

**Root Cause:** The training data conflates emotional/sensational writing style with fake news. Real opinion pieces, satire, and editorial content use similar language patterns.

### Example 2: Short Headlines

**Input:**

### Example 3: Modern Topics (Post-2017)

**Input:**
> "The new COVID-19 vaccine shows 95% efficacy in clinical trials according to Pfizer."

**Model Output:** Topic: -1 (Outlier) or incorrect topic

**Problem:** COVID-19 didn't exist in 2016-2017 training data.

**Root Cause:** The model has no representation of modern topics. Words like "COVID", "vaccine", "Pfizer" are out-of-vocabulary or associated with different contexts.

### Example 4: Quoted Fake Claims in Real News

**Input:**
> "Fact-checkers debunked the viral claim that '5G causes coronavirus.' The claim originated from..."

**Model Output:** FAKE (97% confidence)

**Problem:** This is a fact-checking article (real news) that quotes fake claims.

**Root Cause:** The model detects fake news language patterns regardless of context. Quoting fake claims triggers the fake detector.

---

## Edge Cases That Cause Problems

| Edge Case | Model Behavior | Why |
|-----------|----------------|-----|
| Empty text | Error/crash | No features to extract |
| Non-English text | Random prediction | Trained only on English |
| Very long text (>10k words) | Slow, may timeout | Embedding computation expensive |
| ALL CAPS TEXT | Lower accuracy | Not normalized in preprocessing |
| Text with many URLs | Wrong topic | URLs treated as words |
| Mixed languages | Poor results | Embeddings trained on English |

---

## Limitations of Our Approach

### 1. Style ≠ Truth
- We detect writing patterns, not factual accuracy
- A well-written lie will be classified as "REAL"
- A poorly-written truth may be classified as "FAKE"

### 2. Temporal Bias
- Training data: 2016-2017
- Topics are outdated (2016 election, Obama administration)
- Modern events (COVID, 2020 election, AI) are unknown
- Writing styles may have evolved

### 3. Topic Granularity
- 45-49 topics may still be too many for some use cases
- Some topics overlap (e.g., "Trump Politics" vs "White House")
- Topic names are auto-generated, may not be intuitive

### 4. Binary Classification Oversimplification
- Real news exists on a spectrum: satire, opinion, biased, misleading, false
- Binary fake/real loses nuance
- No distinction between intentional lies and honest mistakes

### 5. No Source Verification
- We don't check if claims are factually true
- We don't verify sources or citations
- We only analyze text patterns

---

## Potential Improvements

| Improvement | Why It Would Help | Effort |
|-------------|-------------------|--------|
| Use nr_topics=50 | More interpretable, less overlap | Low |
| Add 2020+ training data | Cover modern topics | Medium |
| Multi-class labels | Satire, opinion, misleading, false | High |
| Fact-checking API | Verify claims, not just style | High |
| Fine-tune BERT classifier | Better accuracy than TF-IDF | Medium |
| Add source credibility | Consider publisher reputation | Medium |

---

## Topic Differences: Fake vs Real News

**Interesting Finding:** Our topic modeling revealed distinct patterns:

| Fake News Topics | Real News Topics |
|------------------|------------------|
| Conspiracy theories | Policy analysis |
| Sensational claims | Official statements |
| Celebrity gossip | Economic data |
| Health misinformation | International relations |

**Insight:** Fake news tends to focus on emotional, sensational topics while real news covers institutional and policy matters. This aligns with research on misinformation spread.

---

## MLflow Evidence

All failed experiments are documented in MLflow:

**BERTopic experiments:** `BERTopic_Topic_Modeling`
- `bertopic_cluster_5` - Too many topics (180)
- `bertopic_cluster_200` - Too few topics (5), high outliers (78%)
- `bertopic_neighbors_5` - Fragmented topics (120)

**LogReg experiments:** `LogReg_Fake_Detection`
- `logreg_100_features` - Underfitting (accuracy 0.72)
- `logreg_C_0001` - Over-regularized (accuracy 0.65)

View experiments: `mlflow ui --port 5001`

