# Data Processing Decisions

## Dataset Overview

**Fake and Real News Dataset** (Kaggle - Clément Bisaillon)

| File | Samples | Label    |
|------|---------|----------|
| True.csv | 21,417 | Real (0) |
| Fake.csv | 23,481 | Fake (1) |
| **Total** | **44,898** | -        |

---

## Class Imbalance Handling

**Imbalance ratio:** 52.3% Fake vs 47.7% Real (mild imbalance)

**Our approach:**
1. **Stratified train/test split** - Maintains class proportions in both sets
2. **No resampling needed** - Imbalance is mild (<60/40)
3. **Weighted evaluation** - Using weighted F1 score

**Justification:** With only ~5% imbalance, resampling techniques (SMOTE, undersampling) would add complexity without significant benefit. Stratification ensures fair evaluation.

---

## Preprocessing Decisions

| Decision | What We Did | Why | Alternatives Considered | Impact |
|----------|-------------|-----|------------------------|--------|
| **Remove duplicates** | Dropped rows with same title+text | Prevent data leakage | Keep all → would inflate accuracy | Removed ~500 duplicates |
| **Clean dates** | Standardized to YYYY-MM-DD | Consistency | Drop date column → lose temporal info | Enabled time-based analysis |
| **Remove image captions** | Regex to remove "Featured image..." | Noise reduction | Keep all text → adds irrelevant words | Cleaner topic keywords |
| **Combine title+text** | Created 'content' column | More context for topics | Title only → too short | Better topic coherence |
| **Min text length** | Removed texts <50 chars | Too short for meaningful analysis | Keep all → noisy topics | Removed ~200 samples |
| **Stopwords removal** | English stopwords in TF-IDF | Focus on meaningful words | No removal → "the", "is" dominate | Better topic keywords |

---

## Train/Test Split

```
Total: 44,898 samples
├── Training: 35,918 (80%)
│   ├── Real: 17,134
│   └── Fake: 18,784
└── Testing: 8,980 (20%)
    ├── Real: 4,283
    └── Fake: 4,697

Split: Stratified by label, random_state=42
```

---

## Key Insights from the Dataset

### Writing Style Differences
- Fake news uses more emotional, sensational language
- Real news uses more formal, institutional language
- This pattern is consistent enough for 98.7% classification accuracy

### Topic Distribution
- Politics dominates both fake and real news (~60%)
- Fake news has more conspiracy-related topics
- Real news has more policy/economic topics

### Temporal Patterns
- Dataset covers 2016-2017 (US election period)
- Topics are heavily influenced by current events
- Model may not generalize to other time periods

### Class Balance
- 52% Fake, 48% Real - nearly balanced
- No aggressive resampling needed
- Stratified split sufficient

### Interesting Findings

| Finding | Implication |
|---------|-------------|
| Fake news articles are longer on average | More content to analyze, but also more noise |
| Real news has more consistent formatting | Easier to identify institutional sources |
| Topic overlap exists | Some topics appear in both fake and real news |
| Outlier documents are often edge cases | Short texts, non-English, or unusual formats |

