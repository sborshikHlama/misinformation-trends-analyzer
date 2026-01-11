# Fake News Topic Analyzer - Business Report

## A. ML Problem Framing

### Business Goal

**Scenario:** A media monitoring company needs to help fact-checkers and journalists quickly analyze incoming news articles. The system should:

1. **Identify the topic** of an article (politics, health, economy, etc.)
2. **Detect writing style patterns** associated with fake vs real news
3. **Provide confidence scores** to prioritize manual review

**Why it matters:** Fact-checkers receive thousands of articles daily. Automated topic classification and style detection helps them focus on high-risk content first.

### ML Problem Statement

| Task | Type | Method |
|------|------|--------|
| Topic Identification | Unsupervised Learning | BERTopic |
| Style Classification | Supervised Learning | TF-IDF + Logistic Regression |

### Success Metrics

| Metric Type | Metric | Target | Achieved |
|-------------|--------|--------|----------|
| ML - Topics | Number of meaningful topics | 30-100 | **45-49** ✅ |
| ML - Topics | Outlier ratio | <30% | **12%** ✅ |
| ML - Topics | Topic diversity | >0.6 | **0.71** ✅ |
| ML - Classification | Accuracy | >90% | **98%** ✅ |
| ML - Classification | F1 Score | >0.90 | **98%** ✅ |
| ML - Classification | ROC-AUC | >0.90 | **99%** ✅ |
| Business | API Response Time | <2s | **<1s** ✅ |

*All metrics tracked in MLflow: `mlflow ui --port 5001`*

### Stakeholders

| Stakeholder | How They Use the Solution |
|-------------|---------------------------|
| Fact-Checker | Quick screening of articles before manual review |
| Media Analyst | Understanding topic distribution in news sources |
| News Editor | Flagging potentially problematic content |
| Researcher | Studying patterns in misinformation |

---

## Related Documentation

- [Data Processing](data_processing.md) - Dataset overview and preprocessing decisions
- [Model Development](model_development.md) - BERTopic and LogReg experiments with MLflow
- [Failed Approaches](failed_approaches.md) - What didn't work and why (5 failed experiments)
- [Failure Analysis](failure_analysis.md) - Model limitations and edge cases

