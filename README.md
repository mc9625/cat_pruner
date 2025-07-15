# Provenance Pruning Plugin for Cheshire Cat AI

An intelligent document pruning plugin that implements sentence-level relevance filtering based on the Provenance algorithm, with fallback to cosine similarity for maximum reliability.

## 🚀 Features

- **Intelligent Sentence Pruning**: Removes irrelevant sentences while preserving document coherence
- **Multiple Scoring Algorithms**: 
  - Provence Reranker models for best quality
  - Cross-encoder classifiers for high accuracy
  - Cosine similarity as reliable fallback
- **Context Preservation**: 
  - Guaranteed head/tail sentence preservation
  - Neighbor window expansion for reference chains
  - Bonus scoring for sentences with numbers/dates
- **Performance Optimized**:
  - LRU caching with configurable size limits
  - Batch processing for efficiency
  - Multi-device support (CPU, CUDA, Apple MPS)
- **Fully Configurable**: All parameters adjustable via Cheshire Cat admin UI

## 📋 Requirements

```bash
pip install torch sentence-transformers transformers nltk tiktoken scikit-learn
```

The plugin will automatically download required NLTK data on first run.

## ⚙️ Configuration

All settings are configurable through the Cheshire Cat admin interface:

| Setting | Default | Description |
|---------|---------|-------------|
| `enable_pruning` | `true` | Enable/disable the pruning system |
| `keep_ratio` | `0.5` | Fraction of sentences to keep (0.5 = 50%) |
| `min_tokens_for_pruning` | `1500` | Skip pruning for documents smaller than this |
| `preserve_head_tail` | `4` | Number of sentences to always keep at start/end |
| `digit_bonus` | `0.6` | Score bonus for sentences containing numbers |
| `neighbor_window` | `true` | Include adjacent sentences for context |
| `classifier_model` | `cross-encoder/ms-marco-MiniLM-L6-v2` | HuggingFace model for classification |
| `embed_model` | `sentence-transformers/all-MiniLM-L6-v2` | Model for embeddings |
| `cache_max_size` | `100` | Maximum number of cached results |

## 🧠 Supported Models

### Recommended Classifiers
- `cross-encoder/ms-marco-MiniLM-L6-v2` (default, fast)
- `naver/provence-reranker-debertav3-v1` (best quality)
- `microsoft/deberta-v3-base` (high accuracy)

### Embedding Models
- `sentence-transformers/all-MiniLM-L6-v2` (default, fast)
- `sentence-transformers/all-mpnet-base-v2` (better quality)

## 🔧 Available Tools

The plugin provides two diagnostic tools accessible via chat:

### `pruning_status`
Shows current configuration and model status:
```
@pruning_status
```

### `test_pruning`
Test pruning on sample text:
```
@test_pruning "Renaissance art history"
```

## 📊 How It Works

1. **Document Retrieval**: The plugin intercepts retrieved documents before they reach the LLM
2. **Sentence Splitting**: Documents are split into sentences using NLTK
3. **Relevance Scoring**: Each sentence is scored against the user query using:
   - Cross-encoder classifier (if available)
   - Cosine similarity (fallback)
   - Position bonuses (head/tail preservation)
   - Content bonuses (numbers, dates)
4. **Selection**: Top-scoring sentences are selected based on `keep_ratio`
5. **Context Expansion**: Adjacent sentences are included if `neighbor_window` is enabled
6. **Caching**: Results are cached to avoid recomputation

## 🎯 Use Cases

- **Long Document Summarization**: Keep only relevant parts of lengthy texts
- **Narrative Preservation**: Maintain story coherence with head/tail preservation
- **Technical Documentation**: Prioritize sections with specific data/numbers
- **Research Papers**: Extract relevant findings while maintaining context

## ⚡ Performance Tips

- **Conservative Pruning**: Use `keep_ratio` ≥ 0.6 for important documents
- **Aggressive Pruning**: Use `keep_ratio` ≤ 0.4 for large corpora
- **Cache Tuning**: Increase `cache_max_size` for repetitive queries
- **Model Selection**: 
  - Use lighter models on CPU
  - Use Provence models for best quality
  - Disable classifier for pure speed (cosine only)

## 🐛 Troubleshooting

### Models Not Loading
- Check internet connection for model downloads
- Verify HuggingFace token for private models
- Try alternative models if one fails

### High Memory Usage
- Reduce `cache_max_size`
- Use smaller models
- Enable only on specific document types

### Poor Pruning Quality
- Increase `keep_ratio` for more conservative pruning
- Enable `neighbor_window` for better context
- Try different classifier models

## 📝 Example Configuration

For narrative documents with important chronological information:
```json
{
  "keep_ratio": 0.7,
  "preserve_head_tail": 5,
  "digit_bonus": 0.8,
  "neighbor_window": true,
  "classifier_model": "naver/provence-reranker-debertav3-v1"
}
```

For technical documentation with less narrative structure:
```json
{
  "keep_ratio": 0.4,
  "preserve_head_tail": 2,
  "digit_bonus": 0.9,
  "neighbor_window": false,
  "classifier_model": "cross-encoder/ms-marco-MiniLM-L6-v2"
}
```

## 🤝 Contributing

Feel free to submit issues or pull requests on the Cheshire Cat plugins repository.

## 📄 License

This plugin is released under the same license as Cheshire Cat AI.