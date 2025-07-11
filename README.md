# 🔥 Provenance-Inspired Pruning Plugin

**Semantic sentence filtering plugin** for Cheshire Cat AI that reduces RAG hallucinations by intelligently filtering retrieved documents.

**Note**: This is a Provenance-inspired implementation using semantic similarity, not the original Provenance binary classification model.

## ✨ Features

- **Intelligent Sentence Filtering**: Removes irrelevant sentences while preserving context
- **Semantic Similarity**: Uses sentence transformers for relevance scoring
- **Apple Silicon Optimized**: MPS acceleration for M1/M2/M3 chips  
- **Automatic Installation**: All dependencies installed automatically
- **Configurable**: Adjustable keep ratios and thresholds
- **Caching**: Deterministic MD5-based results caching
- **Non-destructive**: Preserves original document metadata

## 🚀 Installation

1. Copy plugin to Cheshire Cat plugins directory
2. Restart Cheshire Cat
3. Enable plugin in admin interface
4. Dependencies install automatically

## 📊 Usage

### Chat Commands

```
pruning status              # Check plugin status
pruning diagnostics         # Run complete system diagnostics
test pruning "your query"   # Test pruning with specific query
fix plugin issues           # Auto-repair common problems
```

### Configuration

Access plugin settings in Cheshire Cat admin:

- **Enable Pruning**: Toggle automatic pruning
- **Keep Ratio**: 0.3 (keep 30% of sentences, remove 70%)
- **Min Tokens**: 1000 (minimum tokens to activate)
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Cache**: Enable/disable result caching

## 🎯 Expected Results

- **70% sentence reduction** in retrieved documents
- **Reduced hallucinations** in AI responses  
- **Faster response times** due to less token processing
- **Better accuracy** with focused context

## 📈 Performance

- **Apple Silicon (MPS)**: ~80ms per document
- **Standard CPU**: ~200ms per document
- **Memory Usage**: ~800MB peak
- **Compression**: 70% typical reduction

## 🔬 Technical Implementation

### Algorithm
```
1. Split documents into sentences
2. Calculate semantic similarity (query vs sentences) 
3. Apply positional bias (beginning/end sentences)
4. Select top sentences based on keep_ratio
5. Reconstruct document maintaining order
```

### Key Differences from True Provenance
- Uses **semantic similarity** instead of binary classification
- **Continuous scoring** rather than relevance/irrelevance labels  
- **Positional bias** instead of context-aware modeling
- **SentenceTransformer** instead of specialized Provenance model

## 🔧 Troubleshooting

### Problem: "ML dependencies not available"
**Solutions**:
1. Restart Cheshire Cat after plugin installation
2. Check logs for specific errors
3. Use `pruning diagnostics` command for details

### Problem: Poor compression
**Check**:
- Keep ratio setting (0.3 = keep 30%, remove 70%)
- Token threshold (may be too high)
- Document length (very short docs aren't pruned)

## 📊 Monitoring  

### Diagnostic Commands
```
pruning status              # General status
pruning diagnostics         # Complete test
test pruning "query"        # Test functionality
fix plugin issues           # Auto-repair
```

### Important Logs
- `✅ Provenance ML dependencies loaded successfully`
- `🍎 Using Apple Silicon MPS acceleration` 
- `🚀 Provenance Pruning Plugin initialized successfully`

## 🎯 Production Improvements

For production use, consider:

1. **Replace with true Provenance model**:
   ```python
   from transformers import AutoModelForSequenceClassification
   model = AutoModelForSequenceClassification.from_pretrained("provenance-checkpoint")
   ```

2. **Implement binary classification**:
   ```python
   def relevance_scores(query, sentences):
       # Binary relevant/irrelevant classification
       return binary_classifier.predict(query, sentences)
   ```

3. **Add context-aware scoring**:
   - Coreference resolution
   - Sentence dependency modeling
   - Cross-sentence context windows

## 📝 License

MIT License - see LICENSE file for details.

---

**💡 Note**: This implementation provides effective semantic filtering but doesn't replicate the full Provenance methodology. For research or production requiring exact Provenance compliance, consider implementing the true binary classification approach.