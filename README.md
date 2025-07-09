# 🔥 Provenance Pruning Plugin

Advanced text pruning plugin for Cheshire Cat AI that reduces RAG hallucinations by intelligently filtering retrieved documents.

## ✨ Features

- **Intelligent Pruning**: Removes irrelevant sentences while preserving context
- **Apple Silicon Optimized**: MPS acceleration for M1/M2/M3 chips
- **Automatic Installation**: All dependencies installed automatically
- **Configurable**: Adjustable compression ratios and thresholds
- **Caching**: Results cached for improved performance

## 🚀 Installation

1. Copy plugin to Cheshire Cat plugins directory
2. Restart Cheshire Cat
3. Enable plugin in admin interface
4. Dependencies install automatically

## 📊 Usage

### Chat Commands

```
pruning status              # Check plugin status
test pruning "your query"   # Test pruning with specific query
```

### Configuration

Access plugin settings in Cheshire Cat admin:

- **Enable Pruning**: Toggle automatic pruning
- **Compression Ratio**: 0.7 (70% compression)
- **Min Tokens**: 1000 (minimum tokens to activate)
- **Model**: sentence-transformers/all-MiniLM-L6-v2

## 🍎 Apple Silicon

Plugin automatically detects Apple Silicon and enables MPS acceleration for optimal performance.

## 🎯 Expected Results

- **70% token reduction** in retrieved documents
- **Reduced hallucinations** in AI responses
- **Faster response times** due to less token processing
- **Better accuracy** with focused context

## 📈 Performance

- **Apple Silicon (MPS)**: ~80ms per document
- **Standard CPU**: ~200ms per document
- **Memory Usage**: ~800MB peak

## 🔧 Troubleshooting

If plugin doesn't work:
1. Check `pruning status` for errors
2. Verify dependencies in logs
3. Restart Cheshire Cat
4. Check admin interface for plugin activation

## 📝 License

MIT License - see LICENSE file for details.