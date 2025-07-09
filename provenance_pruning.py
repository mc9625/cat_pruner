"""
Provenance Pruning Plugin for Cheshire Cat AI
Implements advanced pruning to reduce RAG hallucinations
Optimized for PyTorch MPS on Apple Silicon
"""

import time
import platform
import threading
from typing import List, Dict, Any, Optional, Tuple

from cat.mad_hatter.decorators import hook, tool, plugin
from cat.log import log
from pydantic import BaseModel, Field

# Import ML dependencies with graceful error handling
try:
    import torch
    import nltk
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Setup NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # Detect optimal device
    if platform.machine() in ['arm64', 'aarch64'] and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        log.info("🍎 Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        log.info("⚡ Using CUDA acceleration")
    else:
        DEVICE = torch.device("cpu")
        log.info("💻 Using CPU backend")
    
    ML_AVAILABLE = True
    log.info("✅ Provenance ML dependencies loaded successfully")
    
except ImportError as e:
    ML_AVAILABLE = False
    DEVICE = None
    log.error(f"❌ ML dependencies import failed: {e}")
    
except Exception as e:
    ML_AVAILABLE = False
    DEVICE = None
    log.error(f"❌ Unexpected error loading ML dependencies: {e}")

# Global caches
_model_cache = {}
_cache_lock = threading.Lock()

class ProvenanceSettings(BaseModel):
    """Settings for Provenance Pruning Plugin."""
    
    model_config = {"protected_namespaces": ()}
    
    enable_pruning: bool = Field(default=True, description="Enable/disable automatic pruning")
    compression_ratio: float = Field(default=0.7, ge=0.1, le=0.9, description="Text compression percentage (0.1-0.9)")
    min_tokens_for_pruning: int = Field(default=1000, ge=500, le=5000, description="Minimum tokens to activate pruning")
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Similarity threshold to keep sentences")
    preserve_sentences: int = Field(default=2, ge=1, le=5, description="Minimum sentences to always preserve")
    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Model for similarity calculation")
    cache_enabled: bool = Field(default=True, description="Enable results caching")

@plugin
def settings_model():
    return ProvenanceSettings

class ProvenanceClient:
    """Client for pruning with PyTorch optimization."""
    
    def __init__(self, settings: ProvenanceSettings):
        self.settings = settings
        self.model = None
        self._cache = {}
        
    def get_model(self):
        """Get model with caching."""
        if self.model is not None:
            return self.model
            
        with _cache_lock:
            if self.settings.model_name in _model_cache:
                self.model = _model_cache[self.settings.model_name]
                return self.model
                
            model = SentenceTransformer(self.settings.model_name)
            
            if DEVICE and DEVICE.type != "cpu":
                try:
                    model = model.to(DEVICE)
                except Exception as e:
                    log.warning(f"Device optimization failed: {e}")
                    
            _model_cache[self.settings.model_name] = model
            self.model = model
            return model
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            return nltk.sent_tokenize(text)
        except:
            return [s.strip() for s in text.split('.') if s.strip()]
    
    def calculate_similarity(self, query: str, sentences: List[str]) -> List[float]:
        """Calculate query-sentence similarity."""
        model = self.get_model()
        
        try:
            with torch.no_grad():
                query_emb = model.encode([query])
                sentence_embs = model.encode(sentences)
                similarities = cosine_similarity(query_emb, sentence_embs)[0]
                return similarities.tolist()
                
        except Exception as e:
            log.error(f"Similarity calculation failed: {e}")
            return self._fallback_similarity(query, sentences)
    
    def _fallback_similarity(self, query: str, sentences: List[str]) -> List[float]:
        """Lexical similarity fallback."""
        try:
            query_words = set(query.lower().split())
            similarities = []
            
            for sentence in sentences:
                sentence_words = set(sentence.lower().split())
                intersection = len(query_words & sentence_words)
                union = len(query_words | sentence_words)
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
                
            return similarities
            
        except Exception as e:
            log.error(f"Fallback similarity failed: {e}")
            return [0.5] * len(sentences)
    
    def prune_document(self, document: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """Apply pruning to document."""
        if not document.strip():
            return document, {"compression_ratio": 0, "sentences_kept": 0}
        
        # Cache check
        cache_key = hash(f"{document}_{query}_{self.settings.compression_ratio}")
        if self.settings.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        start_time = time.time()
        
        # Split and check length
        sentences = self.split_sentences(document)
        if len(sentences) <= self.settings.preserve_sentences:
            return document, {"compression_ratio": 0, "sentences_kept": len(sentences)}
        
        # Calculate similarity and apply positional bias
        similarities = self.calculate_similarity(query, sentences)
        
        target_count = max(
            self.settings.preserve_sentences,
            int(len(sentences) * self.settings.compression_ratio)
        )
        
        scored = []
        for i, (sentence, sim) in enumerate(zip(sentences, similarities)):
            position_bias = 0.1 if i < 2 else 0.05 if i >= len(sentences) - 2 else 0
            final_score = sim + position_bias
            scored.append((i, sentence, final_score))
        
        # Select and reorder
        scored.sort(key=lambda x: x[2], reverse=True)
        selected = scored[:target_count]
        selected.sort(key=lambda x: x[0])
        
        pruned_text = ' '.join([item[1] for item in selected])
        
        metadata = {
            "compression_ratio": 1 - (len(selected) / len(sentences)),
            "sentences_kept": len(selected),
            "processing_time": time.time() - start_time
        }
        
        result = (pruned_text, metadata)
        
        if self.settings.cache_enabled:
            self._cache[cache_key] = result
            
        return result

# Global client
_client = None

def get_client(cat) -> Optional[ProvenanceClient]:
    """Get configured client."""
    global _client
    
    if not ML_AVAILABLE:
        return None
        
    try:
        settings = cat.mad_hatter.get_plugin().load_settings()
        settings_obj = ProvenanceSettings(**settings)
        
        if _client is None or _client.settings.model_name != settings_obj.model_name:
            _client = ProvenanceClient(settings_obj)
        return _client
    except Exception as e:
        log.error(f"Error getting client: {e}")
        return None

@hook(priority=2)
def before_cat_recalls_declarative_memories(declarative_recall_config, cat):
    """Increase documents for pruning."""
    settings = cat.mad_hatter.get_plugin().load_settings()
    
    if settings.get("enable_pruning", True) and ML_AVAILABLE:
        original_k = declarative_recall_config.get("k", 3)
        declarative_recall_config["k"] = min(original_k * 2, 10)
    
    return declarative_recall_config

@hook(priority=1)
def after_cat_recalls_declarative_memories(declarative_memories, cat):
    """Apply pruning to documents."""
    settings = cat.mad_hatter.get_plugin().load_settings()
    
    if not settings.get("enable_pruning", True) or not declarative_memories:
        return declarative_memories
        
    client = get_client(cat)
    if not client:
        return declarative_memories
    
    try:
        query = cat.working_memory.user_message_json.text
        
        # Check if pruning is needed
        total_text = "\n".join([mem[0].page_content for mem in declarative_memories])
        estimated_tokens = len(total_text.split()) * 1.3
        
        if estimated_tokens < settings.get("min_tokens_for_pruning", 1000):
            return declarative_memories
            
        # Apply pruning
        for memory in declarative_memories:
            original_content = memory[0].page_content
            pruned_content, metadata = client.prune_document(original_content, query)
            memory[0].page_content = pruned_content
            
        # Log result
        final_text = "\n".join([mem[0].page_content for mem in declarative_memories])
        final_tokens = len(final_text.split()) * 1.3
        compression = (estimated_tokens - final_tokens) / estimated_tokens
        
        log.info(f"✅ Pruning: {estimated_tokens:.0f} → {final_tokens:.0f} tokens ({compression:.1%} compression)")
        
        return declarative_memories
        
    except Exception as e:
        log.error(f"Pruning failed: {e}")
        return declarative_memories

@tool
def pruning_status(tool_input, cat):
    """Show pruning system status."""
    settings = cat.mad_hatter.get_plugin().load_settings()
    
    status = [
        "**🔥 Provenance Pruning Status**",
        "",
        "**System:**",
        f"- ML Available: {'✅' if ML_AVAILABLE else '❌'}",
        f"- Device: {DEVICE if DEVICE else 'N/A'}",
        f"- Platform: {platform.machine()}",
        "",
        "**Configuration:**",
        f"- Pruning: {'🟢 Active' if settings.get('enable_pruning', True) else '🔴 Disabled'}",
        f"- Compression: {settings.get('compression_ratio', 0.7):.1%}",
        f"- Token Threshold: {settings.get('min_tokens_for_pruning', 1000):,}",
        f"- Model: {settings.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')}",
        f"- Cache: {'✅' if settings.get('cache_enabled', True) else '❌'}",
    ]
    
    if _client and _client._cache:
        status.append(f"- Cache Entries: {len(_client._cache)}")
    
    if ML_AVAILABLE:
        status.extend([
            "",
            "**🔧 ML Status:**",
            f"- PyTorch: {torch.__version__}",
            f"- Model Cache: {len(_model_cache)} entries",
        ])
        
        if DEVICE and DEVICE.type == "mps":
            status.append(f"- MPS: {'✅' if torch.backends.mps.is_available() else '❌'}")
        elif DEVICE and DEVICE.type == "cuda":
            status.append(f"- CUDA: {'✅' if torch.cuda.is_available() else '❌'}")
    
    return "\n".join(status)

@tool
def test_pruning(test_query, cat):
    """Test pruning on specific query. Input the query to test."""
    if not ML_AVAILABLE:
        return "❌ ML dependencies not available"
        
    client = get_client(cat)
    if not client:
        return "❌ Client not available"
    
    try:
        test_doc = """The training cost of the DeepSeek-V3 model was 5.576 million dollars.
DeepSeek-V3 uses a transformer architecture with efficient attention mechanism.
The DeepSeek company is based in China and specializes in artificial intelligence.
The model was trained on distributed GPU clusters.
Results show competitive performance compared to other models.
The technical paper describes the optimization mechanisms used.
Metrics include BLEU score and accuracy on standard benchmarks."""
        
        pruned_doc, metadata = client.prune_document(test_doc, test_query)
        
        return f"""**🧪 Test Pruning Results**

**Query:** {test_query}

**Results:**
- Compression: {metadata['compression_ratio']:.1%}
- Sentences kept: {metadata['sentences_kept']}
- Time: {metadata['processing_time']:.3f}s

**Pruned Document:**
{pruned_doc}"""
        
    except Exception as e:
        return f"❌ Test error: {str(e)}"

@tool
def pruning_diagnostics(tool_input, cat):
    """Run complete system diagnostics."""
    if not ML_AVAILABLE:
        return "❌ ML dependencies not available"
    
    diagnostics = []
    
    # Test core dependencies
    try:
        diagnostics.append(f"✅ PyTorch {torch.__version__}")
        diagnostics.append("✅ NLTK available")
        diagnostics.append("✅ SentenceTransformers available")
        diagnostics.append("✅ Scikit-learn available")
        diagnostics.append(f"✅ Device: {DEVICE}")
    except Exception as e:
        diagnostics.append(f"❌ Dependencies: {e}")
    
    # Test model
    try:
        client = get_client(cat)
        if client:
            model = client.get_model()
            test_emb = model.encode(["test"])
            diagnostics.append(f"✅ Model: {client.settings.model_name}")
            diagnostics.append(f"✅ Encoding test: {test_emb.shape}")
        else:
            diagnostics.append("❌ Client not available")
    except Exception as e:
        diagnostics.append(f"❌ Model test: {e}")
    
    return "**🔧 System Diagnostics**\n\n" + "\n".join(diagnostics)

@tool
def fix_plugin_issues(tool_input, cat):
    """Try to automatically fix common plugin issues."""
    if not ML_AVAILABLE:
        return """❌ ML dependencies not available.

**Solutions:**
1. Restart Cheshire Cat completely
2. Check logs for specific errors
3. Use `pruning diagnostics` for details"""
    
    fixes = []
    
    try:
        global _client, _model_cache
        _client = None
        _model_cache.clear()
        fixes.append("✅ Caches cleared")
    except Exception as e:
        fixes.append(f"❌ Cache cleanup failed: {e}")
    
    try:
        client = get_client(cat)
        if client:
            model = client.get_model()
            test_emb = model.encode(["test"])
            fixes.append(f"✅ Model test: {test_emb.shape}")
        else:
            fixes.append("❌ Client not available")
    except Exception as e:
        fixes.append(f"❌ Model test failed: {e}")
    
    return f"""**🔧 Repair Results**

{chr(10).join(fixes)}

**Next steps:**
1. Use `pruning status` to verify
2. Test with `test pruning "query"`"""

@hook
def after_cat_bootstrap(cat):
    """Initialize plugin."""
    if ML_AVAILABLE:
        log.info("🚀 Provenance Pruning Plugin initialized successfully")
        
        def preload():
            try:
                client = get_client(cat)
                if client:
                    model = client.get_model()
                    test_emb = model.encode(["test"])
                    log.info(f"✅ Model preloaded and tested: {test_emb.shape}")
            except Exception as e:
                log.error(f"Model preload failed: {e}")
        
        threading.Thread(target=preload, daemon=True).start()
        
        settings = cat.mad_hatter.get_plugin().load_settings()
        if settings.get("enable_pruning", True):
            device_info = f"on {DEVICE}" if DEVICE else "CPU mode"
            log.info(f"🔥 Provenance Pruning active {device_info}")
            
    else:
        log.warning("⚠️ Provenance Plugin: ML dependencies missing")