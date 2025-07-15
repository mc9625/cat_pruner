"""
Provenance-Style Pruning Plugin for Cheshire Cat AI
Implements intelligent sentence-level document pruning based on query relevance.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

from cat.log import log
from cat.mad_hatter.decorators import hook, plugin, tool
from pydantic import BaseModel, Field, ValidationError

# ML imports with graceful fallback
ML_AVAILABLE = True
try:
    import torch
    import nltk
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    # Download NLTK data if needed
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # Tiktoken for accurate token counting
    try:
        import tiktoken
        TOKENIZER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        TOKENIZER = None

    # Device selection
    if platform.machine() in {"arm64", "aarch64"} and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
        DEVICE_STR = "cpu"  # sentence-transformers compatibility
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        DEVICE_STR = "cuda"
    else:
        DEVICE = torch.device("cpu")
        DEVICE_STR = "cpu"
    
    log.info(f"Provenance Pruning: Using {DEVICE} backend")

except Exception as e:
    ML_AVAILABLE = False
    DEVICE = None
    DEVICE_STR = "cpu"
    log.error(f"ML dependencies unavailable: {e}")


class ProvenanceSettings(BaseModel):
    """Plugin settings for intelligent document pruning."""
    
    enable_pruning: bool = Field(
        True, 
        description="Enable intelligent sentence-level pruning"
    )
    keep_ratio: float = Field(
        0.5,
        ge=0.1,
        le=0.95,
        description="Fraction of sentences to keep (0.5 = 50%)",
    )
    min_tokens_for_pruning: int = Field(
        1500,
        ge=200,
        le=8000,
        description="Skip pruning if document has fewer tokens",
    )
    preserve_head_tail: int = Field(
        4,
        ge=0,
        le=10,
        description="Number of sentences to preserve at start/end",
    )
    digit_bonus: float = Field(
        0.60,
        ge=0.0,
        le=1.0,
        description="Score bonus for sentences with numbers/dates",
    )
    neighbor_window: bool = Field(
        True,
        description="Include adjacent sentences for context continuity",
    )
    classifier_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        description="HuggingFace model for relevance classification",
    )
    embed_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for similarity fallback",
    )
    hf_token: str = Field(
        "", 
        description="HuggingFace token for private models",
    )
    cache_enabled: bool = Field(
        True, 
        description="Cache pruning results"
    )
    cache_max_size: int = Field(
        100,
        ge=10,
        le=1000,
        description="Maximum cache entries",
    )

    model_config = {"protected_namespaces": ()}


@plugin
def settings_model():
    return ProvenanceSettings


class PruningClient:
    """Handles document pruning with multiple relevance scoring strategies."""
    
    MAX_BATCH_SIZE = 32
    TOKEN_ESTIMATE_MULTIPLIER = 1.3
    
    def __init__(self, settings: ProvenanceSettings):
        if not ML_AVAILABLE:
            raise RuntimeError("ML dependencies not available")
            
        self.settings = settings
        self._embed_model: Optional[SentenceTransformer] = None
        self._classifier: Optional[Union[TextClassificationPipeline, Any]] = None
        self._cache = OrderedDict()  # LRU cache
        self._model_lock = threading.Lock()
        self._cache_lock = threading.Lock()

    def _ensure_embedder(self) -> SentenceTransformer:
        """Load embedding model with thread safety."""
        if self._embed_model is not None:
            return self._embed_model
            
        with self._model_lock:
            if self._embed_model is None:
                self._embed_model = SentenceTransformer(
                    self.settings.embed_model, 
                    device=DEVICE_STR
                )
                log.info(f"Loaded embedder: {self.settings.embed_model}")
                
        return self._embed_model

    def _ensure_classifier(self) -> Optional[Any]:
        """Load classifier model with thread safety."""
        if not self.settings.classifier_model:
            return None
            
        if self._classifier is not None:
            return self._classifier
            
        with self._model_lock:
            if self._classifier is None:
                self._classifier = self._load_classifier_model()
                
        return self._classifier
    
    def _load_classifier_model(self) -> Optional[Any]:
        """Load appropriate classifier based on model name."""
        model_name = self.settings.classifier_model
        
        # Check for Provence reranker
        if "provence-reranker" in model_name.lower():
            try:
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    token=self.settings.hf_token or None,
                ).to(DEVICE)
                log.info(f"Loaded Provence reranker: {model_name}")
                return model
            except Exception as e:
                log.error(f"Failed to load Provence model: {e}")
                return None
        
        # Standard cross-encoder
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                token=self.settings.hf_token or None
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                token=self.settings.hf_token or None,
            ).to(DEVICE)
            
            pipeline = TextClassificationPipeline(
                model=model,
                tokenizer=tokenizer,
                top_k=1,
                function_to_apply="sigmoid",
                device=0 if DEVICE.type == "cuda" else -1,
            )
            log.info(f"Loaded cross-encoder: {model_name}")
            return pipeline
            
        except Exception as e:
            log.error(f"Failed to load classifier: {e}")
            return None

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            # Fallback to simple split
            return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        if TOKENIZER:
            try:
                return len(TOKENIZER.encode(text))
            except Exception:
                pass
        return int(len(text.split()) * self.TOKEN_ESTIMATE_MULTIPLIER)

    def _score_with_embeddings(self, query: str, sentences: List[str]) -> List[float]:
        """Score sentences using cosine similarity."""
        embedder = self._ensure_embedder()
        
        with torch.no_grad():
            embeddings = embedder.encode(
                [query] + sentences, 
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
        query_emb = embeddings[0:1]
        sent_embs = embeddings[1:]
        scores = cosine_similarity(query_emb, sent_embs)[0]
        
        return scores.tolist()

    def _score_with_classifier(self, query: str, sentences: List[str]) -> Optional[List[float]]:
        """Score sentences using classifier model."""
        classifier = self._ensure_classifier()
        if classifier is None:
            return None
            
        # Handle Provence model
        if hasattr(classifier, 'process'):
            # Provence uses a different API - return None to use fallback
            return None
            
        # Standard classifier pipeline
        try:
            inputs = [{"text": query, "text_pair": s} for s in sentences]
            
            # Process in batches
            all_scores = []
            for i in range(0, len(inputs), self.MAX_BATCH_SIZE):
                batch = inputs[i:i + self.MAX_BATCH_SIZE]
                predictions = classifier(batch)
                
                # Extract scores
                for pred in predictions:
                    score = pred[0]['score'] if isinstance(pred, list) else pred['score']
                    all_scores.append(score)
                    
            return all_scores
            
        except Exception as e:
            log.warning(f"Classifier scoring failed: {e}")
            return None

    def _apply_scoring_bonuses(
        self, 
        sentences: List[str], 
        base_scores: List[float]
    ) -> List[float]:
        """Apply position and content bonuses to scores."""
        enhanced_scores = []
        total = len(sentences)
        
        for idx, (sent, score) in enumerate(zip(sentences, base_scores)):
            # Position bonus for head/tail
            in_head = idx < self.settings.preserve_head_tail
            in_tail = idx >= total - self.settings.preserve_head_tail
            position_bonus = 0.1 if (in_head or in_tail) else 0.0
            
            # Digit bonus
            has_digit = bool(re.search(r'\d', sent))
            digit_bonus = self.settings.digit_bonus if has_digit else 0.0
            
            final_score = score + position_bonus + digit_bonus
            enhanced_scores.append(min(final_score, 1.0))  # Cap at 1.0
            
        return enhanced_scores

    def _select_sentences(
        self, 
        sentences: List[str], 
        scores: List[float]
    ) -> List[int]:
        """Select sentences to keep based on scores and constraints."""
        total = len(sentences)
        
        # Forced indices (head/tail)
        forced_indices = set()
        if self.settings.preserve_head_tail > 0:
            head_count = min(self.settings.preserve_head_tail, total)
            tail_start = max(0, total - self.settings.preserve_head_tail)
            
            forced_indices.update(range(head_count))
            if tail_start < total:
                forced_indices.update(range(tail_start, total))
        
        # Calculate target count
        min_keep = len(forced_indices) + 1
        target_keep = max(min_keep, int(total * self.settings.keep_ratio))
        target_keep = min(target_keep, total)
        
        # Select top scoring sentences
        scored_indices = sorted(
            enumerate(scores), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        selected = set(forced_indices)
        for idx, _ in scored_indices:
            if len(selected) >= target_keep:
                break
            selected.add(idx)
        
        # Apply neighbor window expansion
        if self.settings.neighbor_window:
            expanded = set(selected)
            for idx in selected:
                if idx > 0:
                    expanded.add(idx - 1)
                if idx < total - 1:
                    expanded.add(idx + 1)
            selected = expanded
        
        # Ensure valid indices
        selected = {idx for idx in selected if 0 <= idx < total}
        
        return sorted(list(selected))

    def _get_cache_key(self, document: str, query: str) -> str:
        """Generate cache key."""
        data = {
            "doc": document,
            "query": query,
            "settings": self.settings.model_dump_json()
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()

    def _cache_get(self, key: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Get from cache with LRU eviction."""
        with self._cache_lock:
            if key in self._cache:
                # Move to end (most recent)
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _cache_put(self, key: str, value: Tuple[str, Dict[str, Any]]):
        """Put in cache with size limit."""
        if not self.settings.cache_enabled:
            return
            
        with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            
            # Evict oldest if over limit
            while len(self._cache) > self.settings.cache_max_size:
                self._cache.popitem(last=False)

    def prune(self, document: str, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Prune document based on query relevance.
        Returns (pruned_text, metadata).
        """
        if not document.strip():
            return document, {
                "compression": 0, 
                "sentences_kept": 0, 
                "sentences_total": 0
            }

        # Check cache
        cache_key = self._get_cache_key(document, query)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        start_time = time.time()
        
        # Check for Provence model special case
        classifier = self._ensure_classifier()
        if classifier and hasattr(classifier, 'process'):
            return self._prune_with_provence(
                document, query, classifier, start_time, cache_key
            )
        
        # Standard pruning logic
        return self._prune_standard(document, query, start_time, cache_key)
    
    def _prune_with_provence(
        self, 
        document: str, 
        query: str, 
        model: Any,
        start_time: float,
        cache_key: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Prune using Provence reranker model."""
        try:
            result = model.process(question=query, context=document)
            pruned_text = result['pruned_context']
            
            # Calculate metadata
            original_sentences = self.split_sentences(document)
            pruned_sentences = self.split_sentences(pruned_text)
            
            metadata = {
                "compression": 1 - (len(pruned_sentences) / len(original_sentences)),
                "sentences_kept": len(pruned_sentences),
                "sentences_total": len(original_sentences),
                "processing_time": round(time.time() - start_time, 3),
                "algorithm": "Provence Reranker",
                "reranking_score": result.get('reranking_score', -1)
            }
            
            self._cache_put(cache_key, (pruned_text, metadata))
            return pruned_text, metadata
            
        except Exception as e:
            log.error(f"Provence pruning failed: {e}")
            # Fall back to standard pruning
            return self._prune_standard(document, query, start_time, cache_key)
    
    def _prune_standard(
        self, 
        document: str, 
        query: str,
        start_time: float,
        cache_key: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Standard sentence-wise pruning."""
        sentences = self.split_sentences(document)
        total = len(sentences)
        
        if total == 0:
            return document, {
                "compression": 0,
                "sentences_kept": 0,
                "sentences_total": 0
            }
        
        # Skip pruning for very short documents
        min_sentences = max(5, int(3 / self.settings.keep_ratio))
        if total <= min_sentences:
            return document, {
                "compression": 0,
                "sentences_kept": total,
                "sentences_total": total
            }
        
        # Score sentences
        base_scores = self._score_with_classifier(query, sentences)
        if base_scores is None:
            base_scores = self._score_with_embeddings(query, sentences)
            algorithm = "Cosine Similarity"
        else:
            algorithm = "Cross-Encoder"
        
        # Apply bonuses and select
        final_scores = self._apply_scoring_bonuses(sentences, base_scores)
        selected_indices = self._select_sentences(sentences, final_scores)
        
        # Build pruned text
        pruned_text = " ".join([sentences[i] for i in selected_indices])
        
        metadata = {
            "compression": 1 - (len(selected_indices) / total),
            "sentences_kept": len(selected_indices),
            "sentences_total": total,
            "processing_time": round(time.time() - start_time, 3),
            "algorithm": algorithm,
            "forced_head_tail": min(self.settings.preserve_head_tail * 2, total),
            "neighbor_expansion": self.settings.neighbor_window,
        }
        
        self._cache_put(cache_key, (pruned_text, metadata))
        return pruned_text, metadata


# Global client management
_CLIENT: Optional[PruningClient] = None
_CLIENT_LOCK = threading.Lock()


def get_client(cat) -> Optional[PruningClient]:
    """Get or create pruning client."""
    global _CLIENT
    
    if not ML_AVAILABLE:
        return None
        
    try:
        raw_settings = cat.mad_hatter.get_plugin().load_settings()
        settings = ProvenanceSettings(**raw_settings)
    except ValidationError as e:
        log.error(f"Invalid settings: {e}")
        return None
    
    with _CLIENT_LOCK:
        if _CLIENT is None or _CLIENT.settings != settings:
            try:
                _CLIENT = PruningClient(settings)
            except Exception as e:
                log.error(f"Failed to create client: {e}")
                _CLIENT = None
                
    return _CLIENT


# Hooks
@hook(priority=2)
def before_cat_recalls_declarative_memories(cfg, cat):
    """Increase retrieval for better pruning candidates."""
    try:
        settings = ProvenanceSettings(**cat.mad_hatter.get_plugin().load_settings())
        if settings.enable_pruning:
            original_k = cfg.get("k", 3)
            cfg["k"] = min(original_k * 2, 12)
    except Exception:
        pass
    return cfg


@hook(priority=1)
def after_cat_recalls_declarative_memories(memories, cat):
    """Apply intelligent pruning to recalled memories."""
    if not memories:
        return memories
        
    client = get_client(cat)
    if client is None or not client.settings.enable_pruning:
        return memories
    
    # Check total token count
    try:
        total_text = "\n".join([doc.page_content for doc, _ in memories])
        if client.estimate_tokens(total_text) < client.settings.min_tokens_for_pruning:
            return memories
    except Exception:
        pass
    
    # Prune each memory
    query = cat.working_memory.user_message_json.text
    pruned_memories = []
    
    total_sentences = 0
    kept_sentences = 0
    total_time = 0
    
    for doc, score in memories:
        new_doc = deepcopy(doc)
        pruned_text, metadata = client.prune(doc.page_content, query)
        new_doc.page_content = pruned_text
        pruned_memories.append((new_doc, score))
        
        total_sentences += metadata["sentences_total"]
        kept_sentences += metadata["sentences_kept"]
        total_time += metadata["processing_time"]
    
    # Log summary
    if total_sentences > 0:
        compression = 1 - (kept_sentences / total_sentences)
        log.info(
            f"Pruning complete: {kept_sentences}/{total_sentences} sentences "
            f"({compression:.1%} compression) in {total_time:.3f}s"
        )
    
    return pruned_memories


# Tools
@tool
def pruning_status(tool_input, cat):
    """Get pruning system status."""
    if not ML_AVAILABLE:
        return "ML dependencies not available - pruning disabled."
    
    client = get_client(cat)
    if client is None:
        return "Pruning client not initialized. Check configuration."
    
    s = client.settings
    
    # Check classifier status
    classifier_info = "Not configured"
    algorithm = "Cosine Similarity"
    
    if s.classifier_model:
        try:
            clf = client._ensure_classifier()
            if clf is not None:
                if hasattr(clf, 'process'):
                    algorithm = "Provence Reranker"
                else:
                    algorithm = "Cross-Encoder"
                classifier_info = f"Active: {s.classifier_model}"
            else:
                classifier_info = f"Failed to load: {s.classifier_model}"
        except Exception:
            classifier_info = "Error loading classifier"
    
    status = f"""**Provenance Pruning Status**

**Configuration:**
- Enabled: {'Yes' if s.enable_pruning else 'No'}
- Algorithm: {algorithm}
- Keep ratio: {s.keep_ratio:.1%}
- Min tokens: {s.min_tokens_for_pruning:,}
- Preserve head/tail: {s.preserve_head_tail} sentences
- Digit bonus: {s.digit_bonus:.1f}
- Neighbor window: {'Yes' if s.neighbor_window else 'No'}
- Cache: {len(client._cache)}/{s.cache_max_size} entries

**Models:**
- Classifier: {classifier_info}
- Embedder: {s.embed_model}
- Device: {DEVICE}
"""
    
    return status


@tool  
def test_pruning(query: str, cat):
    """Test pruning on sample text."""
    if not ML_AVAILABLE:
        return "ML dependencies not available."
    
    client = get_client(cat)
    if client is None:
        return "Client not available."
    
    # Sample document
    sample = """The Renaissance began in Italy during the 14th century. 
    Florence became the cultural center of this movement in 1420.
    Artists like Michelangelo created masterpieces between 1475 and 1564.
    The printing press was invented by Gutenberg in 1440.
    This innovation spread knowledge across Europe rapidly.
    By 1500, over 20 million books had been printed.
    The Renaissance influenced art, science, and philosophy.
    Many ideas from this period still shape modern thinking."""
    
    pruned, metadata = client.prune(sample, query)
    
    return f"""**Pruning Test Results**

Query: "{query}"

Compression: {metadata['compression']:.1%}
Sentences: {metadata['sentences_kept']}/{metadata['sentences_total']}
Time: {metadata['processing_time']:.3f}s
Algorithm: {metadata['algorithm']}

**Pruned Text:**
{pruned}"""


@hook
def after_cat_bootstrap(cat):
    """Initialize models on startup."""
    if not ML_AVAILABLE:
        return
    
    def preload():
        try:
            client = get_client(cat)
            if client:
                # Preload models
                client._ensure_embedder()
                if client.settings.classifier_model:
                    client._ensure_classifier()
                log.info("Provenance pruning models loaded")
        except Exception as e:
            log.error(f"Model preloading failed: {e}")
    
    threading.Thread(target=preload, daemon=True).start()