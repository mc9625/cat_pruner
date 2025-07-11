"""
Provenance‑Style Pruning Plugin for **Cheshire Cat AI** - FIXED VERSION
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import threading
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from cat.log import log
from cat.mad_hatter.decorators import hook, plugin, tool
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Optional imports – guarded so that Cat boots even if ML stack is missing
# ---------------------------------------------------------------------------
ML_AVAILABLE = True
try:
    import torch
    import nltk
    from sklearn.metrics.pairwise import cosine_similarity
    from sentence_transformers import SentenceTransformer
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Pipeline,
        TextClassificationPipeline,
    )

    # NLTK sentence split
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # FIXED: tiktoken with stable encoding
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")  # Always available
    except Exception:
        enc = None

    # IMPROVED: Device detection with better compatibility
    if (
        platform.machine() in {"arm64", "aarch64"}
        and torch.backends.mps.is_available()
    ):
        DEVICE = torch.device("mps")
        DEVICE_STR = "cpu"  # For sentence-transformers compatibility
        log.info("🍎 Using Apple Silicon MPS backend")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        DEVICE_STR = "cuda"
        log.info("⚡ Using CUDA backend")
    else:
        DEVICE = torch.device("cpu")
        DEVICE_STR = "cpu"
        log.info("💻 Using CPU backend")
except Exception as e:
    ML_AVAILABLE = False
    DEVICE = None
    DEVICE_STR = "cpu"
    log.warning(f"⚠️ ML stack unavailable: {e}")

# ---------------------------------------------------------------------------
# IMPROVED: Settings with new parameters for enhanced features
# ---------------------------------------------------------------------------
class ProvenanceSettings(BaseModel):
    """Plugin settings editable from the Admin UI."""

    enable_pruning: bool = Field(
        True, description="Enable/disable sentence‑level pruning"
    )
    keep_ratio: float = Field(
        0.5,  # CHANGED: More conservative default
        ge=0.1,
        le=0.95,
        description="Fraction of sentences to keep after scoring",
    )
    min_tokens_for_pruning: int = Field(
        1500,  # CHANGED: Higher threshold for better content preservation
        ge=200,
        le=8000,
        description="Skip pruning if recalled context has fewer tokens.",
    )
    preserve_head_tail: int = Field(
        4,  # CHANGED: More sentences preserved
        ge=0,
        le=10,
        description="Sentences GUARANTEED preserved from doc head & tail.",
    )
    digit_bonus: float = Field(
        0.25,  # NEW: Bonus for sentences with numbers
        ge=0.0,
        le=1.0,
        description="Score bonus for sentences containing digits/numbers.",
    )
    neighbor_window: bool = Field(
        True,  # NEW: Enable neighbor window expansion
        description="Expand selection to include adjacent sentences.",
    )
    classifier_model: str = Field(
        "", description="HF model id for relevance classifier (binary)."
    )
    embed_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model for cosine fallback.",
    )
    hf_token: str = Field(
        "", description="HF access token (if model is private)."
    )
    cache_enabled: bool = Field(True, description="Cache prune results in‑memory")

    model_config = {"protected_namespaces": ()}


@plugin
def settings_model():
    return ProvenanceSettings

# ---------------------------------------------------------------------------
# FIXED: Enhanced PruningClient with all improvements
# ---------------------------------------------------------------------------
class PruningClient:
    def __init__(self, settings: ProvenanceSettings):
        if not ML_AVAILABLE:
            raise RuntimeError("ML stack missing – cannot create client")
        self.settings = settings
        self._embed_model: Optional[SentenceTransformer] = None
        self._clf_pipeline: Optional[Pipeline] = None
        self._cache: Dict[str, Tuple[str, Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _load_embedder(self) -> SentenceTransformer:
        if self._embed_model is not None:
            return self._embed_model
        # FIXED: Device string compatibility
        model = SentenceTransformer(
            self.settings.embed_model, 
            device=DEVICE_STR
        )
        self._embed_model = model
        log.info(f"🧩 Embedding model loaded: {self.settings.embed_model}")
        return model

    def _load_classifier(self) -> Optional[Pipeline]:
        if not self.settings.classifier_model:
            return None
        if self._clf_pipeline is not None:
            return self._clf_pipeline
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.settings.classifier_model, token=self.settings.hf_token or None
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.classifier_model,
                token=self.settings.hf_token or None,
            ).to(DEVICE or "cpu")
            pipe = TextClassificationPipeline(
                model=model,
                tokenizer=tokenizer,
                return_all_scores=False,
                function_to_apply="sigmoid",
                device=0 if DEVICE and DEVICE.type == "cuda" else -1,
            )
            self._clf_pipeline = pipe
            log.info(f"🔬 Classifier loaded: {self.settings.classifier_model}")
            return pipe
        except Exception as e:
            log.warning(f"Classifier load failed → fallback ({e})")
            self._clf_pipeline = None
            return None

    @staticmethod
    def _sentence_split(text: str) -> List[str]:
        try:
            return nltk.sent_tokenize(text)
        except Exception:
            return [s.strip() for s in text.split(".") if s.strip()]

    def _token_len(self, text: str) -> int:
        if enc:
            try:
                return len(enc.encode(text))
            except Exception:
                pass
        return int(len(text.split()) * 1.3)

    def _cosine_scores(self, query: str, sentences: List[str]) -> List[float]:
        embedder = self._load_embedder()
        with torch.no_grad():
            emb = embedder.encode([query] + sentences, normalize_embeddings=True)
        query_emb, sent_emb = emb[0:1], emb[1:]
        scores = cosine_similarity(query_emb, sent_emb)[0]
        return scores.tolist()

    def _classifier_scores(self, query: str, sentences: List[str]) -> Optional[List[float]]:
        pipe = self._load_classifier()
        if pipe is None:
            return None
        try:
            # FIXED: Correct input format for TextClassificationPipeline
            inputs = [{"text": query, "text_pair": s} for s in sentences]
            preds = pipe(inputs, batch_size=min(32, len(inputs)))
            return [p["score"] for p in preds]
        except Exception as e:
            log.warning(f"Classifier inference failed → fallback ({e})")
            return None

    def _enhanced_scoring(self, query: str, sentences: List[str]) -> List[float]:
        """Enhanced scoring with digit bonus and positional bias."""
        # Base semantic scores
        scores = self._classifier_scores(query, sentences)
        if scores is None:
            scores = self._cosine_scores(query, sentences)
        
        total = len(sentences)
        bias_head_tail = self.settings.preserve_head_tail
        enhanced_scores = []
        
        for idx, (sent, base_score) in enumerate(zip(sentences, scores)):
            # Positional bias for head/tail
            pos_bonus = 0.1 if idx < bias_head_tail or idx >= total - bias_head_tail else 0.0
            
            # NEW: Digit bonus for sentences with numbers/dates/quantities
            has_digit = bool(re.search(r'\d', sent))
            digit_bonus = self.settings.digit_bonus if has_digit else 0.0
            
            final_score = base_score + pos_bonus + digit_bonus
            enhanced_scores.append(final_score)
        
        return enhanced_scores

    def _select_with_guarantees(self, sentences: List[str], scores: List[float]) -> List[int]:
        """Select sentences with guaranteed head/tail preservation and neighbor window."""
        total = len(sentences)
        bias_head_tail = self.settings.preserve_head_tail
        
        # FIXED: Guarantee head/tail preservation
        forced_idx = set()
        if bias_head_tail > 0:
            forced_idx.update(range(min(bias_head_tail, total)))  # Head
            if total > bias_head_tail:
                forced_idx.update(range(max(0, total - bias_head_tail), total))  # Tail
        
        # Calculate target keep count
        keep = max(len(forced_idx) + 1, int(total * self.settings.keep_ratio))
        keep = min(keep, total)  # Don't exceed total
        
        # Score-based selection for remaining slots
        scored = [(idx, sent, score) for idx, (sent, score) in enumerate(zip(sentences, scores))]
        scored.sort(key=lambda x: x[2], reverse=True)
        
        selected_idx = set(forced_idx)
        for idx, _, _ in scored:
            if len(selected_idx) >= keep:
                break
            selected_idx.add(idx)
        
        # NEW: Neighbor window expansion
        if self.settings.neighbor_window:
            expanded_idx = set(selected_idx)
            for idx in list(selected_idx):
                if idx > 0:
                    expanded_idx.add(idx - 1)
                if idx < total - 1:
                    expanded_idx.add(idx + 1)
            selected_idx = expanded_idx
        
        # Ensure we don't exceed bounds
        selected_idx = {idx for idx in selected_idx if 0 <= idx < total}
        
        return sorted(list(selected_idx))

    def prune(self, document: str, query: str) -> Tuple[str, Dict[str, Any]]:
        if not document.strip():
            return document, {"compression": 0, "sentences_kept": 0, "sentences_total": 0}

        # Deterministic cache key with all settings
        if self.settings.cache_enabled:
            cache_key = hashlib.sha256(
                json.dumps([
                    document,
                    query,
                    self.settings.keep_ratio,
                    self.settings.digit_bonus,
                    self.settings.neighbor_window,
                    self.settings.preserve_head_tail,
                    self.settings.classifier_model,
                    self.settings.embed_model,
                ], sort_keys=True).encode()
            ).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]

        start = time.time()
        sentences = self._sentence_split(document)
        total = len(sentences)
        
        if total == 0:
            return document, {"compression": 0, "sentences_kept": 0, "sentences_total": 0}
        
        # IMPROVED: Adaptive pruning - skip if document too small relative to keep_ratio
        min_sentences_for_pruning = max(5, int(3 / self.settings.keep_ratio))
        if total <= min_sentences_for_pruning:
            return document, {"compression": 0, "sentences_kept": total, "sentences_total": total}

        # Enhanced scoring and selection
        scores = self._enhanced_scoring(query, sentences)
        selected_idx = self._select_with_guarantees(sentences, scores)
        
        # Reconstruct text preserving order
        pruned_text = " ".join([sentences[i] for i in selected_idx])

        meta = {
            "compression": 1 - (len(selected_idx) / total),
            "sentences_kept": len(selected_idx),
            "sentences_total": total,
            "processing_time": round(time.time() - start, 3),
            "forced_head_tail": min(self.settings.preserve_head_tail * 2, total),
            "neighbor_expansion": self.settings.neighbor_window,
        }
        
        if self.settings.cache_enabled:
            self._cache[cache_key] = (pruned_text, meta)
        
        return pruned_text, meta


# ---------------------------------------------------------------------------
# Global client management
# ---------------------------------------------------------------------------
_CLIENT: Optional[PruningClient] = None

def _get_client(cat) -> Optional[PruningClient]:
    global _CLIENT
    if not ML_AVAILABLE:
        return None
    try:
        raw = cat.mad_hatter.get_plugin().load_settings()
        settings = ProvenanceSettings(**raw)
    except ValidationError as e:
        log.error(f"Pruning settings invalid: {e}")
        return None

    if _CLIENT is None or _CLIENT.settings != settings:
        try:
            _CLIENT = PruningClient(settings)
        except Exception as e:
            log.error(f"Cannot init PruningClient: {e}")
            _CLIENT = None
    return _CLIENT

# ---------------------------------------------------------------------------
# IMPROVED: Hooks with better logging
# ---------------------------------------------------------------------------
@hook(priority=2)
def before_cat_recalls_declarative_memories(cfg, cat):
    try:
        settings = ProvenanceSettings(**cat.mad_hatter.get_plugin().load_settings())
        if settings.enable_pruning:
            original_k = cfg.get("k", 3)
            cfg["k"] = min(original_k * 2, 12)  # Slightly higher for more content
            log.debug(f"🔍 Increased retrieval: {original_k} → {cfg['k']}")
    except Exception:
        pass
    return cfg

@hook(priority=1)
def after_cat_recalls_declarative_memories(memories, cat):
    """Apply sentence‑level pruning with global compression tracking."""
    if not memories:
        return memories
        
    client = _get_client(cat)
    if client is None:
        return memories

    settings = client.settings
    if not settings.enable_pruning:
        return memories

    # Check if pruning needed based on total tokens
    try:
        total_text = "\n".join([doc.page_content for doc, _ in memories])
        if client._token_len(total_text) < settings.min_tokens_for_pruning:
            log.debug(f"📝 Skipping pruning: {client._token_len(total_text)} < {settings.min_tokens_for_pruning} tokens")
            return memories
    except Exception:
        pass

    query = cat.working_memory.user_message_json.text
    pruned_memories = []
    
    # FIXED: Global compression tracking
    total_sentences = 0
    kept_sentences = 0
    total_processing_time = 0
    
    for doc, score in memories:
        new_doc = deepcopy(doc)
        pruned, meta = client.prune(doc.page_content, query)
        new_doc.page_content = pruned
        pruned_memories.append((new_doc, score))
        
        # Accumulate stats
        total_sentences += meta["sentences_total"]
        kept_sentences += meta["sentences_kept"]
        total_processing_time += meta["processing_time"]

    # FIXED: Accurate global compression logging
    if total_sentences > 0:
        global_compression = 1 - (kept_sentences / total_sentences)
        log.info(
            f"✂️ Enhanced Pruning: {kept_sentences}/{total_sentences} sentences "
            f"({global_compression:.1%} compression) in {total_processing_time:.3f}s"
        )
    
    return pruned_memories

# ---------------------------------------------------------------------------
# ENHANCED: Tools with better diagnostics
# ---------------------------------------------------------------------------
@tool
def pruning_status(tool_input, cat):
    """Return comprehensive pruning system status."""
    if not ML_AVAILABLE:
        return "❌ ML stack not available – pruning disabled."
    
    client = _get_client(cat)
    if client is None:
        return "⚠️ Pruning client not initialised. Check logs."
    
    s = client.settings
    out = [
        "**🪄 Enhanced Provenance Pruning Status**",
        "",
        "**🔧 Configuration:**",
        f"- Enabled: {'✅' if s.enable_pruning else '❌'}",
        f"- Device: {DEVICE} ({DEVICE_STR})",
        f"- Keep ratio: {s.keep_ratio:.1%}",
        f"- Min tokens: {s.min_tokens_for_pruning:,}",
        f"- Head/tail preserve: {s.preserve_head_tail}",
        f"- Digit bonus: {s.digit_bonus}",
        f"- Neighbor window: {'✅' if s.neighbor_window else '❌'}",
        "",
        "**🤖 Models:**",
        f"- Classifier: {s.classifier_model or 'None (cosine fallback)'}",
        f"- Embedder: {s.embed_model}",
        f"- Cache entries: {len(client._cache)}",
    ]
    return "\n".join(out)

@tool
def pruning_diagnostics(tool_input, cat):
    """Run comprehensive system diagnostics."""
    if not ML_AVAILABLE:
        return "❌ ML stack missing."
    
    client = _get_client(cat)
    if client is None:
        return "❌ Cannot init client – see logs."
    
    diag = ["**🔧 Enhanced Diagnostics**", ""]
    
    # System info
    import torch as _t
    diag.append(f"**System:**")
    diag.append(f"- PyTorch: {_t.__version__}")
    diag.append(f"- Device: {DEVICE}")
    diag.append(f"- Device String: {DEVICE_STR}")
    diag.append("")
    
    # Model tests
    diag.append("**Models:**")
    try:
        embedder = client._load_embedder()
        test_emb = embedder.encode(["test sentence"])
        diag.append(f"- Embedder: ✅ {test_emb.shape}")
    except Exception as e:
        diag.append(f"- Embedder: ❌ {e}")
    
    if client.settings.classifier_model:
        try:
            clf = client._load_classifier()
            if clf:
                # Test with correct format
                test_input = [{"text": "test query", "text_pair": "test sentence"}]
                result = clf(test_input)
                diag.append(f"- Classifier: ✅ score {result[0]['score']:.3f}")
            else:
                diag.append("- Classifier: ⚠️ Not loaded (will use cosine)")
        except Exception as e:
            diag.append(f"- Classifier: ❌ {e}")
    else:
        diag.append("- Classifier: 📝 Not configured (using cosine)")
    
    diag.append("")
    diag.append("**Features:**")
    diag.append(f"- Digit bonus: {'✅' if client.settings.digit_bonus > 0 else '❌'}")
    diag.append(f"- Neighbor window: {'✅' if client.settings.neighbor_window else '❌'}")
    diag.append(f"- Head/tail preserve: {client.settings.preserve_head_tail} sentences")
    
    return "\n".join(diag)

@tool
def test_enhanced_pruning(query: str, cat):
    """Test enhanced pruning features on sample text with numbers and structure."""
    if not ML_AVAILABLE:
        return "❌ ML stack missing."
    
    client = _get_client(cat)
    if client is None:
        return "❌ Client not available."
    
    # Sample text with numbers, structure, and narrative elements
    doc = (
        "Nel 1511 Erasmo da Rotterdam scrive l'Elogio della pazzia. "
        "L'opera critica la società del tempo con ironia e satira. "
        "Benvenuto Cellini nasce nel 1500 a Firenze da Giovanni. "
        "La sua autobiografia contiene 142 capitoli e copre 71 anni di vita. "
        "Michelangelo scolpisce il David tra il 1501 e il 1504. "
        "L'opera è alta 5,17 metri e pesa circa 6 tonnellate. "
        "La tecnica dell'oreficeria richiede 10 parti di rame e 2 di stagno. "
        "Cellini perfeziona questa proporzione nella sua bottega fiorentina. "
        "Il Rinascimento italiano vede fiorire arte, letteratura e filosofia. "
        "Molti artisti viaggiano tra Roma, Firenze e Venezia per lavoro."
    )
    
    pruned, meta = client.prune(doc, query)
    
    # Analysis
    original_sentences = client._sentence_split(doc)
    kept_sentences = client._sentence_split(pruned)
    
    digit_sentences = [s for s in original_sentences if re.search(r'\d', s)]
    digit_kept = [s for s in kept_sentences if re.search(r'\d', s)]
    
    return (
        f"**🧪 Enhanced Pruning Test**\n\n"
        f"**Query:** {query}\n\n"
        f"**Results:**\n"
        f"- Sentences: {meta['sentences_kept']}/{meta['sentences_total']} "
        f"({meta['compression']:.1%} compression)\n"
        f"- Processing: {meta['processing_time']:.3f}s\n"
        f"- Digit sentences kept: {len(digit_kept)}/{len(digit_sentences)}\n"
        f"- Head/tail preserved: {meta.get('forced_head_tail', 0)}\n"
        f"- Neighbor expansion: {'✅' if meta.get('neighbor_expansion') else '❌'}\n\n"
        f"**Pruned Text:**\n{pruned}"
    )

# ---------------------------------------------------------------------------
# Startup hook with enhanced preloading
# ---------------------------------------------------------------------------
@hook
def after_cat_bootstrap(cat):
    if not ML_AVAILABLE:
        log.warning("Enhanced Pruning: ML stack unavailable. Skipping preload.")
        return

    def _preload():
        try:
            client = _get_client(cat)
            if client:
                # Preload embedder
                client._load_embedder()
                log.debug("🧩 Embedder preloaded")
                
                # Preload classifier if configured
                if client.settings.classifier_model:
                    clf = client._load_classifier()
                    if clf:
                        log.debug("🔬 Classifier preloaded")
                
                log.info("🚀 Enhanced Pruning models ready")
        except Exception as e:
            log.error(f"Enhanced preload error: {e}")

    threading.Thread(target=_preload, daemon=True).start()