"""
Provenance‑Style Pruning Plugin for **Cheshire Cat AI**
=======================================================

Removes sentence‑level noise from recalled documents before they
reach the LLM, minimising hallucinations.  
Optimised for CPU / CUDA / Apple Silicon MPS.

Changes vs. first draft
-----------------------
* **Real classifier support** – loads a Provenance (binary sentence‑
  relevance) checkpoint if provided; otherwise falls back to cosine
  similarity with an embedding model.
* Parameter renamed **keep_ratio** (fraction of sentences to KEEP).  
  Compression = `1‑keep_ratio`.
* Accurate token counting via *tiktoken* when available.
* Cache key uses SHA‑256 ⇒ stable across restarts.
* Pruning returns **new MemoryDocument copies** preserving original
  retrieval score.
* Diagnostics & status report clearer error causes.

Installation hint
-----------------
Add to your plugin folder and list dependencies in the plugin’s
`requirements.txt` (see bottom of file).  Provide a valid Hugging Face
model id in settings **classifier_model** or keep default (cosine sim).

"""
from __future__ import annotations

import hashlib
import json
import platform
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

    # tiktoken is optional – improves token estimation
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    except Exception:
        enc = None

    # Device detection
    if (
        platform.machine() in {"arm64", "aarch64"}
        and torch.backends.mps.is_available()
    ):
        DEVICE = torch.device("mps")
        log.info("🍎 Using Apple Silicon MPS backend")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        log.info("⚡ Using CUDA backend")
    else:
        DEVICE = torch.device("cpu")
        log.info("💻 Using CPU backend")
except Exception as e:  # broad – anything fails ⇒ disable ML path
    ML_AVAILABLE = False
    DEVICE = None
    log.warning(f"⚠️ ML stack unavailable: {e}")

# ---------------------------------------------------------------------------
# Settings schema – editable from Cat admin UI
# ---------------------------------------------------------------------------
class ProvenanceSettings(BaseModel):
    """Plugin settings editable from the Admin UI."""

    enable_pruning: bool = Field(
        True, description="Enable/disable sentence‑level pruning"
    )
    keep_ratio: float = Field(
        0.3,
        ge=0.05,
        le=0.95,
        description="Fraction of sentences to keep after scoring",
    )
    min_tokens_for_pruning: int = Field(
        800,
        ge=200,
        le=8000,
        description="Skip pruning if recalled context has fewer tokens.",
    )
    preserve_head_tail: int = Field(
        2,
        ge=0,
        le=10,
        description="Sentences always preserved from doc head & tail.",
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

    # internal pydantic config so Cat doesn’t treat Field as memory namespace
    model_config = {"protected_namespaces": ()}


@plugin
def settings_model():  # exposed to Cat
    return ProvenanceSettings

# ---------------------------------------------------------------------------
# Helper client – loads models & executes pruning
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

    # ~~~~~~~~~~~~~ model loaders ~~~~~~~~~~~~~
    def _load_embedder(self) -> SentenceTransformer:
        if self._embed_model is not None:
            return self._embed_model
        model = SentenceTransformer(
            self.settings.embed_model, device=str(DEVICE) if DEVICE else "cpu"
        )
        self._embed_model = model
        log.info(f"🧩 Embedding model loaded: {self.settings.embed_model}")
        return model

    def _load_classifier(self) -> Optional[Pipeline]:
        if not self.settings.classifier_model:
            return None  # user hasn’t set one
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
                device=0 if DEVICE and DEVICE.type != "cpu" else -1,
            )
            self._clf_pipeline = pipe
            log.info(f"🔬 Classifier loaded: {self.settings.classifier_model}")
            return pipe
        except Exception as e:
            log.warning(f"Classifier load failed → fallback ({e})")
            self._clf_pipeline = None
            return None

    # ~~~~~~~~~~~~~ util ~~~~~~~~~~~~~
    @staticmethod
    def _sentence_split(text: str) -> List[str]:
        try:
            return nltk.sent_tokenize(text)
        except Exception:  # fallback simple split
            return [s.strip() for s in text.split(".") if s.strip()]

    def _token_len(self, text: str) -> int:
        if enc:
            try:
                return len(enc.encode(text))
            except Exception:
                pass
        return int(len(text.split()) * 1.3)  # crude fallback

    # ~~~~~~~~~~~~~ scoring ~~~~~~~~~~~~~
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
            inputs = [(query, s) for s in sentences]
            preds = pipe(inputs, batch_size=min(32, len(inputs)))
            # pipeline returns list of dicts [{'label': 'LABEL_1', 'score': 0.87}, …]
            return [p["score"] for p in preds]
        except Exception as e:
            log.warning(f"Classifier inference failed → fallback ({e})")
            return None

    # ~~~~~~~~~~~~~ public API ~~~~~~~~~~~~~
    def prune(self, document: str, query: str) -> Tuple[str, Dict[str, Any]]:
        if not document.strip():
            return document, {"compression": 0, "sentences_kept": 0}

        # deterministic cache key
        if self.settings.cache_enabled:
            cache_key = hashlib.sha256(
                json.dumps([
                    document,
                    query,
                    self.settings.keep_ratio,
                    self.settings.classifier_model,
                    self.settings.embed_model,
                ]).encode()
            ).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]

        start = time.time()
        sentences = self._sentence_split(document)
        total = len(sentences)
        if total == 0:
            return document, {"compression": 0, "sentences_kept": 0}

        # scoring – classifier preferred, else cosine
        scores = self._classifier_scores(query, sentences)
        if scores is None:
            scores = self._cosine_scores(query, sentences)

        # positional bias (keep_ratio == 1 → skip sorting)
        bias_head_tail = self.settings.preserve_head_tail
        biased = []
        for idx, (sent, sc) in enumerate(zip(sentences, scores)):
            pos_bonus = 0.1 if idx < bias_head_tail or idx >= total - bias_head_tail else 0.0
            biased.append((idx, sent, sc + pos_bonus))

        # select top‑k by score then restore order
        keep = max(1, int(total * self.settings.keep_ratio))
        biased.sort(key=lambda x: x[2], reverse=True)
        selected = sorted(biased[:keep], key=lambda x: x[0])
        pruned_text = " ".join([s[1] for s in selected])

        meta = {
            "compression": 1 - (len(selected) / total),
            "sentences_kept": len(selected),
            "processing_time": round(time.time() - start, 3),
        }
        if self.settings.cache_enabled:
            self._cache[cache_key] = (pruned_text, meta)
        return pruned_text, meta


# ---------------------------------------------------------------------------
# Global helpers – instantiated lazily per Cat process
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
# Hooks – attach to Cat retrieval pipeline
# ---------------------------------------------------------------------------
@hook(priority=2)
def before_cat_recalls_declarative_memories(cfg, cat):
    # Increase k so pruning has broader pool
    try:
        settings = ProvenanceSettings(**cat.mad_hatter.get_plugin().load_settings())
        if settings.enable_pruning:
            cfg["k"] = min(cfg.get("k", 3) * 2, 10)
    except Exception:
        pass
    return cfg


@hook(priority=1)
def after_cat_recalls_declarative_memories(memories, cat):
    """Apply sentence‑level pruning and return **new** memories list."""
    if not memories:
        return memories
    client = _get_client(cat)
    if client is None:
        return memories

    settings = client.settings
    if not settings.enable_pruning:
        return memories

    # decide if we need pruning based on total tokens
    try:
        total_text = "\n".join([doc.page_content for doc, _ in memories])
        if client._token_len(total_text) < settings.min_tokens_for_pruning:
            return memories
    except Exception:
        pass  # if anything goes wrong, fail open

    query = cat.working_memory.user_message_json.text
    pruned_memories = []
    for doc, score in memories:
        new_doc = deepcopy(doc)
        pruned, meta = client.prune(doc.page_content, query)
        new_doc.page_content = pruned
        pruned_memories.append((new_doc, score))

    log.info(
        "✂️ Pruning done – compression {:.1%} in {:.3f}s".format(
            meta["compression"], meta["processing_time"]
        )
    )
    return pruned_memories

# ---------------------------------------------------------------------------
# Utility tools – callable by user / agent
# ---------------------------------------------------------------------------
@tool
def pruning_status(tool_input, cat):
    """Return human‑readable pruning system status."""
    if not ML_AVAILABLE:
        return "❌ ML stack not available – pruning disabled."
    client = _get_client(cat)
    if client is None:
        return "⚠️ Pruning client not initialised. Check logs."
    s = client.settings
    out = [
        "**🪄 Provenance Pruning Status**",
        f"- Enabled: {'✅' if s.enable_pruning else '❌'}",
        f"- Device: {DEVICE}",
        f"- Keep ratio: {s.keep_ratio:.2f}",
        f"- Min tokens: {s.min_tokens_for_pruning}",
        f"- Classifier: {s.classifier_model or '—'}",
        f"- Embed model: {s.embed_model}",
        f"- Cache entries: {len(client._cache)}",
    ]
    return "\n".join(out)


@tool
def pruning_diagnostics(tool_input, cat):
    """Run dependency and model sanity checks."""
    if not ML_AVAILABLE:
        return "❌ ML stack missing."
    client = _get_client(cat)
    if client is None:
        return "❌ Cannot init client – see logs."
    diag = ["**🔧 Diagnostics**"]
    # torch & device
    import torch as _t

    diag.append(f"PyTorch: {_t.__version__}")
    diag.append(f"Device: {DEVICE}")
    # model tests
    try:
        emb = client._load_embedder().encode(["test"])
        diag.append(f"Embed OK: {emb.shape}")
    except Exception as e:
        diag.append(f"Embed FAIL: {e}")
    if client.settings.classifier_model:
        try:
            clf = client._load_classifier()
            if clf:
                score = clf([("test", "test")])[0]["score"]
                diag.append(f"Classifier OK: score {score:.2f}")
            else:
                diag.append("Classifier not loaded (fallback mode)")
        except Exception as e:
            diag.append(f"Classifier FAIL: {e}")
    return "\n".join(diag)


@tool
def test_pruning(query: str, cat):
    """Quick self‑test on a canned DeepSeek paragraph."""
    if not ML_AVAILABLE:
        return "❌ ML stack missing."
    client = _get_client(cat)
    if client is None:
        return "❌ Client not available (see logs)."
    doc = (
        "The training cost of the DeepSeek‑V3 model was 5.576 million dollars. "
        "DeepSeek‑V3 uses a transformer architecture with efficient attention mechanism. "
        "The DeepSeek company is based in China and specializes in artificial intelligence. "
        "The model was trained on distributed GPU clusters. "
        "Results show competitive performance compared to other models. "
        "Metrics include BLEU score and accuracy on standard benchmarks."
    )
    pruned, meta = client.prune(doc, query)
    return (
        f"**🧪 Test**\nQuery: {query}\nCompression: {meta['compression']:.1%}\n"
        f"Kept: {meta['sentences_kept']} sentences\n\n{pruned}"
    )


# ---------------------------------------------------------------------------
# Startup hook – preload models asynchronously
# ---------------------------------------------------------------------------
@hook
def after_cat_bootstrap(cat):
    if not ML_AVAILABLE:
        log.warning("Pruning plugin: ML stack unavailable. Skipping preload.")
        return

    def _preload():
        client = _get_client(cat)
        if client:
            try:
                client._load_embedder()
                if client.settings.classifier_model:
                    client._load_classifier()
                log.info("🧊 Pruning models pre‑loaded")
            except Exception as e:
                log.error(f"Preload error: {e}")

    threading.Thread(target=_preload, daemon=True).start()