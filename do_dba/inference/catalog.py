"""Grouping, filtering, and fuzzy lookup over the models the service reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

# The same endpoint lists image, audio, video, embedding, and reranker models.
# None of them work through /v1/chat/completions, so --list-models hides them and
# naming one with -m is refused.
#
# There is no capability field in the response to key off: chat models usually
# carry context_length, but kimi-k3 and the router:* aliases do not, while
# qwen3-tts-voicedesign and wan2-2-t2v-a14b do. Matching on the id is what
# actually separates them.
_NON_CHAT_HINTS = (
    "image",
    "tts",
    "text-to-speech",
    "audio",
    "speech",
    "embed",
    "video",
    "t2v",
    "stable-diffusion",
    "voicedesign",
    "wan2",
    "fal-ai/",
    # Sentence-embedding and reranker families whose names never say "embed".
    "rerank",
    "mpnet",
    "mini-lm",
    "minilm",
    "sentence-transformers",
)

# Same idea, but anchored: bare "e5-"/"gte-" are too short to match safely
# anywhere in an id.
_NON_CHAT_PREFIXES = ("bge-", "e5-", "gte-")

# What a server that names the kind of model outright calls them. LM Studio's
# /api/v0/models does (see inference/details.py) and it beats every guess made
# from an id below: "vlm" is a chat model that can also see, and an embedding
# model named after its architecture rather than its job - nomic-embed is caught
# by the hints, muse-glimmer would not be - is only knowable this way.
_CHAT_TYPES = {"llm", "vlm"}
_NON_CHAT_TYPES = {"embeddings", "embedding", "image", "audio", "tts", "rerank", "reranker"}
# LM Studio's words for whether the weights are in memory.
_LOAD_STATES = {"loaded", "not-loaded", "loading"}

# owned_by values that name the host rather than whoever made the model, so
# grouping by them would put every model in one meaningless heading. LM Studio
# reports "organization_owner" for everything it serves; DigitalOcean reports
# itself. An empty string lands here too, which is where it belongs.
_ANONYMOUS_OWNERS = {"", "digitalocean", "organization_owner", "system", "local", "lm-studio"}

# Ordered longest-prefix-first so "ministral" is not swallowed by "mistral".
_PROVIDERS: tuple[tuple[str, str], ...] = (
    # router:* are DigitalOcean aliases that pick a model for you per request.
    ("router:", "DO Router"),
    ("anthropic", "Anthropic"),
    ("openai", "OpenAI"),
    ("meta-llama", "Meta"),
    ("llama", "Meta"),
    ("deepseek", "DeepSeek"),
    ("qwen", "Alibaba / Qwen"),
    ("wan2", "Alibaba / Qwen"),
    ("ministral", "Mistral AI"),
    ("mistral", "Mistral AI"),
    ("kimi", "Moonshot AI"),
    ("nvidia", "NVIDIA"),
    ("nemotron", "NVIDIA"),
    ("gemma", "Google"),
    ("gemini", "Google"),
    ("glm", "Z.ai"),
    ("mimo", "Xiaomi"),
    ("minimax", "MiniMax"),
    ("arcee", "Arcee"),
    ("stable", "Stability AI"),
    ("fal-ai", "fal"),
)


@dataclass(frozen=True)
class ModelInfo:
    id: str
    provider: str
    is_chat: bool
    context_window: int | None = None
    # Whether the weights are in memory right now, where the gateway says so -
    # None everywhere else, which is not the same as False. A hosted gateway keeps
    # its own models warm and never mentions it; a self-hosted one has one model
    # loaded at a time, and asking for a cold one means waiting for it to be read
    # off disk before the first token.
    loaded: bool | None = None

    @property
    def context_label(self) -> str:
        if not self.context_window:
            return ""
        if self.context_window >= 1_000_000:
            return f"{self.context_window / 1_000_000:.1f}M ctx".replace(".0M", "M")
        if self.context_window >= 1_000:
            return f"{self.context_window // 1000}K ctx"
        return f"{self.context_window} ctx"


class Catalog:
    """An ordered view of the available models, with lookup helpers."""

    def __init__(self, records: Iterable[dict[str, Any]]):
        models = [_to_info(record) for record in records]
        # Group by provider, alphabetical inside each group, providers in the
        # fixed order above so the listing looks the same on every run.
        provider_rank = {label: index for index, (_, label) in enumerate(_PROVIDERS)}
        self.all: list[ModelInfo] = sorted(
            models,
            key=lambda m: (provider_rank.get(m.provider, len(_PROVIDERS)), m.provider, m.id),
        )
        self.chat: list[ModelInfo] = [m for m in self.all if m.is_chat]
        self.other: list[ModelInfo] = [m for m in self.all if not m.is_chat]
        self._by_id = {m.id.lower(): m for m in self.all}

    def __len__(self) -> int:
        return len(self.all)

    def get(self, model_id: str) -> ModelInfo | None:
        return self._by_id.get(model_id.strip().lower())

    def ids(self) -> list[str]:
        return [m.id for m in self.all]

    def resolve(self, query: str) -> tuple[ModelInfo | None, list[ModelInfo]]:
        """Resolve a user-typed model reference.

        Accepts an exact id or a partial name. Returns (match, candidates); when
        match is None the candidates are the ambiguous options worth showing.
        """
        needle = query.strip()
        if not needle:
            return None, []

        exact = self.get(needle)
        if exact is not None:
            return exact, []

        lowered = needle.lower()
        substring = [m for m in self.all if lowered in m.id.lower()]
        if len(substring) == 1:
            return substring[0], []
        if substring:
            # "opus" matches many ids; prefer a chat model whose name starts
            # with the query before declaring it ambiguous.
            prefixed = [m for m in substring if m.id.lower().startswith(lowered)]
            if len(prefixed) == 1:
                return prefixed[0], []
            return None, substring

        # Last resort: every query fragment must appear somewhere in the id.
        fragments = [part for part in lowered.replace("-", " ").replace("/", " ").split() if part]
        loose = [m for m in self.all if all(part in m.id.lower() for part in fragments)]
        if len(loose) == 1:
            return loose[0], []
        return None, loose


def _to_info(record: dict[str, Any]) -> ModelInfo:
    model_id = str(record.get("id", "")).strip()
    lowered = model_id.lower()
    owned_by = str(record.get("owned_by") or "").strip()

    provider = ""
    for prefix, label in _PROVIDERS:
        if lowered.startswith(prefix) or f"/{prefix}" in lowered:
            provider = label
            break
    if not provider and "/" in model_id:
        # OpenRouter ids are vendor/model and it reports no owned_by, so the
        # vendor slug is the only grouping available - and it is a good one.
        provider = model_id.split("/", 1)[0].replace("-", " ").title()
    if not provider:
        provider = owned_by.title() if owned_by.lower() not in _ANONYMOUS_OWNERS else "Other"

    # Field names vary by gateway; take the first plausible one. The length the
    # weights were actually loaded with comes first where it is reported, because
    # it is the limit a request will hit: LM Studio will load a 256K model with a
    # 32K window if that is what it was told to do, and the maximum below it then
    # describes the file rather than the server.
    context_window = None
    for key in ("loaded_context_length", "context_window", "context_length",
                "max_context_length", "max_input_tokens"):
        value = record.get(key)
        if isinstance(value, int) and value > 0:
            context_window = value
            break

    # "loaded", "not-loaded", "loading" are LM Studio's three words for it. Any
    # other value is an answer to some different question, so it leaves this one
    # unanswered rather than reading as a no.
    state = str(record.get("state") or "").strip().lower()
    loaded = state == "loaded" if state in _LOAD_STATES else None

    is_chat = _is_chat(record, lowered)
    return ModelInfo(id=model_id, provider=provider, is_chat=is_chat,
                     context_window=context_window, loaded=loaded)


def _is_chat(record: dict[str, Any], lowered: str) -> bool:
    """Whether /v1/chat/completions will accept this model."""
    # A server that says what kind of model it is has settled the question.
    kind = str(record.get("type") or "").strip().lower()
    if kind in _CHAT_TYPES:
        return True
    if kind in _NON_CHAT_TYPES:
        return False

    # OpenRouter states its modalities outright, which beats inferring anything
    # from the id: a model that can emit text is usable here, whatever it is
    # named. A model that only emits images or audio is not.
    architecture = record.get("architecture")
    if isinstance(architecture, dict):
        outputs = architecture.get("output_modalities")
        if isinstance(outputs, list) and outputs:
            return any(str(kind).strip().lower() == "text" for kind in outputs)

    return not any(hint in lowered for hint in _NON_CHAT_HINTS) and not lowered.startswith(
        _NON_CHAT_PREFIXES
    )
