"""
Deterministic, offline "vibe text metrics" for LLM responses.

This module provides lightweight, reproducible heuristics over text-only responses to
approximate several vibe dimensions without any network/LLM calls.

## Naming conventions
- Metrics are returned as a JSON-serializable dict of ``metric_name -> value``.
- For response-text-only metrics, the module computes two variants when applicable:
  - ``*_all``: computed on the full response text
  - ``*_no_code``: computed after removing fenced code blocks (```...```)
- Prompt- or persona-dependent metrics are computed on the full response only and do
  not receive ``*_no_code`` variants unless explicitly stated in the spec.

## Extending
Add a new ``_compute_<dimension>(...)`` helper that returns a dict of base metric
names (without ``_all`` / ``_no_code`` suffixes). Then register it in
``compute_vibe_text_metrics`` and decide whether it should be computed for both text
variants.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# -----------------------------
# Core text helpers (tested)
# -----------------------------


_FENCE_BLOCK_RE = re.compile(r"```[\s\S]*?```", flags=re.MULTILINE)
_HEADER_RE = re.compile(r"^#{1,6}\s", flags=re.MULTILINE)
_NUMBERED_STEP_RE = re.compile(r"^\s*\d+\.", flags=re.MULTILINE)
_BULLET_LINE_RE = re.compile(r"^\s*(?:-|\*|•)\s", flags=re.MULTILINE)
_RUNNABLE_CMD_RE = re.compile(r"^\s*(pip|conda|python|pytest|git|srun|sbatch)\b")
_PREREQ_RE = re.compile(
    r"^\s*(Prereq|Prerequisite|Requirements|Install|Setup)\b", flags=re.IGNORECASE
)
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_MONTH_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
    r"January|February|March|April|June|July|August|September|October|November|December)\b",
    flags=re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def strip_code_fences(text: str) -> str:
    """
    Remove fenced code blocks from markdown-like text.

    Args:
        text (str): Input response text.

    Returns:
        str: Text with all ```...``` blocks removed.
    """
    if not text:
        return ""
    return _FENCE_BLOCK_RE.sub("", text)


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a light punctuation heuristic.

    The split is performed on [.?!] as specified, and empty segments are dropped.

    Args:
        text (str): Input text.

    Returns:
        List[str]: Sentence strings (without terminal punctuation).
    """
    if not text:
        return []
    parts = re.split(r"[.?!]", text)
    return [p.strip() for p in parts if p.strip()]


def tokenize_normalized(text: str) -> List[str]:
    """
    Tokenize to normalized word tokens (lowercase, punctuation-stripped).

    Args:
        text (str): Input text.

    Returns:
        List[str]: Normalized tokens.
    """
    if not text:
        return []
    # Normalize common curly apostrophes to keep contraction matching stable.
    text = text.replace("’", "'").replace("‘", "'")
    return re.findall(r"[a-z0-9]+", text.lower())


def tokenize_whitespace(text: str) -> List[str]:
    """
    Tokenize by whitespace.

    Args:
        text (str): Input text.

    Returns:
        List[str]: Whitespace-delimited tokens.
    """
    if not text:
        return []
    return text.split()


def extract_fenced_code_blocks(text: str) -> List[str]:
    """
    Extract raw fenced code blocks (inner contents only).

    Args:
        text (str): Response text.

    Returns:
        List[str]: List of code-block contents.
    """
    if not text:
        return []
    blocks: List[str] = []
    # Capture the inner content to count lines and identifiers.
    for match in re.finditer(r"```[a-zA-Z0-9_-]*\s*\n([\s\S]*?)```", text):
        blocks.append(match.group(1))
    return blocks


# -----------------------------
# Stopwords and small utilities
# -----------------------------


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "so",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "too",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
    }
)


def _safe_div(numer: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return float(numer / denom)


def _count_regex(pattern: re.Pattern[str], text: str) -> int:
    if not text:
        return 0
    return int(len(pattern.findall(text)))


def _count_phrases(text_lower: str, phrases: Sequence[str]) -> int:
    if not text_lower:
        return 0
    total = 0
    for phrase in phrases:
        p = phrase.lower()
        if " " in p:
            total += text_lower.count(p)
        else:
            total += len(re.findall(rf"\b{re.escape(p)}\b", text_lower))
    return int(total)


def _count_question_sentences(text: str) -> List[str]:
    if not text:
        return []
    # Count segments ending with '?' as "question sentences".
    segments = re.findall(r"[^.?!]*\?", text)
    return [s.strip() for s in segments if s.strip() and any(ch.isalpha() for ch in s)]


def _iter_noncode_lines(text: str) -> Iterable[str]:
    """
    Iterate lines that are not inside fenced code blocks.

    This uses a simple fence toggling heuristic based on lines containing ``` markers.
    """
    if not text:
        return []
    in_fence = False
    for line in text.splitlines():
        if "```" in line:
            # Toggle state; treat the fence line itself as non-content.
            in_fence = not in_fence
            continue
        if not in_fence:
            yield line


def _split_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


def _top_prompt_keywords(prompt_text: str, top_k: int = 20) -> List[str]:
    tokens = tokenize_normalized(prompt_text)
    filtered = [t for t in tokens if len(t) >= 5 and t not in _STOPWORDS]
    counts = Counter(filtered)
    # Deterministic ordering: by count desc then token asc.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [tok for tok, _ in ranked[:top_k]]


@dataclass(frozen=True)
class _TextViews:
    all_text: str
    no_code_text: str


# -----------------------------
# Dimension implementations
# -----------------------------


def _compute_clarity(text: str) -> Dict[str, Any]:
    lines = text.splitlines()
    total_lines = len(lines)
    bullet_lines = _count_regex(_BULLET_LINE_RE, text)
    sentence_tokens = [len(tokenize_whitespace(s)) for s in split_sentences(text)]
    avg_tokens_per_sentence = (
        sum(sentence_tokens) / float(len(sentence_tokens)) if sentence_tokens else 0.0
    )
    explicit_refs = ["Step ", "In the code", "Above", "Below", "Next"]
    num_explicit_refs = sum(text.count(p) for p in explicit_refs)

    code_blocks = extract_fenced_code_blocks(text)
    num_code_fences = int(text.count("```") // 2)

    has_single_primary_solution_block = False
    if len(code_blocks) == 1:
        nonempty_lines = [ln for ln in code_blocks[0].splitlines() if ln.strip()]
        if len(nonempty_lines) >= 8:
            has_single_primary_solution_block = True

    return {
        "num_headers": int(len(_HEADER_RE.findall(text))),
        "num_numbered_steps": int(len(_NUMBERED_STEP_RE.findall(text))),
        "bullet_lines_ratio": _safe_div(float(bullet_lines), float(max(1, total_lines))),
        "avg_tokens_per_sentence": float(avg_tokens_per_sentence),
        "num_explicit_refs": int(num_explicit_refs),
        "num_code_fences": int(num_code_fences),
        "has_single_primary_solution_block": bool(has_single_primary_solution_block),
    }


def _compute_tone_style_fit(text: str) -> Dict[str, Any]:
    tokens_norm = tokenize_normalized(text)
    tokens_ws = tokenize_whitespace(text)
    total_tokens = len(tokens_norm)
    unique_tokens = len(set(tokens_norm))

    sentences = split_sentences(text)
    imperatives = ("Do ", "Run ", "Use ", "Set ", "Add ", "Install ")
    imperative_count = 0
    for s in sentences:
        s_strip = s.lstrip()
        for prefix in imperatives:
            if s_strip.startswith(prefix):
                imperative_count += 1
                break

    text_lower = text.lower().replace("’", "'").replace("‘", "'")
    hedges = ["maybe", "might", "could", "probably", "i think", "it depends"]
    encouragement = ["you can", "don't worry", "nice", "great"]
    contractions = ["i'm", "don't", "can't", "you're", "we'll"]
    politeness = ["please", "thanks", "sorry"]

    word_count = len(tokens_ws)
    hedge_count = _count_phrases(text_lower, hedges)
    contraction_count = _count_phrases(text_lower, contractions)
    politeness_count = _count_phrases(text_lower, politeness)
    encouragement_count = _count_phrases(text_lower, encouragement)

    per_100w = _safe_div(100.0, float(max(1, word_count)))

    return {
        "word_count": int(word_count),
        "compression_ratio": _safe_div(float(unique_tokens), float(max(1, total_tokens))),
        "imperative_rate": _safe_div(float(imperative_count), float(max(1, len(sentences)))),
        "hedge_rate_per_100w": float(hedge_count * per_100w),
        "encouragement_count": int(encouragement_count),
        "contraction_rate_per_100w": float(contraction_count * per_100w),
        "politeness_rate_per_100w": float(politeness_count * per_100w),
    }


def _compute_workflow_fit(text: str) -> Dict[str, Any]:
    runnable = 0
    for line in text.splitlines():
        if _RUNNABLE_CMD_RE.match(line):
            runnable += 1

    code_blocks = extract_fenced_code_blocks(text)
    copy_paste_blocks = 0
    for block in code_blocks:
        nonempty = [ln for ln in block.splitlines() if ln.strip()]
        if len(nonempty) >= 5:
            copy_paste_blocks += 1

    markers = ["sanity check", "quick check", "assert", "verify", "print("]
    text_lower = text.lower()
    quick_checks = sum(text_lower.count(m) for m in markers)

    question_sents = _count_question_sentences(text)
    blocking_markers = ["need", "must", "cannot", "can't", "required"]
    blocking_questions = 0
    for q in question_sents:
        ql = q.lower().replace("’", "'").replace("‘", "'")
        if any(m in ql for m in blocking_markers):
            blocking_questions += 1

    prereq_steps = int(len(_PREREQ_RE.findall(text)))

    return {
        "num_runnable_commands": int(runnable),
        "num_copy_paste_code_blocks": int(copy_paste_blocks),
        "num_quick_checks": int(quick_checks),
        "num_questions_to_user": int(len(question_sents)),
        "blocking_questions": int(blocking_questions),
        "num_prereq_steps": int(prereq_steps),
    }


def _compute_cognitive_load(text_all: str, text_no_code: str) -> Dict[str, Any]:
    # avg_content_tokens_per_line: mean tokens per nonempty, noncode line
    noncode_lines = [ln for ln in _iter_noncode_lines(text_all) if ln.strip()]
    token_counts = [len(tokenize_whitespace(ln)) for ln in noncode_lines]
    avg_content_tokens_per_line = (
        sum(token_counts) / float(len(token_counts)) if token_counts else 0.0
    )

    # paragraph_mean_length and max_options_in_section computed on *_no_code
    paragraphs = _split_paragraphs(text_no_code)
    para_lengths = [len(tokenize_whitespace(p)) for p in paragraphs]
    paragraph_mean_length = (
        sum(para_lengths) / float(len(para_lengths)) if para_lengths else 0.0
    )

    option_markers = ["Option", "Alternatively", "Either", "Or:", "You can also"]
    num_options_markers = sum(text_all.count(m) for m in option_markers)

    max_options_in_section = 0
    for p in paragraphs:
        count = len(re.findall(r"\b(?:Option|Alternatively)\b", p, flags=re.IGNORECASE))
        if count > max_options_in_section:
            max_options_in_section = count

    num_glossary_defs = 0
    num_glossary_defs += len(re.findall(r"\b\w+\s+is\b", text_all))
    num_glossary_defs += len(re.findall(r"\bmeans\b", text_all, flags=re.IGNORECASE))
    num_glossary_defs += len(re.findall(r"\bi\.e\.\b", text_all, flags=re.IGNORECASE))
    num_glossary_defs += len(re.findall(r"\baka\b", text_all, flags=re.IGNORECASE))

    tokens_nc = tokenize_normalized(text_no_code)
    approx_new_terms = {
        t
        for t in tokens_nc
        if len(t) >= 6 and t not in _STOPWORDS and any(ch.isalpha() for ch in t)
    }

    numbered_steps = int(len(_NUMBERED_STEP_RE.findall(text_all)))
    bullet_steps = int(len(_BULLET_LINE_RE.findall(text_all)))
    noncode_tokens_count = len(tokenize_whitespace(text_no_code))
    avg_stepsize = _safe_div(
        float(noncode_tokens_count), float(max(1, numbered_steps + bullet_steps))
    )

    return {
        "avg_content_tokens_per_line": float(avg_content_tokens_per_line),
        "paragraph_mean_length": float(paragraph_mean_length),
        "num_options_markers": int(num_options_markers),
        "max_options_in_section": int(max_options_in_section),
        "num_glossary_defs": int(num_glossary_defs),
        "approx_new_technical_terms": int(len(approx_new_terms)),
        "avg_stepsize": float(avg_stepsize),
    }


def _extract_constraint_lines(prompt_text: str) -> List[str]:
    keys = ["must", "do not", "don't", "avoid", "required", "format", "constraint"]
    lines = []
    for line in prompt_text.splitlines():
        lower = line.lower().replace("’", "'").replace("‘", "'")
        if any(k in lower for k in keys):
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def _constraint_key_tokens(line: str) -> List[str]:
    toks = tokenize_normalized(line)
    return [t for t in toks if len(t) >= 4 and t not in _STOPWORDS]


def _extract_prompt_entities(prompt_text: str) -> List[str]:
    entities: List[str] = []

    # Backtick spans.
    entities.extend([m.group(1).strip() for m in re.finditer(r"`([^`\n]+)`", prompt_text)])

    # Identifiers near "function" or "class".
    for m in re.finditer(
        r"\b(?:function|class)\s+([A-Za-z_][A-Za-z0-9_]*)", prompt_text, flags=re.IGNORECASE
    ):
        entities.append(m.group(1))

    # Identifiers inside code fences.
    for block in extract_fenced_code_blocks(prompt_text):
        entities.extend(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", block))

    # Deduplicate while preserving order.
    seen = set()
    out: List[str] = []
    for e in entities:
        if not e:
            continue
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def _normalize_identifier(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def _compute_context_awareness(response_text: str, prompt_text: str) -> Dict[str, Any]:
    if prompt_text is None:
        raise ValueError("prompt_text is required for context-awareness metrics.")

    response_tokens = set(tokenize_normalized(response_text))

    constraint_lines = _extract_constraint_lines(prompt_text)
    total_constraints = len(constraint_lines)
    mentioned = 0
    for line in constraint_lines:
        keys = _constraint_key_tokens(line)
        overlap = sum(1 for t in set(keys) if t in response_tokens)
        if overlap >= 2:
            mentioned += 1

    entities = _extract_prompt_entities(prompt_text)
    reused = 0
    response_lower = response_text.lower()
    for ent in entities:
        if not ent:
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", ent):
            if re.search(rf"\b{re.escape(ent)}\b", response_text):
                reused += 1
        else:
            if ent.lower() in response_lower:
                reused += 1

    # Incorrect entity mutations (lightweight; no edit distance).
    response_idents = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", response_text)
    response_idents_set = set(response_idents)
    response_norm_map: Dict[str, List[str]] = {}
    for tok in response_idents_set:
        response_norm_map.setdefault(_normalize_identifier(tok), []).append(tok)

    incorrect_mutations = 0
    for ent in entities:
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", ent):
            continue
        if ent in response_idents_set:
            continue
        ent_norm = _normalize_identifier(ent)
        candidates = response_norm_map.get(ent_norm) or []
        # Count a mutation if we find a normalized match but not the exact identifier.
        if candidates:
            incorrect_mutations += 1
        else:
            # Case-insensitive mismatch without underscore/camel normalization.
            if ent.lower() in (t.lower() for t in response_idents_set):
                incorrect_mutations += 1

    deictics = ["today", "tomorrow", "yesterday", "above", "below", "next week"]
    deictic_terms_count = _count_phrases(
        response_text.lower().replace("’", "'").replace("‘", "'"), deictics
    )

    # Deictic resolutions: within 10 tokens of an anchor.
    tokens_ws = tokenize_whitespace(response_text)
    tokens_norm = [re.sub(r"[^\w-]+", "", t) for t in tokens_ws]
    anchors: set[int] = set()
    for i, tok in enumerate(tokens_ws):
        if _MONTH_RE.search(tok) or _ISO_DATE_RE.search(tok):
            anchors.add(i)
    for i in range(len(tokens_norm) - 1):
        if (tokens_norm[i].lower(), tokens_norm[i + 1].lower()) == ("previous", "message"):
            anchors.add(i)
            anchors.add(i + 1)

    deictic_resolutions_count = 0
    # Count occurrences by token index; include "next week" as bigram.
    for i, tok in enumerate(tokens_norm):
        t = tok.lower()
        is_deictic = t in {"today", "tomorrow", "yesterday", "above", "below"}
        if is_deictic:
            window = range(max(0, i - 10), min(len(tokens_norm), i + 11))
            if any(j in anchors for j in window):
                deictic_resolutions_count += 1
    for i in range(len(tokens_norm) - 1):
        if (tokens_norm[i].lower(), tokens_norm[i + 1].lower()) == ("next", "week"):
            window = range(max(0, i - 10), min(len(tokens_norm), i + 12))
            if any(j in anchors for j in window):
                deictic_resolutions_count += 1

    return {
        "constraint_mention_rate": _safe_div(float(mentioned), float(max(1, total_constraints))),
        "num_reused_entities": int(reused),
        "num_incorrect_entity_mutations": int(incorrect_mutations),
        "deictic_terms_count": int(deictic_terms_count),
        "deictic_resolutions_count": int(deictic_resolutions_count),
    }


def _persona_lexicons() -> Dict[str, List[str]]:
    return {
        "novice": ["step-by-step", "example", "explain", "remember", "pitfall"],
        "intermediate": ["pitfall", "common", "debug", "check", "explain"],
        "researcher": ["reproducible", "traceable", "ablation", "seed", "benchmark"],
        "advanced": ["assumption", "edge case", "complexity", "perf", "benchmark"],
    }


def _persona_target_key(user_id: str) -> str:
    if user_id.startswith("python_novice"):
        return "novice"
    if user_id.startswith("intermediate_learner"):
        return "intermediate"
    if user_id.startswith("ai_researcher"):
        return "researcher"
    if user_id.startswith("advanced_developer"):
        return "advanced"
    raise ValueError(
        f"Unsupported persona user_id prefix for persona metrics: {user_id!r}. "
        "Expected one of: python_novice*, intermediate_learner*, ai_researcher*, advanced_developer*."
    )


def _rate_per_100_words(count: int, word_count: int) -> float:
    return float(count) * _safe_div(100.0, float(max(1, word_count)))


def _compute_register_drift_score(response_text: str) -> float:
    # Split into sections by headers; if none, drift is 0.
    lines = response_text.splitlines()
    header_idxs = [i for i, ln in enumerate(lines) if re.match(r"^#{1,6}\s", ln)]
    if not header_idxs:
        return 0.0
    header_idxs.append(len(lines))
    sections: List[str] = []
    for start, end in zip(header_idxs[:-1], header_idxs[1:]):
        sections.append("\n".join(lines[start:end]).strip())

    section_metrics: List[Tuple[float, float, float]] = []
    for sec in sections:
        ws_tokens = tokenize_whitespace(sec)
        sentences = split_sentences(sec)
        avg_sentence_length = _safe_div(float(len(ws_tokens)), float(max(1, len(sentences))))
        text_lower = sec.lower().replace("’", "'").replace("‘", "'")
        hedge_count = _count_phrases(
            text_lower, ["maybe", "might", "could", "probably", "i think", "it depends"]
        )
        contraction_count = _count_phrases(text_lower, ["i'm", "don't", "can't", "you're", "we'll"])
        hedge_rate = _rate_per_100_words(hedge_count, len(ws_tokens))
        contraction_rate = _rate_per_100_words(contraction_count, len(ws_tokens))
        section_metrics.append((avg_sentence_length, hedge_rate, contraction_rate))

    if not section_metrics:
        return 0.0
    drift = 0.0
    for idx in range(3):
        vals = [m[idx] for m in section_metrics]
        drift += max(vals) - min(vals)
    return float(drift)


def _compute_persona_consistency(response_text: str, persona: Dict[str, Any]) -> Dict[str, Any]:
    if not persona or "user_id" not in persona:
        raise ValueError("persona with a 'user_id' field is required for persona-consistency metrics.")
    user_id = str(persona["user_id"])
    target = _persona_target_key(user_id)
    lexicons = _persona_lexicons()

    text_lower = response_text.lower().replace("’", "'").replace("‘", "'")
    word_count = len(tokenize_whitespace(response_text))

    rates: Dict[str, float] = {}
    for key, words in lexicons.items():
        count = _count_phrases(text_lower, words)
        rates[key] = _rate_per_100_words(count, word_count)

    target_rate = rates[target]
    cross_rate = max(v for k, v in rates.items() if k != target) if len(rates) > 1 else 0.0

    return {
        "persona_lexicon_rate_per_100w": float(target_rate),
        "cross_persona_contamination_per_100w": float(cross_rate),
        "register_drift_score": float(_compute_register_drift_score(response_text)),
    }


def _compute_friction_loss_of_control(
    response_text: str, prompt_text: Optional[str]
) -> Dict[str, Any]:
    lower = response_text.lower().replace("’", "'").replace("‘", "'")
    obstruction = ["i can't", "i cannot", "i won't", "unable to", "not possible", "will not"]
    obstruction_count = _count_phrases(lower, obstruction)

    # Unrequested clarifications: only when prompt_text is provided.
    unrequested_clarifications = 0
    if prompt_text is not None:
        questions = _count_question_sentences(response_text)
        allow = {"input", "data", "code", "version", "environment", "example"}
        for q in questions:
            ql = q.lower()
            if not any(tok in ql for tok in allow):
                unrequested_clarifications += 1

    # Offtask ratio requires prompt_text.
    offtask_ratio = 0.0
    if prompt_text is not None:
        keywords = set(_top_prompt_keywords(prompt_text, top_k=20))
        sentences = split_sentences(response_text)
        if sentences:
            off = 0
            for s in sentences:
                hits = sum(1 for t in set(tokenize_normalized(s)) if t in keywords)
                if hits < 2:
                    off += 1
            offtask_ratio = _safe_div(float(off), float(len(sentences)))

    # False completion claims: only count when there are zero code blocks.
    false_completion_claims_count = 0
    if response_text.count("```") == 0:
        false_claims = ["done", "i created", "as you can see", "implemented"]
        false_completion_claims_count = _count_phrases(lower, false_claims)

    # Multi-question clusters: paragraphs with >=2 '?'.
    multi_question_clusters = 0
    for p in _split_paragraphs(response_text):
        if p.count("?") >= 2:
            multi_question_clusters += 1

    return {
        "obstruction_phrases_count": int(obstruction_count),
        "unrequested_clarifications_count": int(unrequested_clarifications),
        "offtask_ratio": float(offtask_ratio),
        "false_completion_claims_count": int(false_completion_claims_count),
        "multi_question_clusters": int(multi_question_clusters),
    }


def _compute_reliability_user_trust(response_text: str) -> Dict[str, Any]:
    lower = response_text.lower().replace("’", "'").replace("‘", "'")
    word_count = len(tokenize_whitespace(response_text))

    uncertainty_markers = ["i'm not sure", "may", "depends", "likely", "approx"]
    overconfidence_markers = ["definitely", "guaranteed", "always", "never"]
    uncertainty = _count_phrases(lower, uncertainty_markers)
    overconfidence = _count_phrases(lower, overconfidence_markers)

    per_100w = _safe_div(100.0, float(max(1, word_count)))
    uncertainty_per_100w = float(uncertainty * per_100w)
    overconfidence_per_100w = float(overconfidence * per_100w)

    num_asserts = len(re.findall(r"\bassert\b", response_text))
    edge_cases = ["edge case", "n <= 0", "empty", "none", "overflow", "o("]
    num_edge_case_mentions = _count_phrases(lower, edge_cases)

    has_headers = bool(_HEADER_RE.search(response_text))
    has_code_fences = response_text.count("```") > 0
    unresolved_refs_count = 0
    if not has_headers and not has_code_fences:
        unresolved_refs_count = _count_phrases(lower, ["above", "below"])

    numeric_specificity_rate = float(len(_NUMERIC_TOKEN_RE.findall(response_text)) * per_100w)
    vagueness_marker_rate = float(_count_phrases(lower, ["some", "stuff", "things", "etc"]) * per_100w)

    return {
        "uncertainty_markers_per_100w": float(uncertainty_per_100w),
        "overconfidence_markers_per_100w": float(overconfidence_per_100w),
        "calibration_balance": float(_safe_div(float(uncertainty), float(overconfidence + 1))),
        "num_asserts": int(num_asserts),
        "num_edge_case_mentions": int(num_edge_case_mentions),
        "unresolved_refs_count": int(unresolved_refs_count),
        "numeric_specificity_rate_per_100w": float(numeric_specificity_rate),
        "vagueness_marker_rate_per_100w": float(vagueness_marker_rate),
    }


def _compute_anthropomorphism(response_text: str) -> Dict[str, Any]:
    lower = response_text.lower().replace("’", "'").replace("‘", "'")
    word_count = len(tokenize_whitespace(response_text))
    per_100w = _safe_div(100.0, float(max(1, word_count)))

    boilerplate = ["production-ready", "structured for clarity", "ready to be integrated", "ci and test suite"]
    boilerplate_rate = float(_count_phrases(lower, boilerplate) * per_100w)

    tokens = tokenize_normalized(response_text)
    total_4grams = max(0, len(tokens) - 3)
    counts: Counter[Tuple[str, str, str, str]] = Counter()
    for i in range(total_4grams):
        counts[(tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3])] += 1
    repeated_4grams = sum(1 for v in counts.values() if v > 1)
    ngram_repetition_ratio = _safe_div(float(repeated_4grams), float(max(1, total_4grams)))

    discourse = ["so", "now", "ok", "let's", "note that", "by the way"]
    discourse_rate = float(_count_phrases(lower, discourse) * per_100w)

    legalistic = ["hereby", "shall", "pursuant"]
    legalistic_rate = float(_count_phrases(lower, legalistic) * per_100w)

    headers = [ln.strip() for ln in response_text.splitlines() if re.match(r"^#{1,6}\s", ln)]
    header_texts = [re.sub(r"^#{1,6}\s*", "", h).strip().lower() for h in headers if h.strip()]
    total_headers = len(header_texts)
    duplicates = total_headers - len(set(header_texts)) if total_headers else 0
    sectional_symmetry_score = _safe_div(float(duplicates), float(max(1, total_headers)))

    return {
        "boilerplate_rate_per_100w": float(boilerplate_rate),
        "ngram_repetition_4gram_ratio": float(ngram_repetition_ratio),
        "discourse_marker_rate_per_100w": float(discourse_rate),
        "legalistic_rate_per_100w": float(legalistic_rate),
        "sectional_symmetry_score": float(sectional_symmetry_score),
    }


# -----------------------------
# Public entry point
# -----------------------------


def compute_vibe_text_metrics(
    response_text: str,
    prompt_text: str | None = None,
    persona: dict | None = None,
) -> Dict[str, Any]:
    """
    Compute deterministic vibe text metrics for a response.

    Args:
        response_text (str): LLM response text to analyze.
        prompt_text (str | None): Optional prompt text for prompt-dependent metrics.
        persona (dict | None): Optional persona dict for persona-dependent metrics.

    Returns:
        dict: JSON-serializable mapping of metric_name -> value.
    """
    if response_text is None:
        raise ValueError("response_text must be a string (got None).")
    if not isinstance(response_text, str):
        raise TypeError(f"response_text must be str, got {type(response_text).__name__}.")

    views = _TextViews(all_text=response_text, no_code_text=strip_code_fences(response_text))

    metrics: Dict[str, Any] = {}

    # A) Clarity (both variants)
    for suffix, txt in (("all", views.all_text), ("no_code", views.no_code_text)):
        for k, v in _compute_clarity(txt).items():
            metrics[f"{k}_{suffix}"] = v

    # B) Tone/style fit (both variants; features only)
    for suffix, txt in (("all", views.all_text), ("no_code", views.no_code_text)):
        for k, v in _compute_tone_style_fit(txt).items():
            metrics[f"{k}_{suffix}"] = v

    # C) Workflow fit (both variants)
    for suffix, txt in (("all", views.all_text), ("no_code", views.no_code_text)):
        for k, v in _compute_workflow_fit(txt).items():
            metrics[f"{k}_{suffix}"] = v

    # D) Cognitive load (mix: some are no_code; return both variants for all keys)
    cognitive_all = _compute_cognitive_load(views.all_text, views.no_code_text)
    cognitive_no_code = _compute_cognitive_load(views.no_code_text, views.no_code_text)
    for k, v in cognitive_all.items():
        metrics[f"{k}_all"] = v
    for k, v in cognitive_no_code.items():
        metrics[f"{k}_no_code"] = v

    # E) Context awareness (prompt-dependent; full response only)
    if prompt_text is not None:
        for k, v in _compute_context_awareness(views.all_text, prompt_text).items():
            metrics[k] = v

    # F) Persona consistency (persona-dependent; full response only)
    if persona is not None:
        for k, v in _compute_persona_consistency(views.all_text, persona).items():
            metrics[k] = v

    # G) Friction/loss of control
    # - prompt-dependent submetrics computed only when prompt_text is provided (full response)
    # - other submetrics computed for both variants (as specified)
    for suffix, txt in (("all", views.all_text), ("no_code", views.no_code_text)):
        frag = _compute_friction_loss_of_control(txt, prompt_text if suffix == "all" else None)
        for k, v in frag.items():
            if k in {"unrequested_clarifications_count", "offtask_ratio"}:
                if suffix == "all":
                    metrics[k] = v
                continue
            metrics[f"{k}_{suffix}"] = v

    # H) Reliability/user trust (both variants for response-only metrics)
    for suffix, txt in (("all", views.all_text), ("no_code", views.no_code_text)):
        for k, v in _compute_reliability_user_trust(txt).items():
            metrics[f"{k}_{suffix}"] = v

    # I) Anthropomorphism (both variants)
    for suffix, txt in (("all", views.all_text), ("no_code", views.no_code_text)):
        for k, v in _compute_anthropomorphism(txt).items():
            metrics[f"{k}_{suffix}"] = v

    return metrics


def group_vibe_text_metrics_by_dimension(
    metrics: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Group flat vibe text metrics into per-dimension buckets.

    Args:
        metrics (Dict[str, Any]): Flat metric mapping produced by
            :func:`compute_vibe_text_metrics`.

    Returns:
        Dict[str, Dict[str, Any]]: Mapping of vibe dimension name to a sub-dict of
        metric_name -> value. Any unrecognized metric keys are placed under
        the ``"other"`` bucket.
    """
    if metrics is None:
        raise ValueError("metrics must be a dict (got None).")
    if not isinstance(metrics, dict):
        raise TypeError(f"metrics must be dict, got {type(metrics).__name__}.")

    # Base (unsuffixed) names; suffix variants are matched via *_all / *_no_code.
    response_metric_bases: Dict[str, set[str]] = {
        "clarity": {
            "num_headers",
            "num_numbered_steps",
            "bullet_lines_ratio",
            "avg_tokens_per_sentence",
            "num_explicit_refs",
            "num_code_fences",
            "has_single_primary_solution_block",
        },
        "tone_style_fit": {
            "word_count",
            "compression_ratio",
            "imperative_rate",
            "hedge_rate_per_100w",
            "encouragement_count",
            "contraction_rate_per_100w",
            "politeness_rate_per_100w",
        },
        "workflow_fit": {
            "num_runnable_commands",
            "num_copy_paste_code_blocks",
            "num_quick_checks",
            "num_questions_to_user",
            "blocking_questions",
            "num_prereq_steps",
        },
        "cognitive_load": {
            "avg_content_tokens_per_line",
            "paragraph_mean_length",
            "num_options_markers",
            "max_options_in_section",
            "num_glossary_defs",
            "approx_new_technical_terms",
            "avg_stepsize",
        },
        "friction_loss_of_control": {
            "obstruction_phrases_count",
            "false_completion_claims_count",
            "multi_question_clusters",
        },
        "reliability_user_trust": {
            "uncertainty_markers_per_100w",
            "overconfidence_markers_per_100w",
            "calibration_balance",
            "num_asserts",
            "num_edge_case_mentions",
            "unresolved_refs_count",
            "numeric_specificity_rate_per_100w",
            "vagueness_marker_rate_per_100w",
        },
        "anthropomorphism": {
            "boilerplate_rate_per_100w",
            "ngram_repetition_4gram_ratio",
            "discourse_marker_rate_per_100w",
            "legalistic_rate_per_100w",
            "sectional_symmetry_score",
        },
    }

    # Prompt/persona dependent metrics are stored without *_all / *_no_code suffixes.
    unsuffixed: Dict[str, set[str]] = {
        "context_awareness": {
            "constraint_mention_rate",
            "num_reused_entities",
            "num_incorrect_entity_mutations",
            "deictic_terms_count",
            "deictic_resolutions_count",
        },
        "persona_consistency": {
            "persona_lexicon_rate_per_100w",
            "cross_persona_contamination_per_100w",
            "register_drift_score",
        },
        "friction_loss_of_control": {
            "unrequested_clarifications_count",
            "offtask_ratio",
        },
    }

    grouped: Dict[str, Dict[str, Any]] = {k: {} for k in set(response_metric_bases) | set(unsuffixed)}
    grouped["other"] = {}

    for key, value in metrics.items():
        if not isinstance(key, str):
            grouped["other"][str(key)] = value
            continue

        placed = False

        # Unsuffixed keys (prompt/persona dependent).
        for dim, keys in unsuffixed.items():
            if key in keys:
                grouped[dim][key] = value
                placed = True
                break
        if placed:
            continue

        # Suffix-based keys: <base>_<all|no_code>
        for dim, bases in response_metric_bases.items():
            for base in bases:
                if key == f"{base}_all" or key == f"{base}_no_code":
                    grouped[dim][key] = value
                    placed = True
                    break
            if placed:
                break

        if not placed:
            grouped["other"][key] = value

    return grouped

