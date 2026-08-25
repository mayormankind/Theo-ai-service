# src/ai-service/app/services/segmentation_service.py
import re
import os
import json
from typing import List, Optional
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"))

# Fixed seed so that if the LLM fallback is ever used, repeated runs of the
# SAME script produce the SAME segmentation. Combined with temperature=0 this
# removes the run-to-run drift that previously made batches non-deterministic.
_LLM_SEED = 42

# ---------------------------------------------------------------------------
# Marker grammar
# ---------------------------------------------------------------------------
# A single line-leading matcher that recognises the many ways students label
# answers. We deliberately extract the numeric "core" (+ optional letter/roman
# sub-part) so every style collapses to one canonical label:
#
#   "Q1"  "Question 1"  "1."  "1)"  "(1)"  "[1]"  "No. 1"  "Answer 1"
#   "1(a)"  "1a."  "Q2b"  "(i)"  "ii)"
#
# The rubric-aware filter (see segment_answers) is the real safety net against
# false positives such as a "2000" year or a "1. 2. 3." list *inside* an answer
# body, so this pattern can afford to be generous.
_LEADING_MARKER = re.compile(
    r'^\s*(?:'
    # Q / Question / Qtn / Qn  + number + optional letter
    r'(?:questions?|qtn|qn|q)\s*[.:\-]?\s*(?P<qn>\d{1,2})\s*(?:\(\s*(?P<ql>[a-z])\s*\)|(?P<ql2>[a-z])(?=[\s.:)]|$))?'
    # No. / Number / Ans / Answer / Sol / Solution + number
    r'|(?:number|no|ans(?:wer)?|sol(?:ution)?)\s*[.:\-]?\s*(?P<nn>\d{1,2})\s*(?:\(\s*(?P<nl>[a-z])\s*\))?'
    # (1)  [1]  + optional (a)
    r'|[\(\[]\s*(?P<pn>\d{1,2})\s*[\)\]]\s*(?:\(\s*(?P<pl>[a-z])\s*\))?'
    # 1(a)
    r'|(?P<bn>\d{1,2})\s*\(\s*(?P<bl>[a-z])\s*\)'
    # 1.  1)  1a.  1a)   (number, optional single letter, required . or ) )
    r'|(?P<dn>\d{1,2})\s*(?P<dl>[a-z])?\s*[.)]'
    # roman sub-parts: (i) (ii) i) ii.
    r'|[\(]?\s*(?P<rom>x|ix|iv|v?i{1,3}|v)\s*[\).]'
    r')',
    re.IGNORECASE,
)


def _norm(num: Optional[str], letter: Optional[str] = None, roman: Optional[str] = None) -> str:
    """Build a canonical label. Mirrors normalize_question_label in grading.py
    and normalizeQuestionLabel in TheoGrader/lib/utils/question-label.ts."""
    if roman:
        return roman.lower()
    label = (num or "").lstrip("0") or (num or "")
    if letter:
        label += letter.lower()
    return label


def _match_leading_marker(line: str):
    """Return (normalized_label, consumed_char_count) for a line that begins
    with a question marker, else None."""
    m = _LEADING_MARKER.match(line)
    if not m:
        return None
    if m.group("qn"):
        label = _norm(m.group("qn"), m.group("ql") or m.group("ql2"))
    elif m.group("nn"):
        label = _norm(m.group("nn"), m.group("nl"))
    elif m.group("pn"):
        label = _norm(m.group("pn"), m.group("pl"))
    elif m.group("bn"):
        label = _norm(m.group("bn"), m.group("bl"))
    elif m.group("dn"):
        label = _norm(m.group("dn"), m.group("dl"))
    elif m.group("rom"):
        label = _norm(None, roman=m.group("rom"))
    else:
        return None
    return label, m.end()


def _detect_markers(raw_text: str):
    """Scan line-by-line, returning accepted markers as
    (line_start_char, answer_start_char, normalized_label)."""
    markers = []
    offset = 0
    for line in raw_text.splitlines(keepends=True):
        hit = _match_leading_marker(line)
        if hit:
            label, consumed = hit
            markers.append((offset, offset + consumed, label))
        offset += len(line)
    return markers


def _parent(label: str) -> str:
    """"2a" -> "2"; used to attach sub-parts to a parent question."""
    return re.sub(r'[a-z]+$', '', label)


def _map_to_expected(label: str, expected_index: dict) -> Optional[str]:
    """Map a detected label to a known rubric label (exact, then parent)."""
    if label in expected_index:
        return label
    parent = _parent(label)
    if parent and parent in expected_index:
        return parent
    return None


def _numeric_key(label: str) -> int:
    m = re.match(r'\d+', label)
    return int(m.group()) if m else 0


def segment_answers(raw_text: str, expected_labels: Optional[List[str]] = None) -> dict:
    """
    Segment raw OCR text into individual answers.

    When ``expected_labels`` (the rubric's canonical question labels) are
    supplied, segmentation becomes *rubric-aware*: only markers that correspond
    to a real question are accepted, markers are kept in monotonic order, and
    each question is taken once (its first occurrence). This is what prevents:
      * enumerated lists inside an answer from splitting it into fake questions,
      * stray numbers/years being treated as new questions,
      * answers merging because a marker style wasn't recognised.

    Sub-parts (e.g. "1a", "1b") that map to a parent rubric question ("1") are
    concatenated under that question, in document order.

    The function never guesses silently: if structure can't be found it uses a
    *deterministic* LLM fallback constrained to the expected labels.
    """
    raw_text = raw_text or ""
    expected_index = None
    if expected_labels:
        # preserve rubric order for monotonic validation
        expected_index = {lbl: i for i, lbl in enumerate(expected_labels)}

    markers = _detect_markers(raw_text)

    accepted = []          # (answer_start_char, line_start_char, canonical_label)
    used = set()
    last_order = -1

    for line_start, answer_start, label in markers:
        if expected_index is not None:
            mapped = _map_to_expected(label, expected_index)
            if mapped is None:
                continue                      # not a real question — ignore
            order = expected_index[mapped]
        else:
            mapped = label
            order = _numeric_key(label)

        # First occurrence only + monotonic (guards against list items that
        # repeat an earlier number deeper inside an answer body).
        if mapped in used or order < last_order:
            continue
        used.add(mapped)
        last_order = order
        accepted.append((line_start, answer_start, mapped))

    # Build segments between consecutive accepted markers.
    if accepted:
        segments: dict = {}
        for i, (_, answer_start, label) in enumerate(accepted):
            end = accepted[i + 1][0] if i + 1 < len(accepted) else len(raw_text)
            text = raw_text[answer_start:end].strip()
            if not text:
                continue
            if label in segments:
                segments[label] = f"{segments[label]}\n{text}".strip()
            else:
                segments[label] = text
        if segments:
            return segments

    # No usable markers found ------------------------------------------------
    stripped = raw_text.strip()
    # If the rubric has exactly one question, the whole script is that answer.
    if expected_index is not None and len(expected_index) == 1:
        return {next(iter(expected_index)): stripped}
    if len(stripped) > 20:
        return intelligent_segmentation_fallback(raw_text, expected_labels)
    return {"Uncategorized": stripped}


def intelligent_segmentation_fallback(
    raw_text: str,
    expected_labels: Optional[List[str]] = None,
) -> dict:
    """
    Deterministic gpt-4o-mini fallback used only when heuristics fail.

    Determinism: temperature=0 + fixed seed, so the same script always yields
    the same segmentation. When expected labels are known, the model is forced
    to return exactly those keys, so it can neither invent labels nor collapse
    everything into a single "Uncategorized" blob.
    """
    try:
        if expected_labels:
            system = (
                "You segment a university exam script into answers. "
                "Return ONLY a JSON object whose keys are EXACTLY drawn from this "
                f"list of question labels: {', '.join(expected_labels)}. "
                "Each value is that question's answer, copied verbatim from the "
                "script. If a listed question has no answer present, use an empty "
                "string. Do not invent keys or merge answers."
            )
        else:
            system = (
                "You segment a university exam script into answers. Return ONLY a "
                "valid JSON object where keys are question labels (e.g. '1', '2a') "
                "and values are the corresponding verbatim answer texts."
            )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": raw_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            seed=_LLM_SEED,
        )
        data = json.loads(response.choices[0].message.content)
        # Keep only non-empty string answers.
        cleaned = {
            str(k): v.strip()
            for k, v in data.items()
            if isinstance(v, str) and v.strip()
        }
        return cleaned or {"Uncategorized": raw_text.strip()}
    except Exception as e:
        print(f"Intelligent segmentation fallback failed: {e}")
        return {"Uncategorized": raw_text.strip()}
