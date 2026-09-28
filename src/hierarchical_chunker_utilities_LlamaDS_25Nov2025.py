#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
Token adaptor for Llama4 and DeepSeek V3 models.
"""

from together import Together
import json
import re
from typing import List, Dict, Tuple, Any, Optional, Sequence, Union
from transformers import AutoTokenizer  # Hugging-Face fast-tokenizer
import tiktoken
from ast import literal_eval      # safer than json.loads for a tiny int list
import unicodedata
import concurrent.futures as cf
import threading
from tqdm import tqdm

class TokenAdapter:
    """
    Wrap any tokenizer so the rest of the chunker can call
        .encode(text)                 → List[int]
        .decode(List[int])            → str
        .convert_ids_to_tokens(ids)   → List[str] | str
        .offsets(text)                → List[Tuple[int,int]]
    regardless of the backend.
    """
    def __init__(self, model_name: str):
        if "llama" in model_name:
            self.kind = "Llama"
            self.tok  = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.1-70B-Instruct",
                # "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                add_prefix_space=False,
                use_fast=True          # guarantees offset_mapping
            )
            self._need_slow_offsets = False

        if "deepseek" in model_name:
            self.kind = "DeepSeek"
            self.tok = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-V3",
            use_fast=True,          # <- IMPORTANT
            # Do NOT pass add_prefix_space here
            trust_remote_code=True  # DeepSeek provides its own subclass
    )

    # ---------- unified API ------------------------------------------------
    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def convert_ids_to_tokens(self, ids: Union[int, Sequence[int]]):
        return self.tok.convert_ids_to_tokens(ids)

    # --- NEW: get byte-offset mapping for *any* backend --------------------
    def offsets(self, text: str) -> List[Tuple[int, int]]:
        if self.kind == "Llama":                       # fast tokenizer has it
            enc = self.tok(text,
                           add_special_tokens=False,
                           return_offsets_mapping=True)
            return enc["offset_mapping"]

        if self.kind == "DeepSeek":                    # fast tokenizer has it
            # --- slow fallback for tiktoken -------------------------------
            cur, spans = 0, []
            for tid in self.encode(text):
                tok = self.decode([tid])
                idx = text.find(tok, cur)
                if idx == -1: idx = cur
                spans.append((idx, idx + len(tok)))
                cur = idx + len(tok)
            return spans

# -----------------------------------------------------------
# 2.  select_tokenizer() convenience helper
# -----------------------------------------------------------
def select_tokenizer(model_name: str) -> TokenAdapter:
    """
    Return a TokenAdapter for the given model name.
    Extend the if/elif map if you have private checkpoints whose
    repo names do not include the family keyword.
    """
    if "gpt" in model_name or "openai" in model_name:
        # OpenAI models use tiktoken
        return tiktoken.get_encoding("cl100k_base")
    else:
        return TokenAdapter(model_name)


#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
Hierarchical Chunker class for LLama4 and DeepSeek V3
"""

class HierarchicalChunker:

    @staticmethod
    def strip_code_fences(segments: List[str]) -> List[str]:
        """
        Remove stray triple-backtick code-fence markers but KEEP any spaces
        that belong to the original text.

        • Strips an opening fence  ```[lang]   that appears at the very start.
        ─ Removes the backticks and optional language tag, **not** the first space.
        • Strips a closing fence   ```         that appears at the very end.
        ─ Removes the backticks (and any spaces after them), **not** the last space.
        """
        cleaned: List[str] = []
        for seg in segments:
            # opening fence: optional leading spaces, ``` and optional language tag
            seg = re.sub(r'^\s*```[\w+-]*', '', seg)

            # closing fence: ``` plus any trailing whitespace up to segment end
            seg = re.sub(r'```[\s]*$', '', seg)

            cleaned.append(seg)
        return cleaned


    def __init__(self, api_key: str,
                 model: str = None,  #"deepseek-ai/DeepSeek-V3",
                 *,
                 client: Optional[object] = None,
                 k: int = 4,
                 short_cutoff: int = 6,
                 max_workers: int = 8,
                 multithread: bool = False,
                 debug: bool = False,
                 enable_fallback: bool = True
                 ):
        """
        Initialize the hierarchical chunker.

        Args:
            api_key:       OpenAI API key
            model:         OpenAI model to use
            k:             Maximum number of segments at each level
            short_cutoff:  If the span has ≤ this many tokens,
                           use index-based segmentation
        """
        self.multithread  = multithread
        self.max_workers  = max_workers      # remember the desired pool size
        self.debug       = debug
        self.enable_fallback = enable_fallback

        # stats + lock (needed for multithread mode)
        self.total_spans          = 0
        self.fallback_spans       = 0
        self.fallback_token_total = 0
        self.fallback_lengths: list[int] = []
        self.fallback_segments: list[List[str]] = []
        self.fallback_texts: list[str] = []

        # NEW: keep all individual failures
        self.failed_unsplit_counts: list[int] = []          # just the counts
        self.failed_unsplit_records: list[dict] = []        # optional rich info

        # self._lock = threading.Lock()
        # self._leaf_lock   = threading.Lock()

        self._lock = getattr(self, "_lock", threading.Lock())

        self.leaf_count   = 0
        self.total_tokens = 0
        self.pbar: Optional[tqdm] = None
        self.api_key = api_key

        if client is not None:
            self.client = client
        else:
            self.client = Together(api_key = self.api_key)

        self.model  = model
        self.k      = k
        self.tok = select_tokenizer(model)

        # --- new attribute ---
        self.short_cutoff = short_cutoff

    # ───────────────────────────────────────────────────────────────────
    def _call_llm(self,
                  system_prompt: str,
                  user_prompt: str,
                  *,
                  temperature: float = 0.1,
                  max_tokens: int = 2000,
                  model: Optional[str] = None) -> str:
        """
        Unified LLM wrapper.

        Returns
        -------
        str  – stripped content of the first assistant message.
        """
        chosen_model = model or self.model      # default to instance model

        response = self.client.chat.completions.create(
            model      = chosen_model,
            messages   = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = temperature,
            max_tokens  = max_tokens,
        )
        return response.choices[0].message.content.strip()
    # ───────────────────────────────────────────────────────────────────

    def tokenize_text(self, text: str, tokenizer: TokenAdapter,
                    *, debug=False) -> Tuple[List[str], List[Tuple[int,int]]]:
        ids      = tokenizer.encode(text)
        offsets  = tokenizer.offsets(text)           # ← now list of tuples
        tokens   = [text[s:e] for s, e in offsets]

        if debug:
            print("[tokenize_text] first 5 token–offset pairs:")
            for t,(s,e) in zip(tokens[:5], offsets[:5]):
                print(f"  {t!r:<12} {s:>4}–{e}")

        return tokens, offsets


    # def get_system_prompt(self) -> str:
    #     """Get the system prompt for LLM segmentation."""
    #     return (
    #         "You are a text-segmentation assistant.\n"
    #         f"Split the entire input into up to *{self.k}* contiguous, non-overlapping segments such that concatenating them in order reproduces the original text exactly. Including ALL whitespaces and punctuation.\n"
    #         "Each segment should be semantically coherent.\n"
    #         "You should always return more than one segment, unless it is already a single token.\n"
    #         "In the case of simple phrases, segment into semantically contiguous chunks of tokens.\n"
    #         "*CRITICAL*: Preserve ALL whitespace and quotation marks exactly as it appears in the input. If the input starts with a space, the first segment should start with that space.\n"
    #         "Return a raw JSON array of strings — do NOT wrap in markdown or code fences.\n"
    #         "Examples:\n"
    #         "Input: \"The quick brown fox jumps over the lazy dog.\"\n"
    #         "Output: [\"The quick brown fox\", \" jumps over\", \" the lazy dog.\"]\n\n"
    #         "Input: \" brown fox\"\n"
    #         "Output: [\" brown\", \" fox\"]\n\n"
    #         "Input: \"the lazy dog.\"\n"
    #         "Output: [\"the\", \" lazy dog\", \".\"]\n\n"
    #         "Input: \" jumps over\"\n"
    #         "Output: [\" jumps\", \" over\"]\n\n"
    #     )

    def get_system_prompt(self) -> str:
        """Get the system prompt for LLM segmentation."""
        return (
            "You are a text-segmentation assistant.\n"
            f"Split the entire input into up to *{self.k}* contiguous, non-overlapping segments such that concatenating them in order reproduces the original text exactly. Including ALL whitespaces and punctuation.\n"
            "Each segment should be semantically coherent.\n"
            "In the case of simple phrases, segment into semantically contiguous chunks of tokens.\n"
            "*CRITICAL*: Preserve ALL whitespace and quotation marks exactly as it appears in the input. If the input starts with a space, the first segment should start with that space.\n"
            "Return a raw JSON array of strings — do NOT wrap in markdown or code fences.\n"
            "Examples:\n"
            "Input: \"The quick brown fox jumps over the lazy dog.\"\n"
            "Output: [\"The quick brown fox\", \" jumps over\", \" the lazy dog.\"]\n\n"
            "Input: \" brown fox\"\n"
            "Output: [\" brown\", \" fox\"]\n\n"
            "Input: \"the lazy dog.\"\n"
            "Output: [\"the\", \" lazy dog\", \".\"]\n\n"
            "Input: \" jumps over\"\n"
            "Output: [\" jumps\", \" over\"]\n\n"
        )

#            "You should always return more than one segment, unless it is already a single token.\n"


    def get_sentence_system_prompt(self) -> str:
        """
        Sentence-ID mode, but the model returns a list of CUT POINTS
        rather than groups.  A cut-point i (1 ≤ i < N) means
        “start a new segment **at sentence i**”.
        """
        return (
            "You are a text-segmentation assistant **in CUT-POINT mode**.\n"
            f"The input will be N numbered sentences (0 … N-1).\n"
            f"Choose up to *{self.k-1}* cut points so that the resulting segments\n"
            "are semantically contiguous and most conceptually disjoint from the other segments.\n"
            "Return **raw JSON** – a strictly ascending list of integers in the\n"
            "range 1 … N-1.  Do NOT wrap in markdown.\n"
            "Example (two segments):\n"
            "Input:\n"
            "[0] Yeah I was in the boy scouts at the time.\n"
            "[1] And we was doing the 50-yard dash racing but we was at the pier marked off and so we was doing the 50-yard dash.\n"
            "[2] There was about 8 or 9 of us you know, going down, coming back.\n"
            "[3] And going down the third time I caught cramps and I started yelling 'Help!' but the fellows didn't believe me you know.\n"
            "[4] They thought I was just trying to catch up because I was going on or slowing down.\n"
            "Output: [1, 3]\n"
        )

    # def get_indices_system_prompt(self) -> str:
    #         return(
    #             "You are a **phrase-level segmentation assistant**.\n"
    #             f"Given a SHORT span (≤ {self.short_cutoff} tokens) and its token list with indices, "
    #             "return a JSON array of **cut-points** (0-based integers) where a new "
    #             "segment should START.\n\n"

    #             "Principles (in order of importance):\n"
    #             "1. **Meaningful units** – segments should be the largest contiguous group "
    #             "   of tokens that form a coherent phrase (e.g. determiner + noun, "
    #             "   adjective + noun, verb + particle).\n"
    #             "2. **Non-trivial cuts** – produce **at least one** cut-point but "
    #             f"no more than *{self.k-1}*, yielding up to *{self.k}* segments in "
    #             "total; NEVER split into single-token segments unless a token truly stands "
    #             "alone (punctuation, conjunction, etc.).\n"
    #             "3. **Ascending order** – cut-points must be strictly increasing and "
    #             "between 1 and len(tokens)-1.\n"
    #             "4. **JSON only** – respond with the RAW JSON array (e.g. [3]) and nothing "
    #             "else.\n\n"

    #             "Guidelines:\n"
    #             "• Keep a leading article with its noun:  ['The', ' quick', ' brown', ' fox']\n"
    #             "  ⇒  cut-points [3]   →  ['The quick brown', ' fox']\n"
    #             "• Keep adjective(s) with the noun they modify.\n"
    #             "• Keep multi-word verbs together: [' jumps', ' over'] stays intact.\n"
    #             "• Attach sentence-final punctuation to the preceding word.\n"
    #             "• If the span is already one coherent phrase, output an empty array [].\n"
    #             "If you believe the span is already atomic, still output an empty array []."
    #             "Otherwise you MUST return ≥ 1 cut.\n"
    #             "If you start to output an empty array but then realise cuts are possible,"
    #             "**regenerate** your answer instead of sending the empty array.\n"
    #         )

    # simplified version
    def get_indices_system_prompt(self) -> str:
            return(
                "You are a **phrase-level segmentation assistant**.\n"
                "Given a token list with indices, "
                "return a JSON array of cut-points (0-based) where a new segment starts. \n"
                f"no more than *{self.k-1}*, yielding up to *{self.k}* segments in "
                "total. \n"
                "Segments should be the largest contiguous group of tokens that "
                "form a coherent phrase (e.g. determiner + noun, adjective + noun, verb + particle).\n"
                "Attach sentence-final punctuation to the preceding word.\n"
                "Cut-points must be strictly increasing and between 1 and len(tokens)-1.\n"
                "**JSON only** – respond with the RAW JSON array (e.g. [3]) and nothing else.\n"
                "Example: tokens ['The',' quick',' brown',' fox'] → cut-points [3] → ['The quick brown',' fox']\n"
            )

# ; NEVER split into single-token segments unless a token truly stands alone

    # ❶  ────────────────────────────────────────────────────────────────────

    def reconcile_boundaries(self, original: str, segments: List[str]) -> List[str]:
        """
        Post-process *segments* so that their concatenation equals *original*
        even when the LLM dropped / duplicated leading spaces or quotes.

        Order of fixes
        ---------------
        1. leading whitespace
        2. ending quotation mark(s)
        3. existing length-based add / trim
        """

        # ---------------------------------------------------------------------
        # Character classes used throughout
        # ---------------------------------------------------------------------
        _WS          = r' \t'                      # leading whitespace
        _QUOTES      = r'"“”‘’\''                  # straight + curly + single
        _BOUNDARY_CH = _WS + _QUOTES

        _LEAD_WS_RE  = re.compile(rf'^([{_WS}]+)')          # run of leading spaces/tabs
        _TRAIL_Q_RE  = re.compile(rf'([{_QUOTES}]+\s*)$')   # quotes (opt. space) at end
        _LEAD_ANY_RE = re.compile(rf'^([{_BOUNDARY_CH}]+)')
        _TAIL_ANY_RE = re.compile(rf'([{_BOUNDARY_CH}]+)$')

        if not segments:
            return segments

        fixed, cursor = [], 0
        for s in segments:
            idx = original.find(s, cursor)       # first occurrence at/after cursor
            if idx == -1:
                # catastrophic mismatch → stop trying
                return segments
            gap = original[cursor:idx]           # bytes that got lost
            fixed.append(gap + s)                # prepend them to current slice
            cursor = idx + len(s)

        segments = fixed
        # ────────────────────────────────────────────────────────────────
        # 1️⃣  Leading whitespace  (spaces/tabs only)
        # ────────────────────────────────────────────────────────────────
        lead_ws_match = _LEAD_WS_RE.match(original)
        if lead_ws_match:
            lead_ws = lead_ws_match.group(0)
            if not segments[0].startswith(lead_ws):
                segments[0] = lead_ws + segments[0]

                if ''.join(segments) == original:
                    return segments

        # ────────────────────────────────────────────────────────────────
        # 2️⃣  Ending quotation mark(s) on the last segment
        #     (handles straight or curly quotes, possibly trailed by space)
        # ────────────────────────────────────────────────────────────────
        trail_q_match = _TRAIL_Q_RE.search(original)
        if trail_q_match:
            trail_q = trail_q_match.group(0)
            if not segments[-1].endswith(trail_q):
                # append the missing tail, one char at a time until it matches
                needed = trail_q
                while not segments[-1].endswith(trail_q) and needed:
                    segments[-1] += needed[0]
                    needed = needed[1:]

                if ''.join(segments) == original:
                    return segments

        # ────────────────────────────────────────────────────────────────
        # 3️⃣  Original length-based add / trim (your previous logic)
        # ────────────────────────────────────────────────────────────────
        _FRONT = _BOUNDARY_CH
        _BACK  = _FRONT[::-1]

        concat = ''.join(segments)
        if len(concat) < len(original):
            # -------- add missing on the left ----------
            lead_any = _LEAD_ANY_RE.match(original)
            if lead_any:
                lead = lead_any.group(0)
                while not segments[0].startswith(lead) and lead:
                    segments[0] = lead[-1] + segments[0]
                    lead = lead[:-1]

            # -------- add missing on the right ----------
            tail_any = _TAIL_ANY_RE.search(original)
            if tail_any:
                tail = tail_any.group(0)
                while not segments[-1].endswith(tail) and tail:
                    segments[-1] = segments[-1] + tail[0]
                    tail = tail[1:]

        # -------- trim surplus if now too long ----------
        concat = ''.join(segments)
        if len(concat) > len(original):
            surplus = len(concat) - len(original)
            while surplus and segments[0] and segments[0][0] in _FRONT:
                segments[0] = segments[0][1:]
                surplus -= 1
            while surplus and segments[-1] and segments[-1][-1] in _BACK:
                segments[-1] = segments[-1][-1:]
                surplus -= 1

        # Optional debug output
        if ''.join(segments) != original and getattr(self, "debug", False):
            print("[reconcile_boundaries] could not match bytes exactly")
            print(f"  original: '{original}'")
            print(f"  segments: {segments}")

        # return segments

        segs = []
        carry = ""
        for s in segments[:-1]:
            full = carry + s
            carry = re.search(r'\s*$', full).group(0)     # trailing WS
            segs.append(full[:-len(carry)] or full)       # keep non-empty
        segs.append(carry + segments[-1])
        return segs


    def segment_text(self, text: str, temperature: float = 0.1, level: int = 0) -> List[str]:
        """
        Use LLM to segment text into chunks.

        Args:
            text: Text to segment
            temperature: OpenAI temperature parameter
            level: Current level for debugging

        Returns:
            List of text segments
        """
        system_prompt = self.get_system_prompt()
        user_prompt = f"```{text}```"

        if self.debug:
            print(f"\n{'='*60}")
            print(f"SEGMENTING AT LEVEL {level} (temp={temperature})")
            print(f"{'='*60}")
            print(f"Input text: '{text}'")
            # print(f"System prompt: {system_prompt}")
            # print(f"User prompt: {user_prompt}")

        try:
            content = self._call_llm(system_prompt,
                                    user_prompt,
                                    temperature=temperature)

            if self.debug:
                print(f"LLM Response: {content}")

            # Try to parse as JSON
            try:
                # segments = json.loads(content)
                # segments = self.strip_code_fences(segments)  # <-- add this line
                segments_json = json.loads(content)
                segments_raw = self.strip_code_fences(segments_json)  # <-- add this line
                segments = self.reconcile_boundaries(text, segments_raw)

                if isinstance(segments, list):

                    # if self.debug:
                    #     print(f"Parsed segments: {segments}")
                    #     print(f"Verification: {''.join(segments)} == {text} -> {self.verify_segmentation(text, segments)}")
                    return segments

            except json.JSONDecodeError:
                if self.debug:
                    print(f"JSON parse failed: {content}")

            # Fallback: try to extract array from response
            array_match = re.search(r'\[(.*?)\]', content, re.DOTALL)
            if array_match:
                array_content = '[' + array_match.group(1) + ']'

                segments = json.loads(array_content)
                if isinstance(segments, list):
                    # if self.debug:
                    #     print(f"Fallback parsed segments: {segments}")
                    #     print(f"Verification: {''.join(segments)} == {text} -> {self.verify_segmentation(text, segments)}")
                    return segments

            raise ValueError(f"Could not parse LLM response: {content}")

        except Exception as e:
            if self.debug:
                print(f"Error in text segmentation mode: {e}")
                print(content)
            return [text]  # Return original text as single segment on failure

    # ❷  ────────────────────────────────────────────────────────────────────
    # NEW helper: prompt + call for the "≤ short_cutoff tokens" case
    def segment_by_indices(self,
                           text_span: str,
                           span_tokens: List[str],
                           temperature: float,
                           level: int) -> Optional[List[str]]:
        """
        Ask the LLM for integer *cut points* rather than sliced text when the
        span is short.  Returns the correctly reconstructed segments.
        """
        # Build a numbered token list to show the model
        enumerated = [f"{i}: '{tok}'" for i, tok in enumerate(span_tokens)]
        token_table = "\n".join(enumerated)
        system_prompt = self.get_indices_system_prompt()

        user_prompt = (
            f"Text span:\n```{text_span}```\n\n"
            f"Tokens ({len(span_tokens)} total):\n{token_table}"
        )

        try:
            raw = self._call_llm(system_prompt,
                     user_prompt,
                     temperature=temperature)

            # Robust, fence-free parse:
            cuts = literal_eval(re.search(r'\[.*\]', raw).group(0))

            if not cuts:                         # empty list ⇒ no split
                return None

            # Build segments from cut indices
            segments: List[str] = []
            start = 0
            for cut in cuts:
                segments.append(''.join(span_tokens[start:cut]))
                start = cut
            segments.append(''.join(span_tokens[start:]))

            return segments

        except Exception as e:
            # print(f"Error in token-index segmentation mode: {e}")
            return None                          # ← parse / regex / eval failed

    # ❸  ────────────────────────────────────────────────────────────────────

    def sentences_with_spans(self, text: str) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Sentence splitter for transcript-like text.

        • Leading spaces belong to the sentence **they precede**.
        • Trailing spaces after the final punctuation belong to the *next*
        sentence (or are dropped if none).
        That matches GPT-style tokenisers, which encode leading spaces as part
        of the following token (e.g. `' But'`).
        """
        # Regex:   ^\s*       → skip initial whitespace once
        #          .*?        → minimal grab until ...
        #          [.!?]      →   a sentence-ending punctuation
        #          (?=|\s)    →   *peek* at whitespace/end but don't consume it
        SENT_RE = re.compile(r'\s*.*?[.!?](?=\s|$)', re.DOTALL)

        out: List[Tuple[str, Tuple[int, int]]] = []
        idx = 0
        for m in SENT_RE.finditer(text):
            span_start, span_end = m.span()
            sent = text[span_start:span_end]
            out.append((sent, (span_start, span_end)))
            idx = span_end                        # next search resumes here

        # Handle a possible tail fragment without terminal punctuation
        if idx < len(text):
            out.append((text[idx:], (idx, len(text))))

        return out


    def segment_by_sentence(
            self,
            text: str,
            temperature: float = 0.1,
            level: int = 0,
        ) -> List[str]:
        """
        Long-span splitter that:
        1. numbers the sentences locally,
        2. asks the LLM for cut-points,
        3. re-assembles exact substrings.
        """
        sents = self.sentences_with_spans(text)
        N = len(sents)
        if N <= 1:
            return [text]

        numbered = "\n".join(f"[{i}] {sents[i][0]}" for i in range(N))

        content = self._call_llm(
            self.get_sentence_system_prompt(),
            numbered,
            temperature=temperature,
            max_tokens=200,
            model=self.model,
        )
        if self.debug:
            print("[segment_by_sentence] LLM raw output:", repr(content))

        # ------------------------------------------------------------
        # Parse cut-points ① plain → ② strip fence → fallback
        # ------------------------------------------------------------
        clean = re.sub(r'^\s*```[\w+-]*\s*|\s*```\s*$', '', content).strip()
        for raw in (content, clean):
            try:
                cuts = json.loads(raw)
                if (isinstance(cuts, list)
                        and all(isinstance(c, int) for c in cuts)):
                    cuts = sorted({c for c in cuts if 1 <= c < N})
                    seg_bounds = [0] + cuts + [N]
                    segments = [
                        text[sents[seg_bounds[i]][1][0]: sents[seg_bounds[i+1]-1][1][1]]
                        for i in range(len(seg_bounds)-1)
                    ]
                    if self.verify_segmentation(text, segments):
                        return segments
            except json.JSONDecodeError as e:
                if self.debug:
                    print("   -- JSON parse failed:", e)

        # Final fallback: return entire span
        if self.debug:
            print("   -- Falling back to single segment")
        return [text]


    def verify_segmentation(self, original: str, segments: List[str]) -> bool:
        """
        Verify that concatenating segments reproduces the original text.
        """
        concatenated = ''.join(segments)
        text_match = concatenated.rstrip() == original.rstrip()
        valid_count = len(segments) <= self.k  # Must have >1 segment unless single token

        # if self.debug:
        #     print(f"Verifying segmentation:")
        #     print(f"  Original: '{original}' (len={len(original)})")
        #     print(f"  Concatenated: '{concatenated}' (len={len(concatenated)})")
        #     print(f"  Original bytes: {original.encode('utf-8')}")
        #     print(f"  Concatenated bytes: {concatenated.encode('utf-8')}")
        return text_match and valid_count


    def find_token_ranges(self,
            segments: Sequence[str],
            offsets:  Sequence[Tuple[int,int]],
            *,
            text: str,
            start_token_idx: int = 0
    ) -> List[Tuple[int,int]]:

        ranges, tok = [], start_token_idx
        char_cursor = offsets[tok][0] if tok < len(offsets) else 0

        for seg in segments:
            seg_start = text.find(seg, char_cursor)
            if seg_start == -1:
                seg_start = char_cursor
            seg_end = seg_start + len(seg)

            while tok < len(offsets) and offsets[tok][1] <= seg_start:
                tok += 1
            left = tok

            while tok < len(offsets) and offsets[tok][0] < seg_end:
                tok += 1
            right = tok

            ranges.append((left, right))
            char_cursor = seg_end
        return ranges


    def _bump_total(self):
        with self._lock:
            self.total_spans += 1

    def _bump_fallback(self, tok_count: int, text: str, failed_segments: List[str]):
        with self._lock:
            self.fallback_spans       += 1
            self.fallback_token_total += tok_count
            self.fallback_lengths.append(tok_count)   # NEW – record the length
            if tok_count > 10:
                self.fallback_segments.append(failed_segments)
                self.fallback_texts.append(text)

    def _record_failed_unsplit(self,
                            token_count: int,
                            text: str,
                            failed_segments,
                            *,
                            level: int = None,
                            start_token_idx: int = None,
                            end_token_idx: int = None) -> None:
        """Accumulate info for spans that could not be split and we chose not to fallback."""
        with self._lock:
            self.failed_unsplit_counts.append(token_count)
            self.failed_unsplit_records.append({
                "tokens": token_count,
                "level": level,
                "start": start_token_idx,
                "end": end_token_idx,
                "text": text,
                "attempted_segments": failed_segments or [],
            })
        if getattr(self, "debug", False):
            print(f"[No-fallback] failed to split span ({token_count} tokens) @ level={level}, "
                f"{start_token_idx}:{end_token_idx} → {text!r}")


    def _bump_leaf(self, n_tokens: int) -> int:
        """
        Advance progress by the number of tokens covered by this leaf.
        Returns the running leaf counter for logging.
        """
        # keep an internal counter if you want to label leaves
        if not hasattr(self, "_leaf_counter"):
            self._leaf_counter = 0
        self._leaf_counter += 1

        if self.pbar is not None and n_tokens > 0:
            self.pbar.update(n_tokens)   # <-- crucial: advance by token count

        return self._leaf_counter


    def _binary_split_tokens(self, span_tokens):
        """
        Deterministically split a list of tokens into TWO contiguous halves.

        • If the span has N tokens, the left segment gets ⌊N/2⌋ tokens,
        the right gets the rest.
        • All original spacing is preserved because tokens already hold
        their leading spaces.
        """
        mid = len(span_tokens) // 2            # floor division
        left  = ''.join(span_tokens[:mid])
        right = ''.join(span_tokens[mid:])
        return [left, right]

    def _print_stats(self) -> None:
        """
        Report how often the deterministic fallback was used and the
        mean token-length of those rescued spans.
        """
        if self.total_spans == 0:
            print("\nNo multi-token spans processed.")
            return

        if self.enable_fallback==True:

            fallback_rate = self.fallback_spans / self.total_spans
            avg_len = (self.fallback_token_total / self.fallback_spans
                    if self.fallback_spans else 0)

            print(
                f"\nFallback used on {self.fallback_spans}/{self.total_spans} spans "
                f"({fallback_rate:.1%}); "
                f"average span length when fallback triggered: {avg_len:.2f} tokens"
            )

        if self.enable_fallback==False:
            failed_n = self.failed_unsplit_counts
            failed_sum = sum(self.failed_unsplit_counts) if failed_n else 0
            avg_len = (failed_sum / len(failed_n)) if failed_n else 0
            print(
                f"(Sum. tokens = {failed_sum}, avg. = {avg_len:.2f} tokens)"
            )

    def _normalize_ranges(self, ranges, parent_start, parent_end):
        """
        Keep only valid, strictly increasing, non-overlapping, non-empty slices
        within [parent_start, parent_end). Returns a cleaned list.
        """
        # clamp to parent bounds and drop empties
        clipped = []
        for (l, r) in ranges:
            l2 = max(parent_start, min(l, parent_end))
            r2 = max(parent_start, min(r, parent_end))
            if r2 > l2:
                clipped.append((l2, r2))

        # sort & dedupe, enforce monotonicity
        clipped.sort()
        cleaned = []
        cursor = parent_start
        for (l, r) in clipped:
            if l < cursor:        # overlaps or goes backwards → drop
                continue
            cleaned.append((l, r))
            cursor = r

        return cleaned

    def recursive_segment(
            self,
            text: str,
            tokens: List[str],
            offsets: List[int],
            start_token_idx: int = 0,
            level: int = 0,
            max_temp: float = 1.0,
            end_token_idx: Optional[int] = None,      # ← NEW
    ) -> Dict[str, Any]:

        """
        Recursively segment text into hierarchical chunks.

        Args:
            text: Text to segment
            tokens: All tokens from original text
            token_indices: Character indices of all tokens
            start_token_idx: Starting token index
            level: Current recursion level
            max_temp: Maximum temperature to try

        Returns:
            Dictionary representing hierarchical segmentation
        """
        # if self.debug:
        #     print(f"\n--- RECURSIVE SEGMENT LEVEL {level} ---")
        #     print(f"Text: '{text}'")
        #     print(f"Start token idx: {start_token_idx}")


        if end_token_idx is not None:
            token_count = end_token_idx - start_token_idx
        else:
            # Calculate how many tokens this text segment contains
            # by reconstructing from original tokens
            reconstructed = ""
            token_count = 0
            temp_idx = start_token_idx

            while temp_idx < len(tokens) and len(reconstructed) < len(text):
                if reconstructed + tokens[temp_idx] == text[:len(reconstructed + tokens[temp_idx])]:
                    reconstructed += tokens[temp_idx]
                    token_count += 1
                    temp_idx += 1
                else:
                    break
        #     if self.debug:
        #         print(f"Reconstructed: '{reconstructed}'")
        #         print(f"Match: {reconstructed == text}")

        # if self.debug:
        #     print(f"Token count for this segment: {token_count}")


        # --- bookkeeping for fallback statistics --------------------------
        if token_count > 1:                  # only spans that *can* be split
            self.total_spans += 1


        # If this is a single token, return as leaf
        if token_count <= 1:
            # leaf_num = self._bump_leaf()
            leaf_num = self._bump_leaf(token_count)  # <- not 1

            if self.debug:
                print(f"[Leaf {leaf_num}] level={level}, tokens {start_token_idx}:{start_token_idx+token_count} → “{text!r}”")

            return {
                'text': text,
                'token_range': (start_token_idx, start_token_idx + token_count),
                'level': level,
                'children': None
            }

        # Decide which segmentation strategy to use
        use_index_mode = (1 < token_count <= self.short_cutoff)
        use_sentence_mode = (token_count >= 200) #500


        temperature = 0.1
        segments = None
        successful_segmentation = False

        while temperature <= max_temp:
            if use_index_mode:
                # --- New path for short spans ----------------------------
                span_tokens = tokens[start_token_idx : start_token_idx + token_count]
                segments = self.segment_by_indices(text, span_tokens,
                                                   temperature, level)
            elif use_sentence_mode:
                segments = self.segment_by_sentence(text, temperature, level)

            else:
                # --- Original path --------------------------------------
                segments = self.segment_text(text, temperature, level)

            # Verify & accept
            if (segments and len(segments) > 1 and
                    self.verify_segmentation(text, segments) and
                    all(seg.strip() for seg in segments)):
                successful_segmentation = True
                break

            temperature += 0.2

        # if not successful_segmentation and token_count > 1:
        #     failed_segments = segments.copy() if segments else []
        #     # --- deterministic binary split fallback --------------------------
        #     span_tokens = tokens[start_token_idx : start_token_idx + token_count]

        #     segments = self._binary_split_tokens(span_tokens)

        #     # accept the fallback only if it reconstructs perfectly
        #     if self.verify_segmentation(text, segments):
        #         successful_segmentation = True
        #         self._bump_fallback(token_count, text, failed_segments)   # ← single, thread-safe call

        if not successful_segmentation and token_count > 1:
            failed_segments = segments.copy() if segments else []

            if self.enable_fallback:
                # --- deterministic binary split fallback ---
                span_tokens = tokens[start_token_idx : start_token_idx + token_count]
                segments = self._binary_split_tokens(span_tokens)

                if self.verify_segmentation(text, segments):
                    successful_segmentation = True
                    self._bump_fallback(token_count, text, failed_segments)
            else:
                # No fallback: record and return a LEAF for the whole span
                self._record_failed_unsplit(
                    token_count,
                    text,
                    failed_segments,
                    level=level,
                    start_token_idx=start_token_idx,
                    end_token_idx=start_token_idx + token_count,
                )
                leaf_num = self._bump_leaf(token_count)
                ...
                return {
                    'text': text,
                    'token_range': (start_token_idx, start_token_idx + token_count),
                    'level': level,
                    'children': None
                }

        # ----------------------------------------------------------------------

        # If segmentation failed, return as leaf node
        if not successful_segmentation or len(segments) <= 1:
            # leaf_num = self._bump_leaf()
            leaf_num = self._bump_leaf(token_count)  # <- not 1

            if self.debug:
                print(f"[Leaf {leaf_num}] (fallback) level={level}, tokens {start_token_idx}:{start_token_idx+token_count} → “{text!r}”")
            return {
                'text': text,
                'token_range': (start_token_idx, start_token_idx + token_count),
                'level': level,
                'children': None
            }

        # Find token ranges for segments
        token_ranges = self.find_token_ranges(
            segments,
            offsets,
            start_token_idx = start_token_idx,
            text=text          # ← parent span string
        )

        # after: token_ranges = self.find_token_ranges(...)
        parent_start = start_token_idx
        parent_end   = start_token_idx + token_count
        parent_len   = parent_end - parent_start

        # normalize to parent bounds, drop empties, sort, de-overlap
        token_ranges = self._normalize_ranges(token_ranges, parent_start, parent_end)

        # coverage & sanity checks
        covered       = sum(r - l for (l, r) in token_ranges)
        any_self_span = any((r - l) == parent_len for (l, r) in token_ranges)
        bad_mapping   = (
            not token_ranges or
            covered != parent_len or                 # gaps or overlaps
            any_self_span or                         # child equals parent
            len(token_ranges) != len(segments)       # mismatch with LLM chunk count
        )

        if bad_mapping:
            if not self.enable_fallback:
                self._record_failed_unsplit(
                    parent_len, text, failed_segments=[],
                    level=level,
                    start_token_idx=parent_start,
                    end_token_idx=parent_end,
                )
                leaf_num = self._bump_leaf(token_count)

                if self.debug:
                    print(f"[Leaf {leaf_num}] (no-fallback: bad mapping) level={level}, "
                        f"tokens {parent_start}:{parent_end} → {text!r}")
                return {
                    'text': text,
                    'token_range': (parent_start, parent_end),
                    'level': level,
                    'children': None
                }

            # ⬇️ instead of "return leaf", do a deterministic binary split
            mid = parent_start + (parent_len // 2)
            if mid == parent_start or mid == parent_end:
                # ultra-edge case (shouldn't happen): fall back to leaf
                leaf_num = self._bump_leaf(token_count)
                if self.debug:
                    print(f"[Leaf {leaf_num}] (degenerate mapping) level={level}, "
                        f"tokens {parent_start}:{parent_end} → {text!r}")
                return {
                    'text': text,
                    'token_range': (parent_start, parent_end),
                    'level': level,
                    'children': None
                }

            token_ranges = [(parent_start, mid), (mid, parent_end)]
            segments = [
                ''.join(tokens[parent_start:mid]),
                ''.join(tokens[mid:parent_end]),
            ]
            # optional bookkeeping
            self._bump_fallback(parent_len, text, failed_segments=[])


        # if self.debug:
        #     print(f"Finding token ranges for segments: {segments}")
        #     print(f"Token ranges: {token_ranges}")

        # Build canonical text for every slice so leading / trailing spaces match
        canonical_segments = [
            ''.join(tokens[l:r]) for (l, r) in token_ranges
        ]

        # ──────────────────────────────────────────────────────────────────────
        # Recursively segment each chunk
        # ──────────────────────────────────────────────────────────────────────
        children = [None] * len(segments)

        if self.multithread and len(segments) > 1:
            # depth-local thread pool
            with cf.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                future_to_idx = {
                    pool.submit(
                        self.recursive_segment,
                        canonical_segments[idx],           # ← use canonical text
                        tokens,
                        offsets,
                        tr[0],                 # start idx
                        level + 1,
                        max_temp,
                        end_token_idx=tr[1]    # ← pass the right boundary
                    ): idx
                    for idx, (seg, tr) in enumerate(zip(segments, token_ranges))
                }
                for fut in cf.as_completed(future_to_idx):
                    children[future_to_idx[fut]] = fut.result()
        else:
            # sequential fallback
            for idx, (seg, tr) in enumerate(zip(segments, token_ranges)):
                children[idx] = self.recursive_segment(
                    canonical_segments[idx],           # ← here too
                    tokens,
                    offsets,
                    tr[0],
                    level + 1,
                    max_temp,
                    end_token_idx=tr[1]       # ← same here
                )

        return {
            'text': text,
            'token_range': (start_token_idx, start_token_idx + token_count),
            'level': level,
            'children': children
        }



    def chunk_text(self, text: str) -> Dict[str, Any]:
        """
        Main method to perform hierarchical chunking.

        Args:
            text: Input text to chunk

        Returns:
            Dictionary with hierarchical segmentation and metadata
        """
        # Step 1: Tokenize the entire input
        tokens, offsets = self.tokenize_text(text,self.tok)
        # set total tokens so progress has a denominator
        self.total_tokens = len(tokens)

        # Initialize tqdm bar once we know how many leaves to expect
        # (we set total to number of tokens, since each token becomes exactly one leaf)
        self.pbar = tqdm(
            total=len(tokens),
            desc="Hierarchical chunking",
            unit="leaf",
            ncols=80,
        )

        if self.debug:
            print(f"[chunk_text] → total tokens = {len(tokens)}")

        # Step 2: Recursive segmentation
        hierarchy = self.recursive_segment(text, tokens, offsets)

        # Done — close the bar
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

        self._print_stats()        # ← optional helper that prints rate & avg length
        return {
            'original_text': text,
            'tokens': tokens,
            'offsets': offsets,
            'hierarchy': hierarchy
        }



    def print_hierarchy(self, node: Dict[str, Any], indent: int = 0):
        """Print the hierarchical structure in a readable format."""
        prefix = "  " * indent
        token_range = node['token_range']
        print(f"{prefix}Level {node['level']}: '{node['text']}' (tokens {token_range[0]}-{token_range[1]})")

        if node['children']:
            for child in node['children']:
                self.print_hierarchy(child, indent + 1)


#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
Helper functions for processing text
"""


def hierarchy_to_span_tree(node: dict) -> dict:
    """
    Convert the chunker’s node format

        {'text': str,
         'token_range': (start_idx, end_idx),
         'level': int,
         'children': [ ... ] or None}

    into the minimal span dictionary used by downstream code:

        {'start': int,
         'end': int,
         'children': [ ... ]}

    Notes
    -----
    • `start` is inclusive, `end` is exclusive (Python-slice style).
    • If the node has no children, an *empty list* is stored (not None) so the
      output tree is homogenous and easy to traverse.
    """
    start, end = node["token_range"]
    children_nodes = node.get("children") or []

    return {
        "start": start,
        "end":   end,
        "children": [hierarchy_to_span_tree(child) for child in children_nodes]
    }

def bracketize_hierarchy(node: dict, level: int = 0) -> str:
    """
    Recursively wrap each child segment in square brackets, so that
    the final string visualises *every* split point.

    Example for top-level ["The quick brown", " fox"]:
        '[The quick brown][ fox]'

    Nested splits generate nested brackets:
        '[[The][ quick brown]][ fox]'

    Parameters
    ----------
    node   : a node from `chunker.chunk_text(... )["hierarchy"]`
    level  : recursion depth (the root is 0).  Kept only in case you want
             different bracket symbols for different levels.

    Returns
    -------
    A single string equal in bytes to the original text except for the
    added '[' and ']' markers.
    """
    # Leaf → just return raw text
    if not node.get("children"):
        return node["text"]

    # Recursively bracket each child, then glue them together
    parts = [bracketize_hierarchy(child, level + 1) for child in node["children"]]

    # Surround every child with its own pair of brackets
    bracketed_children = "".join(f"[{p}]" for p in parts)

    return bracketed_children


# def clean_input_text(text: str) -> str:
#     """
#     Sanitize user text before chunking.

#     Steps
#     -----
#     0.  Remove an outer ```fence``` if present.
#     1.  Normalise CR/LF line-break variants.
#     2.  Replace newline / tab with a single space.
#     3.  *Insert a space* after ".", "!" or "?" if not already followed by
#         whitespace and the next char starts a new sentence (capital or digit).
#     4.  Collapse runs of   2+ spaces  into a single space.
#     5.  Trim leading / trailing whitespace.
#     6.  Drop control characters (except the newline we already turned to space).
#     7.  Remove all single/double quotation marks (ASCII + common Unicode forms).

#     The resulting string preserves interior spaces and punctuation (minus quotes),
#     ensuring the chunker & verifier see the exact same byte sequence.
#     """


#     # ── 0️⃣ remove outer triple-backtick fence ───────────────────────────
#     fence = re.compile(r"^\s*```[\w+-]*\n?(.*?)\n?```\s*$", re.DOTALL)
#     m = fence.match(text)
#     if m:
#         text = m.group(1)

#     # ── 1️⃣ universal newlines ──────────────────────────────────────────
#     text = text.replace('\r\n', '\n').replace('\r', '\n')

#     # ── 2️⃣ newline / tab → single space ────────────────────────────────
#     text = re.sub(r'[\n\t]+', ' ', text)

#     # ── 3️⃣ fix missing sentence space  (".They" → ". They") ────────────
#     sent_end = re.compile(r'([.!?])([A-Z0-9])')
#     text = sent_end.sub(r'\1 \2', text)

#     # ── 4️⃣ collapse multiple spaces ────────────────────────────────────
#     text = re.sub(r' {2,}', ' ', text)

#     # ── 5️⃣ trim outer spaces ───────────────────────────────────────────
#     text = text.strip()

#     # ── 6️⃣ strip other control chars ───────────────────────────────────
#     text = ''.join(ch for ch in text
#                    if unicodedata.category(ch)[0] != 'C' or ch == '\n')

#     # ── 7️⃣ remove all single/double quotes (ASCII + common Unicode) ────
#     # Enhanced version combining both approaches
#     _SINGLE_QUOTES = {
#         "'",          # U+0027  apostrophe
#         '‘', '’',     # U+2018, U+2019  left/right single curly
#         '‚', '‛',     # U+201A, U+201B  low-9 / high-reversed-9 single
#         '❛', '❜',     # U+275B, U+275C  heavy single quotes
#         '＇',         # U+FF07  full-width apostrophe
#         '‹', '›',     # U+2039, U+203A  single angle quote
#         '`',          # U+0060  grave accent (backtick)
#     }
#     _DOUBLE_QUOTES = {
#         '"',          # U+0022  double quote
#         '“', '”',     # U+201C, U+201D  left/right double curly
#         '„', '‟',     # U+201E, U+201F  low/high double comma
#         '❝', '❞',     # U+275D, U+275E  heavy double quotes
#         '＂',         # U+FF02  full-width double quote
#         '«', '»',     # U+00AB, U+00BB  angle quotes
#     }
#     _QUOTE_CHARS = _SINGLE_QUOTES | _DOUBLE_QUOTES

#     # First, unescape backslash-escaped quotes so we can remove the quote
#     # and not leave a stray backslash behind: \" → "  then remove it.
#     _escaped_class = ''.join(map(re.escape, _QUOTE_CHARS))
#     _ESCAPED_QUOTE_RE = re.compile(rf'\\([{_escaped_class}])')
#     text = _ESCAPED_QUOTE_RE.sub(r'\1', text)

#     # Delete all quote characters using efficient translation
#     _QUOTE_DELETE_TABLE = {ord(ch): None for ch in _QUOTE_CHARS}
#     text = text.translate(_QUOTE_DELETE_TABLE)

#     return text

# restore accidentally removed apostrophes in contractions like "isn't"
def clean_input_text(text: str) -> str:
    """
    Sanitize user text before chunking.

    Steps
    -----
    0.  Remove an outer ```fence``` if present.
    1.  Normalise CR/LF line-break variants.
    2.  Replace newline / tab with a single space.
    3.  *Insert a space* after ".", "!" or "?" if not already followed by
        whitespace and the next char starts a new sentence (capital or digit).
    4.  Collapse runs of   2+ spaces  into a single space.
    5.  Trim leading / trailing whitespace.
    6.  Drop control characters (except the newline we already turned to space).
    7.  Remove all single/double quotation marks (ASCII + common Unicode forms).

    The resulting string preserves interior spaces and punctuation (minus quotes),
    ensuring the chunker & verifier see the exact same byte sequence.
    """


    # ── 0️⃣ remove outer triple-backtick fence ───────────────────────────
    fence = re.compile(r"^\s*```[\w+-]*\n?(.*?)\n?```\s*$", re.DOTALL)
    m = fence.match(text)
    if m:
        text = m.group(1)

    # ── 1️⃣ universal newlines ──────────────────────────────────────────
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # ── 2️⃣ newline / tab → single space ────────────────────────────────
    text = re.sub(r'[\n\t]+', ' ', text)

    # ── 2️⃣b escape *unescaped* dollar signs to avoid math mode in plotting ──
    # turns `$` -> `\$`, but leaves `\$` unchanged
    text = text.translate({ord('$'): None, ord('\\'): None})

    # ── 3️⃣ fix missing sentence space  (".They" → ". They") ────────────
    sent_end = re.compile(r'([.!?])([A-Z0-9])')
    text = sent_end.sub(r'\1 \2', text)

    # ── 4️⃣ collapse multiple spaces ────────────────────────────────────
    text = re.sub(r' {2,}', ' ', text)

    # ── 5️⃣ trim outer spaces ───────────────────────────────────────────
    text = text.strip()

    # ── 6️⃣ strip other control chars ───────────────────────────────────
    text = ''.join(ch for ch in text
                   if unicodedata.category(ch)[0] != 'C' or ch == '\n')

    # ── 7️⃣ normalize apostrophes; remove backticks/angle quotes & all double quotes ─
    # Keep/normalize these apostrophe forms to ASCII "'" to preserve contractions
    _APOSTROPHE_NORMALIZE = {
        '’': "'",   # U+2019 right single quote
        '‘': "'",   # U+2018 left single quote (often misused as apostrophe)
        'ʼ': "'",   # U+02BC modifier letter apostrophe
        '＇': "'",  # U+FF07 full-width apostrophe
    }
    # Characters to DELETE outright (not apostrophes):
    #  - backtick/grave accent and single angle quotes
    #  - single-quote *quotation* variants
    _SINGLE_QUOTES_DELETE = {
        '‹', '›',      # U+2039, U+203A  single angle quotes
        '`',           # U+0060        backtick
        '‚', '‛',      # U+201A, U+201B low-9 / reversed-9 single quotes
        '❛', '❜',      # U+275B, U+275C heavy single quotes
    }
    # Remove all double-quote characters
    _DOUBLE_QUOTES_DELETE = {
        '"',           # U+0022
        '“', '”',      # U+201C, U+201D
        '„', '‟',      # U+201E, U+201F
        '❝', '❞',      # U+275D, U+275E
        '＂',          # U+FF02
        '«', '»',      # U+00AB, U+00BB
    }

    # We'll unescape backslash-escaped forms first so we don't leave stray backslashes
    _ALL_QUOTE_CHARS_FOR_ESCAPED = set(_APOSTROPHE_NORMALIZE.keys()) \
        | _SINGLE_QUOTES_DELETE | _DOUBLE_QUOTES_DELETE | {"'"}
    _escaped_class = ''.join(map(re.escape, _ALL_QUOTE_CHARS_FOR_ESCAPED))
    _ESCAPED_QUOTE_RE = re.compile(rf'\\([{_escaped_class}])')
    text = _ESCAPED_QUOTE_RE.sub(r'\1', text)

    # 1) Normalize apostrophes to ASCII "'"
    if _APOSTROPHE_NORMALIZE:
        text = text.translate(str.maketrans(_APOSTROPHE_NORMALIZE))

    # 2) Delete all double quotes and the unwanted single-quote lookalikes
    _DELETE_TABLE = {ord(ch): None for ch in (_SINGLE_QUOTES_DELETE | _DOUBLE_QUOTES_DELETE)}
    text = text.translate(_DELETE_TABLE)

    # 3) Ensure we keep ASCII apostrophe itself as-is (do not remove "'")
    #    (No action needed; we intentionally do not include "'" in delete table)

    # (Optionally collapse spaces again if deletions created doubles)
    text = re.sub(r' {2,}', ' ', text).strip()

    return text

#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
Tree class for hierarchical spans
"""

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
import numpy as np
from typing import List, Optional, Union, Any, Tuple, Iterator, Dict

# Use a math font
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 10


def range_tree_to_counts(node: Dict[str, Any]) -> Union[int, List]:
    """
    Given a tree of {"start":i, "end":j, "children":[...]} returns
    either an int (for leaves) or a list of ints/lists for internal nodes.
    """
    children = node.get("children", [])
    if not children:
        # leaf: exactly the number of tokens in this span
        return node["end"] - node["start"]
    # internal: convert each child
    return [range_tree_to_counts(child) for child in children]



class Node:
    def __init__(self, count: int, children: Optional[List['Node']] = None, token: Optional[str] = None):
        self.value = count
        self.children = children or []
        self.token = token


def build_token_tree(partition: Union[int, List[Any]], token_iter: Iterator[int], encoding) -> Node:
    """
    Recursively builds a tree of Nodes from a partition and a token iterator.
    Groups `count` token IDs, decodes them together to preserve full text.
    """
    if isinstance(partition, int):
        count = partition
        # Collect token IDs
        ids = [next(token_iter, None) for _ in range(count)]
        ids = [tid for tid in ids if tid is not None]
        text = encoding.decode(ids)
        return Node(count, token=text)
    else:
        children = [build_token_tree(p, token_iter, encoding) for p in partition]
        total = sum(child.value for child in children)
        return Node(total, children=children)


def traverse_tree(node: Node) -> Tuple[List[Tuple[int, Dict[str, Any]]],
                                       List[Tuple[int, int]]]:
    nodes: List[Tuple[int, Dict[str, Any]]] = []
    edges: List[Tuple[int, int]] = []

    def _traverse(n: Optional[Node], parent: Optional[Node] = None):
        if n is None:
            return

        node_id = id(n)

        if n.children:                                 # ── internal node
            nodes.append((node_id, {'value': n.value}))
            if parent:
                edges.append((id(parent), node_id))
            for child in n.children:
                _traverse(child, n)

        else:                                          # ── leaf node
            if n.token is None:
                return

            # ── keep pure whitespace; otherwise trim ───────────────────
            if n.token.strip() == "":
                label = n.token          # exactly as given (may look blank)
            else:
                label = n.token.strip()  # drop leading/trailing spaces

            nodes.append((node_id, {'value': label}))
            if parent:
                edges.append((id(parent), node_id))

    _traverse(node)
    return nodes, edges



def get_positions(node: Node, y: float = 0, positions: Optional[Dict[int, Tuple[float, float]]] = None,
                  leaf_x: Optional[List[int]] = None) -> Dict[int, Tuple[float, float]]:
    if positions is None:
        positions = {}
    if leaf_x is None:
        leaf_x = [0]
    if node.children:
        for child in node.children:
            get_positions(child, y - 1, positions, leaf_x)
        xs = [positions[id(c)][0] for c in node.children if id(c) in positions]
        positions[id(node)] = (sum(xs) / len(xs), y)
    else:
        positions[id(node)] = (leaf_x[0], y)
        leaf_x[0] += 1
    return positions

def plot_tree_vertical(nodes: List[Tuple[int, Dict[str, Any]]],
                       edges: List[Tuple[int, int]],
                       pos: Dict[int, Tuple[float, float]],
                       figsize: Tuple[float, float] = (8, 12),
                       show_text: bool = True,
                       out_file: str = "token_tree_vertical.png",
                       showfig=True,
                       horizontal: bool = False
                    ):
    """
    Plot the token tree **top-to-bottom** instead of left-to-right.

    Rotation trick:
        new_x = old_y
        new_y = -old_x        # minus sign keeps root at the top
    """
    # ── rotate the layout ───────────────────────────────────────────────
    if horizontal:
        vpos = {nid: (x, y) for nid, (x, y) in pos.items()}
    else:
        vpos = {nid: (y, -x) for nid, (x, y) in pos.items()}

    # ── build graph ─────────────────────────────────────────────────────
    plt.figure(figsize=figsize)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    internal = [nid for nid, _ in nodes if G.out_degree(nid) > 0]
    leaf     = [nid for nid, _ in nodes if G.out_degree(nid) == 0]

    my_blue  = '#42c5f5'
    my_green = '#42f5b3'
    color_map = {nid: (my_blue if nid in leaf else my_green) for nid, _ in nodes}

    # nodes & edges
    nx.draw_networkx_nodes(G, vpos, nodelist=internal,
                           node_size=800,
                           node_color=[color_map[n] for n in internal],
                           edgecolors='black')
    nx.draw_networkx_nodes(G, vpos, nodelist=leaf,
                           node_size=1200,
                           node_color=[color_map[n] for n in leaf],
                           edgecolors='black')
    nx.draw_networkx_edges(G, vpos, arrows=False)

    # labels
    labels_int = {nid: data['value'] for nid, data in nodes if nid in internal}
    nx.draw_networkx_labels(G, vpos, labels_int, font_size=20,
                            font_family='STIXGeneral')

    if show_text:
        labels_leaf = {nid: data['value']
                       for nid, data in nodes if nid in leaf and data['value']}
        nx.draw_networkx_labels(G, vpos, labels_leaf, font_size=14,
                                font_family='STIXGeneral')

    # tight limits
    xs = [x for x, _ in vpos.values()]
    ys = [y for _, y in vpos.values()]
    plt.xlim(min(xs) - 0.5, max(xs) + 0.5)
    plt.ylim(min(ys) - 0.5, max(ys) + 0.5)

    plt.axis('off')
    plt.savefig(out_file, bbox_inches='tight', dpi=100)

    if showfig:
        plt.show()
    else:
        plt.close()


def plot_chunks_with_text_vertical(
    partition: Union[int, List[Any]],
    text: str,
    *,
    figsize: Tuple[float, float] = (30, 4),
    show_text: bool = True,
    model: str = None, #"deepseek-ai/DeepSeek-V3",

    showfig: bool = True,
    out_file: Optional[str] = None,
    horizontal: bool = False
) -> Node:

    encoding = select_tokenizer(model)
    token_ids  = encoding.encode(text)
    token_iter = iter(token_ids)

    root = build_token_tree(partition, token_iter, encoding)
    nodes, edges = traverse_tree(root)
    pos          = get_positions(root)

    plot_tree_vertical(
        nodes,
        edges,
        pos,
        figsize=figsize,
        show_text=show_text,
        out_file=out_file or "token_tree_vertical.png",
        showfig=showfig,
        horizontal=horizontal
    )

    return root


#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
High level chunker functions.
"""




from typing import Optional, List, Tuple, Dict, Any

def chunk_main(
    input_text: str,
    *,
    model: str = None,  #"deepseek-ai/DeepSeek-V3",
    client: Optional[object] = None,     # e.g. Together(...)
    api_key: Optional[str] = None,       # ignored if client is given
    multithread: bool = False,
    debug: bool = False,
    enable_fallback: bool = True,
    k: int = 4,
) -> Tuple[Dict[str, Any], List[int]]:    # ← use typing.List / Tuple
    """
    Run the hierarchical chunker on `input_text`.

    Parameters
    ----------
    model        : model name (ignored if `client` hard-codes its own default)
    client       : an instantiated client with `.chat.completions.create`
    api_key      : only used to build the default OpenAI client
    multithread  : enable nested thread-pool segmentation
    k            : max segments per LLM call

    Returns
    -------
    (result, lengths)
        result   : full dict returned by `chunk_text`
        lengths  : list[int] of span lengths where deterministic fallback fired
    """

    chunker = HierarchicalChunker(
        api_key     = api_key,
        model       = model,
        client      = client,
        k           = k,
        multithread = multithread,
        debug       = debug,
        enable_fallback = enable_fallback
    )

    result  = chunker.chunk_text(input_text)

    if enable_fallback == True:
        lengths = chunker.fallback_lengths
        print(f"Fallback lengths: {lengths}")

    if enable_fallback == False:
        lengths = chunker.failed_unsplit_counts
        print(f"Failed unsplit counts: {lengths}")

    return result, lengths, chunker


def main(
    input_text: str,
    *,
    model: str = None,  # "deepseek-ai/DeepSeek-V3",
    client: Optional[object] = None,
    api_key: Optional[str] = None,
    multithread: bool = False,
    debug: bool = False,
    enable_fallback: bool = True,
    k: int = 4,
    figsize: Tuple[int, int] = (10, 40),
    show_text: bool = True,
    showfig: bool = True,
    horizontal: bool = False
) -> Dict[str, Any]:
    """
    Clean text, chunk it, plot the vertical tree, and return a dict with:
      - 'original_text': the cleaned input
      - 'tokens': token list
      - 'token_indices': character indices for each token
      - 'hierarchy': raw hierarchy tree from the chunker
      - 'fallback_lengths': list of fallback span-lengths
      - 'span_tree': the minimal span-tree dict
      - 'partition': token-count partition used for plotting
    """
    # 1. clean + chunk
    cleaned = clean_input_text(input_text)
    chunk_result, fallback_lengths, chunker = chunk_main(
        cleaned,
        model       = model,
        client      = client,
        api_key     = api_key,
        multithread = multithread,
        debug       = debug,
        k           = k,
        enable_fallback = enable_fallback
    )

    # 2. build the span-tree and partition
    span_tree  = hierarchy_to_span_tree(chunk_result["hierarchy"])
    partition  = range_tree_to_counts(span_tree)
    fallback_segments = chunker.fallback_segments
    fallback_texts = chunker.fallback_texts

    root = plot_chunks_with_text_vertical(partition, cleaned,
                                   figsize=figsize,
                                   show_text=show_text,
                                   model=model,
                                   showfig=showfig,
                                   horizontal=horizontal)

    # 4. return everything in a flat dict
    return {
        "original_text":    cleaned,
        "tokens":           chunk_result["tokens"],
        "hierarchy":        chunk_result["hierarchy"],
        "fallback_lengths": fallback_lengths,
        "fallback_segments": fallback_segments,  # ← list of segments where fallback was used
        "fallback_texts":   fallback_texts,     # ← list of texts where fallback was used
        "span_tree":        span_tree,
        "partition":        partition,
    }


#========================================================================================================
#========================================================================================================
#========================================================================================================

"""
Statistics of Empirical Trees
"""


from collections import deque, defaultdict
from typing import List, Any
import tiktoken
import os

def nodes_per_level(root: Any) -> List[int]:
    """
    Traverse the node tree and count how many nodes appear at each depth level.
    Returns a list where index `i` is the number of nodes at level `i`.
    """
    counts: defaultdict[int, int] = defaultdict(int)
    queue = deque([(root, 0)])
    while queue:
        node, level = queue.popleft()
        counts[level] += 1
        for child in getattr(node, 'children', []):
            queue.append((child, level + 1))
    max_level = max(counts.keys(), default=-1)
    return [counts[i] for i in range(max_level + 1)]


def branching_ratios_per_level(root: Any) -> List[List[int]]:
    """
    Traverse the node tree and collect each node's number of children, grouped by depth level.
    Returns a list of lists: element `i` is a list of branching counts at level `i`.
    """
    ratios: defaultdict[int, List[int]] = defaultdict(list)
    queue = deque([(root, 0)])
    while queue:
        node, level = queue.popleft()
        child_count = len(getattr(node, 'children', []))
        ratios[level].append(child_count)
        for child in getattr(node, 'children', []):
            queue.append((child, level + 1))
    max_level = max(ratios.keys(), default=-1)
    return [ratios[i] for i in range(max_level + 1)]


def chunk_sizes_per_level(root: Any) -> List[List[int]]:
    """
    Traverse the node tree and collect each node's `value`, grouped by depth level.
    Returns a list of lists: element `i` is a list of node values at level `i`.
    """
    values: defaultdict[int, List[int]] = defaultdict(list)
    queue = deque([(root, 0)])          # (node, depth)

    while queue:
        node, level = queue.popleft()

        # Record this node's value at its depth level
        values[level].append(getattr(node, 'value', None))

        # Enqueue children (if any) one level deeper
        for child in getattr(node, 'children', []):
            queue.append((child, level + 1))

    # Convert sparse defaultdict → dense list with contiguous levels [0 … max_level]
    max_level = max(values.keys(), default=-1)
    return [values[i] for i in range(max_level + 1)]

def chunk_sizes_per_tree(N,root: Any) -> List[List[int]]:
    """
    Return a flat list of `value` for every node in the tree except the root.
    Traverses breadth-first. If a node lacks `value`, None is inserted
    (matching `chunk_sizes_per_level` behavior).
    """
    sizes: List[Optional[int]] = []
    queue = deque(getattr(root, 'children', []))  # start from level 1
    # queue = deque([root])  # start with the root (level 0)

    while queue:
        node = queue.popleft()
        sizes.append(getattr(node, 'value', None)/N)
        for child in getattr(node, 'children', []):
            queue.append(child)

    return sizes

def dist_per_level(chunk_sizes: List[List[int]]) -> List[List[int]]:
    dist_per_level = []
    N = chunk_sizes[0][0]
    levels = len(chunk_sizes)
    for l in range(levels):
        dist, bin_edges = np.histogram(chunk_sizes[l], bins=np.arange(1,N+1,1), density=True)
        dist_per_level.append(dist)

    return dist_per_level, bin_edges

def dist_per_tree(N, chunk_sizes_tree: List[List[int]]) -> List[List[int]]:
    # dist, bin_edges = np.histogram(chunk_sizes_tree, bins=np.arange(1,N+1,1), density=True)
    dist, bin_edges = np.histogram(chunk_sizes_tree, bins=np.logspace(-3,0,16), density=True)

    # replace 0s in dist with NaNs
    dist = np.where(dist == 0, np.nan, dist)*N

    return dist, bin_edges

def retrieve_partition(partition, D):
    # Helper function to compute the sum of integers in the subtree rooted at 'node'
    def subtree_sum(node):
        if isinstance(node, int):
            return node
        elif isinstance(node, list):
            return sum(subtree_sum(child) for child in node)
        else:
            return 0  # Should not reach here

    result = []

    # Depth-first traversal function
    def dfs(node, depth):
        if isinstance(node, int) or depth == D:
            result.append(subtree_sum(node))
            return
        elif isinstance(node, list):
            for child in node:
                dfs(child, depth + 1)
        else:
            pass  # Should not reach here

    dfs(partition, 0)
    return result

from typing import Union, List

def leaves_up_to_level(partition: Union[int, List], L: int) -> int:
    """
    Count the cumulative number of leaves (ints) with depth <= L.
    Depth convention: root is depth 0; elements of a list are at parent_depth + 1.
    """
    if L < 0:
        return 0

    def dfs(node, depth) -> int:
        if isinstance(node, int):
            return 1 if depth <= L else 0
        if depth >= L:  # any children would be deeper than L
            return 0
        if isinstance(node, list):
            return sum(dfs(child, depth + 1) for child in node)
        return 0

    return dfs(partition, 0)

# partition is a nested structure of lists and ints
Nested = Union[int, List["Nested"]]

def max_depth_list(partition: Nested) -> int:
    """
    Maximum depth with root at depth 0.
    - int → depth 0
    - list → 1 + max(child depths); empty list → depth 0
    """
    if isinstance(partition, int):
        return 0
    if isinstance(partition, list) and partition:
        return 1 + max(max_depth_list(child) for child in partition)
    return 0

def retrieved_counts_by_level_list(partition: Nested) -> List[int]:
    """
    For D = 0..max_depth(partition), call retrieve_partition(partition, D)
    and record how many items are retrieved.
    Note: counts reflect retrieve_partition's semantics (ints are retrieved
    immediately, even if depth < D).
    """
    dmax = max_depth_list(partition)
    return [len(retrieve_partition(partition, D)) for D in range(dmax + 1)]

def leaves_counts_by_level_list(partition: Nested) -> List[int]:
    dmax = max_depth_list(partition)
    return [leaves_up_to_level(partition, D) for D in range(dmax + 1)]

def tree_analysis(
    results_dir: str,
    model: str,
    # *,
    # encoding_name: str = "cl100k_base"
) -> Dict[str, Dict[str, Any]]:
    """
    For every `*_result.json` in `results_dir`, rebuild the Node tree,
    compute its statistics, and return an *ordered* dict mapping:

        "<story_name>" → {
            "root": <Node>,
            "num_tokens": int,
            "chunk_sizes": List[List[int]],
            "distribution": {"dist": List[int], "bin_edges": List[float]},
            "nodes_per_level": List[int],
            "branching_ratios": List[List[int]],
            "partition": <original partition data>
        }

    The dict is ordered by ascending `num_tokens`.
    """
    # prepare tokenizer
    encoding = select_tokenizer(model)

    # encoding = tiktoken.get_encoding(encoding_name)

    # first collect everything unsorted
    all_stats: Dict[str, Dict[str, Any]] = {}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith("_result.json"):
            continue

        story_name = fname[:-len("_result.json")]
        path = os.path.join(results_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)

        text      = result["original_text"]
        partition = result["partition"]
        tokens    = result["tokens"]
        N = len(tokens)
        # rebuild the tree
        token_ids   = encoding.encode(text)
        root        = build_token_tree(partition, iter(token_ids), encoding)

        # compute stats
        chunk_sizes_level  = chunk_sizes_per_level(root)
        dist_level, bins_level   = dist_per_level(chunk_sizes_level)
        chunk_sizes_tree   = chunk_sizes_per_tree(N, root)
        dist_tree, bins_tree     = dist_per_tree(N, chunk_sizes_tree)
        Cs           = nodes_per_level(root)
        branches     = branching_ratios_per_level(root)
        retrieved_counts = retrieved_counts_by_level_list(partition)
        cumulative_leaves = leaves_counts_by_level_list(partition)

        all_stats[story_name] = {
            "num_tokens": N,
            "chunk_sizes_level": chunk_sizes_level,
            "chunk_sizes_tree": chunk_sizes_tree,
            "distribution_level": {
                "dist": dist_level,
                "bin_edges": bins_level
            },
            "distribution_tree": {
                "dist": dist_tree,
                "bin_edges": bins_tree
            },
            "nodes_per_level": Cs,
            "branching_ratios": branches,
            "partition": partition,
            "retrieved_counts": retrieved_counts,
            "cumulative_leaves": cumulative_leaves
        }

    # now sort by num_tokens and build a new dict in that order
    sorted_stats = dict(
        sorted(
            all_stats.items(),
            key=lambda item: item[1]["num_tokens"]
        )
    )

    return sorted_stats

def collect_level_data(
    stats: Dict[str, Dict[str, Any]],
    level: int
) -> Tuple[List[Any], List[Any], List[int]]:
    """
    For each story in `stats`, if it has a distribution and bin_edges at `level`,
    collect:
      - dist[level]
      - bin_edges[level]
      - num_tokens

    Returns three parallel lists: (dists, bin_edges_list, Ns)
    """
    dists: List[Any]      = []
    bin_edges_list: List[Any] = []
    Ns: List[int]         = []
    s_samples: List[List[float]] = []


    for story_name, info in stats.items():
        dist_arr     = info["distribution_level"]["dist"]
        bin_edges_arr= info["distribution_level"]["bin_edges"]
        N            = info["num_tokens"]
        chunk_sizes   = info["chunk_sizes_level"]            # raw n values


        # only include if this level exists in both lists
        if level < len(dist_arr) and level < len(bin_edges_arr):
            dists.append(dist_arr[level-1]) # level-1 for 0-indexing
            bin_edges_list.append(bin_edges_arr)
            Ns.append(N)

            # --- rescale n → s = n/N ---
            s_vals = [n / N for n in chunk_sizes[level-1]]
            s_samples.append(s_vals)

    return dists, bin_edges_list, Ns, s_samples
