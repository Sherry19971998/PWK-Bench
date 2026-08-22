"""
Real LLM agent adapters (Plan A). Each presents the revealed (bucketized)
evidence for a variant and asks the model which not-yet-queried channel to
acquire next, step by step, until budget K. Output: an (n, K) order array.

Model IDs come from configs/models.yaml (resolved at run time), never
hard-coded here. API keys are read from the environment; nothing is logged.
These adapters are optional — install the matching extra:
    pip install -e ".[anthropic]"   # AnthropicAgent
    pip install -e ".[openai]"      # OpenAIAgent
    pip install -e ".[gemini]"      # GeminiAgent (GEMINI_API_KEY)
    pip install -e ".[hf]"          # HFLocalAgent
"""
from __future__ import annotations
import os
import numpy as np
from ..domains.base import Cohort
from ..spec import GLOBAL_SEED
from .base import Agent


class RefusalError(Exception):
    """The model deterministically refused this prompt. NOT retryable (the same
    prompt refuses every time) -- it is treated as a parse failure (counted, run
    continues), never as a transient error that burns the retry budget and
    aborts the paid sweep."""


class DailyQuotaExhausted(Exception):
    """The provider's DAILY free-tier request cap is exhausted (distinct from a
    transient per-minute rate limit): retrying within the same process cannot
    help (the server-suggested delay is seconds, but the cap resets on a
    ~24h clock), so this is raised immediately, with NO retry burned, instead
    of looping `retries` times against a wall that will not move today. Callers
    that persist a disk-backed cache (`cache_path`) can catch this, stop
    cleanly, and resume tomorrow picking up only the NOT-yet-cached prompts."""


def _is_daily_quota_error(exc) -> bool:
    """True iff `exc` is a provider DAILY (not per-minute) quota/rate-limit
    error. Detected from Google's Gemini free-tier error text (quotaId
    'GenerateRequestsPerDayPerProjectPerModel-FreeTier'): matched loosely on
    "PerDay" + "FreeTier" so it survives minor message wording changes without
    also catching an ordinary per-minute 429 (which IS worth retrying)."""
    s = str(exc)
    return "PerDay" in s and "FreeTier" in s

_SYS = (
    "You are triaging clinical genetic evidence under a query budget. "
    "At each step choose the single most informative NOT-yet-acquired evidence "
    "category to reveal next. Respond with ONLY the category name."
)

# Output-token budget for one elicitation step. The REPLY is a single category
# name (~3 tokens), but on every current reasoning model this budget covers
# THINKING + TEXT, so a tight value returns finish_reason='length' with an EMPTY
# text block -- which this module correctly counts as a parse failure and then
# silently falls back to a fixed order, i.e. it manufactures exactly the
# relevance-greedy behaviour the benchmark is trying to measure.
# Measured on gpt-5.5-2026-04-23 with the real step prompts: step 2 spent all
# 256 tokens on reasoning and returned '' (38% parse-failure rate across a
# 40-variant run); at 2000 the same step returns 'PM2' after 332 tokens.
# Keep this generous -- the marginal cost is reasoning tokens the model would
# have spent anyway, whereas a truncated step is a silently corrupted datapoint.
_MAX_TOKENS = 2000


def _suggested_retry_delay(exc) -> float | None:
    """Seconds the server asked us to wait, parsed out of a rate-limit error.

    A quota-limited endpoint returns 429 with an explicit delay (the Gemini free
    tier is 5 req/min and replies "Please retry in 36.6s" / retryDelay: '36s').
    The generic exponential ramp below caps far short of that, so without
    honouring the server's number the run aborts on the first quota hit.
    """
    import re
    s = str(exc)
    for pat in (r"retryDelay['\"]?:\s*['\"](\d+(?:\.\d+)?)s",
                r"retry in (\d+(?:\.\d+)?)s"):
        m = re.search(pat, s)
        if m:
            return min(float(m.group(1)) + 1.0, 120.0)   # +1s guard, hard cap
    return None


def _load_disk_cache(path: str) -> dict:
    """Load a persisted prompt->completion cache written by `_save_disk_cache`.
    Missing/unreadable file -> empty cache (first run of the day), never an
    error -- a corrupt or partial cache should degrade to "re-ask", not crash
    the resumable driver."""
    import json, os
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_disk_cache(path: str, cache: dict) -> None:
    """Persist the full prompt->completion cache, atomically (write to a temp
    file in the same directory then os.replace), so a process killed mid-write
    (e.g. hitting the daily quota wall) never leaves a half-written, corrupt
    cache file that would lose yesterday's progress."""
    import json, os
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


# Wordings for the STOP action. The observed stop points never include 1 (the
# agent always takes a second category before it will stop), which is either a
# real judgement -- one category is never enough to classify -- or an artifact of
# how the option is phrased. These variants separate the two: `neutral` strips
# the cost framing and the word "sufficient", and `explicit` additionally states
# outright that stopping after a single category is allowed. If the stop-point
# distribution is unchanged across all three, the behaviour is a property of the
# agent rather than of the prompt.
_STOP_WORDINGS = {
    "default": [
        "Each further category costs one query. If the evidence already "
        "acquired is sufficient to classify this variant, reply STOP instead "
        "of naming a category.",
        "Reply with either STOP or exactly one not-yet-acquired category name.",
    ],
    "neutral": [
        "STOP is also an available reply.",
        "Reply with either STOP or exactly one not-yet-acquired category name.",
    ],
    "explicit": [
        "You may reply STOP at any point to stop acquiring, including now, "
        "after only a single category. Stopping early is a legitimate choice, "
        "not a failure to answer.",
        "Reply with either STOP or exactly one not-yet-acquired category name.",
    ],
}


def _prompt(channels, revealed, context=None, allow_stop=False,
            stop_wording="default"):
    """Build the step prompt. If `context` (a dict with gene/consequence/domain)
    is given, the variant is described so the agent CAN plan instance-adaptively
    (ablation route A); without it, every variant's prompt is identical at k=1
    and the agent cannot adapt (the P1(b) confound).

    `allow_stop` offers an explicit STOP action alongside the categories, turning
    "when to stop acquiring" into a decision the agent makes rather than a budget
    the harness imposes. It is only ever passed True once at least one category
    has been acquired -- stopping at zero evidence would leave nothing to make a
    call on, so the choice would be degenerate rather than informative.
    """
    lines = []
    if context:
        lines.append(
            f"Variant: gene {context.get('gene','?')}, "
            f"consequence {context.get('consequence','?')}, "
            f"disease domain {context.get('domain','?')}.")
    lines.append("Evidence categories:")
    for c in channels:
        state = revealed.get(c, "not yet acquired")
        lines.append(f"- {c}: {state}")
    if allow_stop:
        lines.extend(_STOP_WORDINGS[stop_wording])
    else:
        lines.append("Which category do you acquire next? Name exactly one "
                     "not-yet-acquired category.")
    return "\n".join(lines)


class _StepwiseLLMAgent(Agent):
    """Shared step-wise elicitation loop; subclasses implement _complete().

    `context=True` describes each variant (gene / consequence / disease domain)
    in the prompt so the agent can plan instance-adaptively; `context=False`
    (default) reproduces the no-context confound. Run the same model_id with
    both to get the P1(b) ablation.
    """
    def __init__(self, model_id: str, name: str, context: bool = False,
                 retries: int = 6, max_workers: int = 1, seed: int = GLOBAL_SEED,
                 cache_path: str | None = None, randomize_channel_order: bool = False,
                 allow_stop: bool = False, stop_wording: str = "default"):
        self.model_id = model_id
        self.name = name
        self.context = context
        # DIAGNOSTIC, default OFF (does not affect any existing result): when
        # True, `order()` lists the channels in a fresh random permutation for
        # EACH variant (seeded off self.seed + variant index, so it is still
        # reproducible) instead of the fixed cohort.domain.channels order every
        # step normally uses. This isolates prompt-POSITION effects (does the
        # model tend to query whichever channel is listed last, regardless of
        # which channel that is?) from CONTENT effects (does it deprioritize a
        # specific channel like PM1 regardless of where it's listed?) -- the
        # two are confounded under the fixed listing order every other run
        # uses, since the domain's declared relevance order also determines
        # what the model sees in the prompt. See self.channel_perms.
        self.randomize_channel_order = randomize_channel_order
        self.channel_perms = {}         # variant index -> permutation used (only if randomized)
        # EXPLICIT STOP ACTION, default OFF (so every previously reported number
        # is reproduced bit-for-bit by the default path). When True the agent may
        # reply STOP at any step after its first acquisition, and `order()` also
        # exposes `self.stop_k` -- the per-variant number of channels actually
        # acquired. The returned (n, K) order array is still FULL (the untouched
        # channels are appended in the order they would have been offered), so
        # every existing metric that slices order[:, :k] keeps working unchanged
        # and only the new stop-aware endpoint reads stop_k.
        self.allow_stop = allow_stop
        if stop_wording not in _STOP_WORDINGS:
            raise ValueError(f"stop_wording must be one of "
                             f"{sorted(_STOP_WORDINGS)}, got {stop_wording!r}")
        self.stop_wording = stop_wording
        self.stop_k = None              # (n,) int, filled by order() when allow_stop
        self.stop_requests = 0          # steps where the model chose STOP
        # Sampling seed, pinned into every request the vendor will accept it on.
        # This does NOT buy bit-reproducibility -- see OpenAIAgent for why
        # (reasoning models refuse temperature=0, and `seed` is best-effort).
        # Repeated runs are still required to bound the residual variance.
        self.seed = seed
        self.parse_failures = 0        # steps where the model named no valid channel
        self.forced_parse_failures = 0  # ...of which had only ONE channel left, so
                                        # the fallback was the only admissible pick
                                        # and the order is unchanged (see _order_one)
        self.forced_steps = 0          # steps with a single remaining channel
        self.total_steps = 0
        self.retries = retries
        # cache_path (optional): persists the prompt-hash -> completion cache to
        # disk so a run that dies mid-sweep (e.g. a daily free-tier quota wall)
        # can be re-invoked tomorrow and only pay for the NOT-yet-cached prompts,
        # instead of re-asking everything from an empty in-memory cache. Loaded
        # once at construction; every NEW completion is written straight back
        # (see _complete_cached), so progress survives even a hard kill.
        self.cache_path = cache_path
        self._cache = _load_disk_cache(cache_path) if cache_path else {}
        self.cache_hits = 0
        # Per-variant calls are independent, so `order()` can fan them out across
        # threads (each variant is a chain of blocking API calls -> threads
        # overlap the network waits). max_workers=1 (default) keeps the run
        # fully serial and deterministic for tests; set >1 to parallelise the
        # paid real-model sweep. The shared cache and the parse/step counters
        # are guarded by _lock so concurrent workers can't corrupt them.
        import threading
        self.max_workers = max(1, int(max_workers))
        self._lock = threading.Lock()

    def _complete(self, system: str, user: str) -> str:
        raise NotImplementedError

    def _sampling_profile(self) -> str:
        """String identifying the sampler settings this adapter actually sends.

        Two jobs. It is mixed into the prompt cache key so a sampling change
        cannot be silently absorbed by a warm cache (see _complete_cached), and
        it is the one place that states each vendor's ACTUAL regime, which is
        not obvious from the call sites.

        POLICY (revised 2026-08-02): every vendor runs at its DEFAULT sampler,
        with only a best-effort `seed` where the endpoint takes one. The earlier
        policy was "pin the sampler where allowed", which sounds stricter but
        produced rows that cannot be compared: Gemini and Anthropic accept
        temperature=0 and gpt-5.x rejects it outright ("Unsupported value:
        'temperature' does not support 0 with this model. Only the default (1)
        value is supported" -- measured 2026-08-02), so the OpenAI rows carried
        real sampling noise (two gpt-5.4 runs moved rho_vs_oracle by 0.056,
        the same magnitude as effects under test) while the Gemini rows carried
        none. Freezing a sampler does not make a result robust, it only hides
        the run-to-run spread -- so the pinned rows were not more trustworthy,
        just less legible. Repeated runs, not a frozen sampler, are how this
        repo bounds spread.

        Subclasses that send something different MUST override this, or their
        cached completions will be indistinguishable from another regime's.
        """
        return f"default+seed={self.seed}"

    def _complete_cached(self, system: str, user: str) -> str:
        """Cache + retry wrapper around _complete. Identical prompts (e.g. the
        491 identical k=1 no-context prompts) hit the cache, cutting cost and
        giving crude resumability -- persisted to disk when `cache_path` is
        set, so resumability survives a process restart, not just a run.
        Transient API errors are retried with exponential backoff; a DAILY
        quota wall (see `DailyQuotaExhausted`) is raised immediately instead,
        since no in-process backoff can wait out a ~24h reset."""
        import hashlib, time
        # The sampling profile is part of the key. It was NOT, and that was a
        # silent-wrong-answer trap: changing temperature/seed and re-running
        # with an existing cache_path returned every completion from the old
        # regime, so the run "succeeded", produced identical numbers, and would
        # have supported the conclusion "the sampler makes no difference" --
        # having made zero API calls. Keying on the profile makes such a change
        # a cache MISS, which is the honest outcome: it costs a re-run instead
        # of silently reporting stale results.
        key = hashlib.sha256(
            f"{self.model_id}\x00{self._sampling_profile()}\x00{system}\x00{user}"
            .encode()).hexdigest()
        with self._lock:
            if key in self._cache:
                self.cache_hits += 1
                return self._cache[key]
        last = None
        for attempt in range(max(1, self.retries)):       # always attempt >= once
            try:
                out = self._complete(system, user)   # network call OUTSIDE the lock
                with self._lock:
                    self._cache[key] = out
                    if self.cache_path:
                        _save_disk_cache(self.cache_path, self._cache)
                return out
            except RefusalError:
                # Deterministic refusal: retrying the identical prompt just
                # refuses again and would abort the whole sweep. Return "" so the
                # parse layer counts it as a parse_failure and the run continues.
                with self._lock:
                    self._cache[key] = ""
                    if self.cache_path:
                        _save_disk_cache(self.cache_path, self._cache)
                return ""
            except Exception as e:                        # noqa: BLE001 (transient API/network)
                if _is_daily_quota_error(e):
                    # Not retryable within this process: the cap resets on a
                    # ~24h clock, so burning the retry budget here would just
                    # sleep through a handful of seconds and hit the same wall.
                    # Whatever is already in self._cache (and on disk, if
                    # cache_path is set) is real progress -- raise now so the
                    # caller can stop cleanly and resume tomorrow.
                    raise DailyQuotaExhausted(str(e)) from e
                last = e
                # Honour a server-supplied rate-limit delay when present; the
                # plain 2**attempt ramp capped at 8s guaranteed an abort against
                # any quota-limited endpoint that asks for ~36s.
                delay = _suggested_retry_delay(e)
                time.sleep(delay if delay is not None else min(5 * 2 ** attempt, 60))
        raise last

    def _order_one(self, channels, reveal_row, ctx=None) -> tuple[list[int], int]:
        """Returns (full order over all channels, stop_k).

        `stop_k` is how many channels the agent actually chose to acquire: it
        equals len(channels) unless allow_stop is on AND the model replied STOP,
        in which case the order is still completed with the unacquired channels
        so the (n, K) array downstream metrics expect stays well-formed.
        """
        remaining = list(range(len(channels)))
        order = []
        revealed = {}
        while remaining:
            # Pass the FULL channel list with each channel's revealed state, so
            # the model actually conditions on evidence acquired so far (already
            # acquired channels carry their bucket; the rest read "not yet
            # acquired"). Passing only the remaining names would hide the
            # accumulated evidence and make every step's prompt collapse to a
            # fixed sequence -> a relevance-greedy artifact, not model behaviour.
            # STOP is only offered once something has been acquired (see _prompt).
            offer_stop = self.allow_stop and bool(order)
            txt = self._complete_cached(
                _SYS, _prompt(list(channels), revealed,
                              context=ctx if self.context else None,
                              allow_stop=offer_stop,
                              stop_wording=self.stop_wording))
            with self._lock:
                self.total_steps += 1
                if len(remaining) == 1:
                    self.forced_steps += 1
            # Pick the not-yet-acquired channel the model names FIRST in its
            # reply (earliest text position), not the first in our own channel
            # order. NOTE: this is a plain earliest-mention heuristic with NO
            # negation handling -- "not PM2, take PM1" still selects PM2 because
            # PM2 appears first. The system prompt asks the model to reply with
            # ONLY the channel name to keep this reliable; a garbled/negated
            # reply that picks the wrong channel is a modelling limitation, not a
            # parse we can robustly recover, and shows up as agent behaviour.
            low = (txt or "").lower()
            best_pos, pick = None, None
            for i in remaining:
                p = low.find(channels[i].lower())
                if p >= 0 and (best_pos is None or p < best_pos):
                    best_pos, pick = p, i
            if offer_stop:
                # Same earliest-mention rule the channel parse uses, so STOP and
                # a category name are adjudicated on one consistent criterion:
                # whichever the model names FIRST wins. Deciding STOP by mere
                # presence instead would make "STOP only after PM1" read as an
                # immediate stop, and would not compose with the existing
                # earliest-mention semantics for channels.
                sp = low.find("stop")
                if sp >= 0 and (best_pos is None or sp < best_pos):
                    with self._lock:
                        self.stop_requests += 1
                    # Record the stop, then fall through to complete the row:
                    # the remaining channels still fill out the (n, K) order so
                    # fixed-budget metrics stay computable on the same array.
                    return order + [i for i in remaining], len(order)
            if pick is None:
                # Model named no valid channel (empty/blocked/garbled reply).
                # We fall back to the first remaining channel to keep the run
                # going, but COUNT it: a high parse_failures rate means the
                # results are a fallback artifact, not model behaviour, and the
                # runner warns loudly. (Silent fallback would masquerade as a
                # perfect relevance-greedy agent — exactly the paper's claim.)
                #
                # Split FORCED from CONSEQUENTIAL failures. On the LAST step only
                # one channel remains, so the fallback picks the only admissible
                # answer and the resulting order is identical to a parsed reply --
                # the failure cannot bias anything. Observed on
                # gpt-5.5-2026-04-23: at the final step it re-names an
                # already-acquired category ("PVS1" when only PP3 is left), which
                # inflated the raw rate to 7.2% on the 491-variant cohort while
                # changing exactly zero acquisition decisions. Only failures with
                # >1 channel still remaining actually fabricate a choice, so the
                # runner must gate its reliability warning on those.
                with self._lock:
                    self.parse_failures += 1
                    if len(remaining) == 1:
                        self.forced_parse_failures += 1
                pick = remaining[0]
            order.append(pick)
            bkt = int(reveal_row[pick])
            # reveal() returns -1 for an UNDEFINED entry (channel not applicable /
            # no data for this variant). Render it as an explicit no-data state,
            # not the literal "bucket -1", so the model reads "acquired but no
            # evidence" rather than an out-of-range bucket index.
            revealed[channels[pick]] = (
                "acquired: no data available for this variant" if bkt < 0
                else f"bucket {bkt}")
            remaining.remove(pick)
        return order, len(order)

    @staticmethod
    def _row_context(cohort: Cohort, i: int) -> dict:
        row = cohort.df.iloc[i]
        # consequence derived from is_missense when present; else unknown
        cons = "unknown"
        if "is_missense" in cohort.df.columns:
            cons = "missense" if bool(row["is_missense"]) else "LoF/other"
        return {"gene": row.get("gene", "?"),
                "consequence": cons,
                "domain": row.get("domain", "?")}

    def order(self, cohort: Cohort) -> np.ndarray:
        channels = list(cohort.domain.channels)
        K = len(channels)
        reveals = np.stack([cohort.reveal(c) for c in channels], axis=1)
        n = len(cohort)
        out = np.zeros((n, K), int)

        stop_k = np.full(n, K, int)

        def _one(i):
            ctx = self._row_context(cohort, i) if self.context else None
            if not self.randomize_channel_order:
                picks, sk = self._order_one(channels, reveals[i], ctx=ctx)
                return i, picks, None, sk
            # Per-variant random LISTING order (position-bias diagnostic): the
            # model only ever sees `listed_channels`/`listed_reveal`, so
            # _order_one's returned picks are indices into THAT permuted list,
            # not canonical channel indices -- remap through `perm` before
            # this variant's row goes into `out`, or every downstream
            # metrics.py consumer (which assumes `out[i, k]` is a canonical
            # cohort.domain.channels index, consistent across rows) would
            # silently score the wrong channel for permuted rows.
            rng = np.random.default_rng(self.seed + i)
            perm = rng.permutation(K)                  # perm[listed_pos] = canonical channel idx
            listed_channels = [channels[j] for j in perm]
            listed_reveal = reveals[i][perm]
            picks, sk = self._order_one(listed_channels, listed_reveal, ctx=ctx)
            canonical_picks = [int(perm[p]) for p in picks]
            # sk counts POSITIONS in the acquisition sequence, not channel
            # identities, so it needs no remapping through `perm`.
            return i, canonical_picks, perm, sk

        if self.max_workers <= 1:
            for i in range(n):                       # serial + deterministic
                i_, picks, perm, sk = _one(i)
                out[i_] = picks
                stop_k[i_] = sk
                if perm is not None:
                    self.channel_perms[i_] = perm
        else:
            from concurrent.futures import ThreadPoolExecutor
            # Threads (not processes): each _order_one is a chain of blocking
            # network calls, so threads overlap the waits while the GIL keeps
            # the shared cache/counter mutations (already _lock-guarded) safe.
            # Results are written back by index, so ordering is preserved.
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                for i_, picks, perm, sk in ex.map(_one, range(n)):
                    out[i_] = picks
                    stop_k[i_] = sk
                    if perm is not None:
                        self.channel_perms[i_] = perm
        self.stop_k = stop_k
        return out


class AnthropicAgent(_StepwiseLLMAgent):
    def __init__(self, model_id: str, name: str = "anthropic", context: bool = False,
                 max_workers: int = 1, seed: int = GLOBAL_SEED,
                 cache_path: str | None = None, randomize_channel_order: bool = False,
                 allow_stop: bool = False, stop_wording: str = "default"):
        super().__init__(model_id, name, context=context, max_workers=max_workers,
                         seed=seed, cache_path=cache_path,
                         randomize_channel_order=randomize_channel_order,
                         allow_stop=allow_stop, stop_wording=stop_wording)
        import anthropic                       # requires [anthropic] extra
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _sampling_profile(self):
        # The Messages API exposes no `seed`, so this vendor has nothing to pin
        # at all -- fully default, which is also the closest available match to
        # gpt-5.x's forced temperature=1.
        return "default"

    def _complete(self, system, user):
        # max_tokens must comfortably exceed a category name. On models with
        # extended thinking, max_tokens is the thinking+text budget, so a tiny
        # value returns empty text; keep it generous and extract the TEXT block
        # explicitly rather than content[0] (which may be a thinking block).
        # NO temperature, and the Messages API has no `seed`, so this vendor
        # runs fully at its default -- the closest available match to gpt-5.x,
        # which is forced to temperature=1. See _sampling_profile(): pinning
        # only the vendors that ALLOW pinning is what made the frozen rows'
        # error bars incomparable, so the policy is now "default everywhere".
        m = self._client.messages.create(
            model=self.model_id, max_tokens=_MAX_TOKENS, system=system,
            messages=[{"role": "user", "content": user}])
        if getattr(m, "stop_reason", None) == "refusal":
            raise RefusalError("model refused")
        for b in m.content:
            if getattr(b, "type", None) == "text":
                return b.text
        return ""                              # no text block -> counted as parse failure


class OpenAIAgent(_StepwiseLLMAgent):
    """OpenAI adapter.

    DETERMINISM IS BEST-EFFORT, NOT GUARANTEED -- read before citing any number.
    Reasoning models (gpt-5.x) reject a pinned temperature outright ("Unsupported
    value: 'temperature' does not support 0 with this model. Only the default (1)
    is supported"), so the sampler cannot be frozen. They DO accept `seed`, which
    OpenAI documents as best-effort. We therefore pin the strongest combination
    the endpoint will take, in this order:
        1. seed                   (gpt-5.x reasoning models, and everything
                                   else that accepts a seed)
        2. neither                (endpoint refuses it)
    temperature is NOT on this ladder any more. It used to be rung 1, which
    meant a non-reasoning model plugged into this matrix would silently get a
    frozen sampler while the gpt-5.x rows could not -- the same incomparable-
    error-bars problem described in _sampling_profile(). Every vendor now runs
    at its default temperature.
    The winning profile is resolved ONCE per agent and reused.

    Measured consequence of NOT pinning: two runs of gpt-5.4-2026-03-05 over the
    same 491-variant cohort moved rho_vs_oracle 0.6513 -> 0.5951 (0.056), which
    is the same magnitude as the context-ablation effect being measured. Pinning
    the seed narrows this but does not provably remove it, so any headline claim
    still needs repeated runs with a reported spread.
    """

    # (label, kwargs-builder) tried in order; first accepted wins.
    _SAMPLING_LADDER = ("seed", "none")

    def _sampling_profile(self):
        # Names the POLICY, not the resolved rung. The rung is discovered on the
        # first live call (_resolve_profile), so keying on it would give that
        # first prompt a different cache key from every prompt after it, silently
        # orphaning one paid completion per run. The ladder itself is what is
        # stable and what actually characterises this adapter.
        return f"ladder+seed={self.seed}"

    def __init__(self, model_id: str, name: str = "openai", context: bool = False,
                 max_workers: int = 1, seed: int = GLOBAL_SEED,
                 cache_path: str | None = None, randomize_channel_order: bool = False,
                 allow_stop: bool = False, stop_wording: str = "default"):
        super().__init__(model_id, name, context=context, max_workers=max_workers,
                         seed=seed, cache_path=cache_path,
                         randomize_channel_order=randomize_channel_order,
                         allow_stop=allow_stop, stop_wording=stop_wording)
        import openai                          # requires [openai] extra
        import threading
        self._client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._token_param = None      # 'max_completion_tokens' | 'max_tokens'
        self._sampling = None         # resolved entry of _SAMPLING_LADDER
        # Single-flight: without this the first 8 worker threads would each pay
        # their own rejected probe call before converging on the same answer.
        self._probe_lock = threading.Lock()

    def _sampling_kwargs(self, profile):
        if profile == "seed":
            return {"seed": self.seed}
        return {}

    def _call(self, system, user, token_param, profile):
        kw = {"model": self.model_id,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}],
              token_param: _MAX_TOKENS,
              **self._sampling_kwargs(profile)}
        return self._client.chat.completions.create(**kw)

    def _resolve_profile(self, system, user):
        """Discover the token parameter and richest accepted sampling profile."""
        with self._probe_lock:
            if self._sampling is not None:
                return
            last = None
            for token_param in ("max_completion_tokens", "max_tokens"):
                for profile in self._SAMPLING_LADDER:
                    try:
                        r = self._call(system, user, token_param, profile)
                    except Exception as e:                  # noqa: BLE001
                        last = e
                        msg = str(e)
                        # A 400 naming a parameter we chose = that parameter is
                        # unsupported here; step down the ladder. Anything else
                        # (auth, rate limit, network) must propagate so the
                        # caller's retry/backoff handles it.
                        if not any(t in msg for t in
                                   ("temperature", "seed", "max_tokens",
                                    "max_completion_tokens")):
                            raise
                        continue
                    self._token_param, self._sampling = token_param, profile
                    return r
            raise last

    def _complete(self, system, user):
        if self._sampling is None:
            r = self._resolve_profile(system, user)
            if r is not None:                  # reuse the probe's own answer
                return r.choices[0].message.content or ""
        return self._call(system, user, self._token_param,
                          self._sampling).choices[0].message.content or ""


class GeminiAgent(_StepwiseLLMAgent):
    """Verified live on gemini-3.5-flash (2026-07-27): a real generate_content
    call returns a valid category name. Full-cohort sweeps are still blocked by
    the Google AI Studio project's FREE-TIER DAILY cap (quotaId
    GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue 20/day) --
    the model and prompt path work, the account is just not on a paid tier yet.
    Pass `cache_path` to persist completions across process restarts so a
    resumable driver (scripts/variant/warm_llm_cache.py) can chip away at the ~90
    distinct prompts a few per day until billing is enabled or the cache fills."""
    def __init__(self, model_id: str, name: str = "gemini", context: bool = False,
                 max_workers: int = 1, seed: int = GLOBAL_SEED,
                 cache_path: str | None = None, randomize_channel_order: bool = False,
                 allow_stop: bool = False, stop_wording: str = "default"):
        super().__init__(model_id, name, context=context, max_workers=max_workers,
                         seed=seed, cache_path=cache_path,
                         randomize_channel_order=randomize_channel_order,
                         allow_stop=allow_stop, stop_wording=stop_wording)
        from google import genai                # requires [gemini] extra
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _complete(self, system, user):
        from google.genai import types
        r = self._client.models.generate_content(
            model=self.model_id,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=_MAX_TOKENS,
                # NO temperature, deliberately -- see _sampling_profile().
                # Gemini accepts temperature=0 and it genuinely works (measured
                # 2026-08-02: 6/6 identical replies at 0, 2 distinct at the
                # default), which is exactly why sending it was wrong here: it
                # froze this vendor's sampler to zero noise while gpt-5.x is
                # forced to temperature=1 and cannot be frozen at all. Rows
                # produced under those two regimes have incomparable error bars,
                # and a reader comparing them across a table cannot see that.
                # Both vendors now run at their default sampler with only a
                # best-effort seed pinned.
                seed=self.seed))
        return getattr(r, "text", "") or ""     # no text -> counted as parse failure


class HFLocalAgent(_StepwiseLLMAgent):
    def __init__(self, model_id: str, name: str = "hf", context: bool = False,
                 max_workers: int = 1, seed: int = GLOBAL_SEED,
                 cache_path: str | None = None, randomize_channel_order: bool = False,
                 allow_stop: bool = False, stop_wording: str = "default"):
        super().__init__(model_id, name, context=context, max_workers=max_workers,
                         seed=seed, cache_path=cache_path,
                         randomize_channel_order=randomize_channel_order,
                         allow_stop=allow_stop, stop_wording=stop_wording)
        from transformers import pipeline      # requires [hf] extra
        self._pipe = pipeline("text-generation", model=model_id, max_new_tokens=32,
                              return_full_text=False)  # generated text only, not prompt+gen

    def _complete(self, system, user):
        # return_full_text=False -> generated_text is ONLY the model's output
        # (no prompt echo), so return it whole rather than a tail slice that
        # could cut a short valid reply.
        return self._pipe(f"{system}\n\n{user}\n")[0]["generated_text"]


# ---------------------------------------------------------------------------
# Claude Code CLI adapter
# ---------------------------------------------------------------------------
# Tools the CLI ships by default. Disabling them is NOT a permissions nicety:
# measured 2026-08-01, it cuts the system+tools prefix the model actually sees
# from ~27.5k tokens to ~12.5k, and it stops the model from being able to read
# the filesystem mid-answer (which would be run-time leakage, the third leakage
# class in this repo's audit). Anything the CLI adds in a future version is NOT
# covered by this list -- _verify_isolation() in scripts/variant/smoke_claude_cli.py
# is the check that actually holds, not this constant.
_CLAUDE_CLI_DISALLOWED_TOOLS = (
    "Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch", "WebSearch",
    "Task", "Agent", "TodoWrite", "NotebookEdit", "Artifact", "Skill",
    "ToolSearch", "Workflow", "AskUserQuestion", "Monitor",
)


def _find_claude_cli() -> str:
    """Locate the Claude Code binary. $CLAUDE_CODE_EXECPATH is set when running
    inside the VS Code extension / a Claude Code session; otherwise fall back to
    PATH. Raises with an actionable message rather than letting subprocess fail
    with a bare FileNotFoundError three frames down."""
    import shutil
    p = os.environ.get("CLAUDE_CODE_EXECPATH")
    if p and os.path.exists(p):
        return p
    p = shutil.which("claude")
    if p:
        return p
    raise RuntimeError(
        "Claude Code CLI not found. Set CLAUDE_CODE_EXECPATH to the binary, or "
        "put `claude` on PATH. (Inside a Claude Code session the env var is "
        "already set; a plain shell usually needs the PATH install.)")


class ClaudeCLIAgent(_StepwiseLLMAgent):
    """Anthropic adapter that goes through the Claude Code CLI (`claude -p`)
    instead of the Messages API, so a Claude row can be produced from a Claude
    subscription with NO ANTHROPIC_API_KEY on the machine.

    READ THIS BEFORE CITING ANY NUMBER FROM THIS ADAPTER.

    It is NOT equivalent to AnthropicAgent, and the difference is not cosmetic:

    1. HARNESS PREFIX. `claude -p` is an agent CLI, not a bare model endpoint.
       Even with `--system-prompt` (which REPLACES the default system prompt)
       and every tool disabled, the model still receives ~12.5k tokens of
       Claude Code scaffolding ahead of the benchmark prompt (measured
       2026-08-01; ~27.5k with tools left on). Every other adapter in this file
       sends the benchmark's ~200-token system prompt and nothing else. So this
       row measures "Claude Code answering the benchmark", not "Claude
       answering the benchmark", and it is NOT a clean vendor-axis control.
    2. NO SAMPLER CONTROL. The CLI exposes no temperature and no seed, so
       `self.seed` is accepted (for signature compatibility) and NOT forwarded.
       Determinism is weaker here than for OpenAIAgent, which at least pins
       `seed`. Repeated runs are mandatory before any claim.
    3. AUXILIARY MODEL. The CLI issues a side call to a small model
       (claude-haiku-4-5 as of 2.1.220) on every invocation. It does not
       produce the answer -- `result` comes from the pinned model -- but it does
       see the prompt, and it appears in the cost accounting.
    4. VERSION IS THE CLIENT'S, NOT THE EXPERIMENT'S. `--model` is honoured but
       the resolution, any silent fallback, and the harness prompt all belong to
       whatever CLI version is installed. _complete() therefore VERIFIES the
       model that actually ran against `model_id` and raises on mismatch rather
       than quietly recording a different model's answers.

    Isolation: every call runs with cwd set to a private empty directory, never
    the repo. Claude Code loads CLAUDE.md and its per-project auto-memory keyed
    by cwd; running in the repo would feed this benchmark's own notes and
    conclusions into the model's context -- construct leakage far worse than the
    ClinVar-assertion tool already removed from pwkbench/live/tools.py.
    """

    def __init__(self, model_id: str, name: str = "claude_cli", context: bool = False,
                 max_workers: int = 1, seed: int = GLOBAL_SEED,
                 cache_path: str | None = None, randomize_channel_order: bool = False,
                 allow_stop: bool = False, stop_wording: str = "default",
                 timeout_s: int = 300):
        super().__init__(model_id, name, context=context, max_workers=max_workers,
                         seed=seed, cache_path=cache_path,
                         randomize_channel_order=randomize_channel_order,
                         allow_stop=allow_stop, stop_wording=stop_wording)
        import tempfile
        self._exe = _find_claude_cli()
        self.timeout_s = int(timeout_s)
        # Private, EMPTY, non-git cwd -- see the isolation note in the class
        # docstring. Created once and reused so the CLI's prompt cache keeps
        # hitting across the sweep (a fresh dir per call would also re-key the
        # per-project memory path each time, which is fine, but would pay the
        # full cold cache-write on every single prompt).
        self._cwd = tempfile.mkdtemp(prefix="pwkbench_claude_cli_")
        # Every distinct model string the CLI reported, so a silent mid-sweep
        # switch is visible in the run record even if the guard is relaxed.
        self.models_seen = {}
        self.cost_usd = 0.0

    def _complete(self, system, user):
        import json, subprocess
        cmd = [self._exe, "-p", user,
               "--model", self.model_id,
               "--system-prompt", system,
               "--output-format", "json",
               "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
               "--disallowed-tools", *_CLAUDE_CLI_DISALLOWED_TOOLS]
        # check=False deliberately: the returncode is handled below so the
        # error message can carry the CLI's stderr, which CalledProcessError
        # would discard.
        proc = subprocess.run(cmd, cwd=self._cwd, capture_output=True,
                              text=True, timeout=self.timeout_s, check=False)
        if proc.returncode != 0:
            # Retryable by the base class; include stderr so a persistent
            # failure (bad flag, expired auth) is diagnosable from the traceback
            # instead of showing up as `retries` identical silent failures.
            raise RuntimeError(f"claude -p exited {proc.returncode}: "
                               f"{proc.stderr.strip()[:500]}")
        try:
            d = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"claude -p returned non-JSON: "
                               f"{proc.stdout.strip()[:500]}") from e

        # Model identity. The CLI may resolve an alias, or fall back to another
        # model when the requested one is unavailable on the account. Recording
        # answers under a model_id that did not produce them is exactly the kind
        # of silent mislabelling this benchmark is supposed to catch, so this
        # raises rather than warns. Non-fatal: the base retry loop will surface
        # it after `retries` attempts, and the mismatch is deterministic, so it
        # aborts the sweep quickly instead of paying for a whole bad run.
        usage = d.get("modelUsage") or {}
        answered_by = {v.get("canonicalModel") or k for k, v in usage.items()
                       if not str(v.get("canonicalModel") or k).startswith("claude-haiku")}
        with self._lock:
            for k, v in usage.items():
                self.models_seen[v.get("canonicalModel") or k] = \
                    self.models_seen.get(v.get("canonicalModel") or k, 0) + 1
            self.cost_usd += float(d.get("total_cost_usd") or 0.0)
        if answered_by and answered_by != {self.model_id}:
            raise RuntimeError(
                f"claude -p answered with {sorted(answered_by)}, not the "
                f"requested {self.model_id!r} -- refusing to record these "
                f"completions under the wrong model id.")

        if d.get("stop_reason") == "refusal":
            raise RefusalError("model refused")
        if d.get("is_error") or d.get("api_error_status"):
            raise RuntimeError(f"claude -p error: {d.get('api_error_status')} "
                               f"{str(d.get('result'))[:300]}")
        return d.get("result") or ""            # empty -> counted as parse failure
