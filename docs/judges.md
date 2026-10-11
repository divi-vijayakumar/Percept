# Judges

A judge decides what is true in a request's images. Every judge returns, for each condition it
was asked about, an answer (`yes`, `no` or `unknown`), a confidence in `[0, 1]` and a reason of at
most 160 characters. A judge never raises: a timeout, outage, refusal or malformed reply comes back
as a *failed* answer, which never changes a decision.

## Built-in judges

Each vendor SDK is an optional extra, imported only when the judge is built. There is no default
model: pass the model id your provider offers.

```python
percept.judges.gemini(model, *, api_key=None, price=None, can_batch=True, client=None, name=None,
                      structured_output=True, thinking_budget=0)          # pip install "percept-vision[gemini]"
percept.judges.claude(model, *, api_key=None, price=None, can_batch=True, client=None, name=None,
                      structured_output=True)                             # pip install "percept-vision[claude]"
percept.judges.openai_compatible(model, *, base_url=None, api_key=None, price=None, can_batch=True,
                                 client=None, name=None, structured_output=True,
                                 token_param="auto")                      # pip install "percept-vision[openai]"
```

- **Keys.** `gemini` reads `GEMINI_API_KEY`, then `GOOGLE_API_KEY`. `claude` uses whatever the
  Anthropic SDK resolves (`ANTHROPIC_API_KEY`). `openai_compatible` reads `OPENAI_API_KEY`; with a
  `base_url` and no key it sends a placeholder, as local servers expect.
- **`openai_compatible`** covers OpenAI and servers that speak its chat-completions API (vLLM,
  Ollama, LM Studio): point `base_url` at the server. `token_param="auto"` sends the output limit
  as `max_completion_tokens` to OpenAI and Azure OpenAI hosts and as `max_tokens` elsewhere.
- **`name`** (default: the model id) names the judge in observations, and is the route name a
  `Condition.judge` pins to inside a cascade.
- **`structured_output=False`** drops the JSON schema for models that reject one; JSON is still
  requested and validated.
- **`thinking_budget`** (Gemini) caps thinking tokens, which count against the reply limit. The
  default 0 turns thinking off where the model allows it. A model that can't turn it off rejects
  0: pass a budget it accepts, or `None` to send no setting.
- **`can_batch=True`** asks about every due condition with the same evidence in one request.
  With `False`, each condition is its own request (and its own share of the budget).
- **`client`** replaces the SDK client (for proxies, custom transports and tests).

`percept.judges.FakeJudge(script)` is a scripted judge for tests and offline demos. It runs the
same prompt, parsing, retry and failure code as a real judge; only the provider is scripted.

## What a request looks like

- **Images**: JPEG, quality 80, longest side at most 640 px, cropped to the condition's zones.
  A state is judged on the newest frame; an event on up to 4 frames from the 4 s before its
  trigger, oldest first.
- **Prompt**: each condition's id, text and kind, its `not_when` cases, and up to 3 notes from
  occurrences marked as false alarms. Verification asks the judge to be strict.
- **Reply**: JSON, `{"results": [{"id", "answer", "confidence", "reason"}]}`, one entry per
  condition. A missing entry fails that condition as `missing`; a broken one as `malformed`.
- **Deadline and retries**: each request has 8 s. A rate limit (408, 429), a server error (5xx)
  or a dropped connection is retried once, after 0.25 to 0.5 s or the provider's retry-after hint,
  if that still fits in the deadline. Any other error fails the request as `unavailable`.
- **Reply limit.** A request's `max_output_tokens` grows with the conditions it carries, so a
  batched reply is not cut off: 64 tokens plus 80 per condition plus a third of the condition ids'
  length in characters, at least 400 and at most 4096. A reply cut off at the limit fails as
  `malformed`, never as a partial answer. The cost reservation uses the same limit.

## Cascades and routes

```python
cheap = percept.judges.openai_compatible("<local model>", base_url="http://localhost:8000/v1")
strong = percept.judges.gemini("<model id>")
judge = percept.judges.cascade(cheap, strong, thresholds=(0.8, None))
```

`cascade(*judges, thresholds=None, name=None)` asks its members in order, cheapest first:

- Per condition, a `yes` or `no` at or above that member's threshold stands. `None` (the default)
  uses the condition's `min_confidence`. An `unknown`, a weaker answer or a failure escalates the
  condition to the next member. The last member's answer always stands.
- **Routes.** `Condition(..., judge="<member name>")` asks only that member, for detection.
- **Verification** always goes to the last member, for every condition; routes don't apply.
- All hops share the request's deadline, and every hop's usage is counted against the budget and
  charged to the conditions it carried.
- Member names must be distinct. The cascade has a `price` only when every member has one.

## Prices: PriceModel

```python
percept.PriceModel(usd_per_mtok_in=..., usd_per_mtok_out=..., tokens_per_image=...)
```

Prices are yours to set from your provider's price list; Percept ships none. A judge with a price
reports its cost per request, and the monitor reserves each request's **worst case** before it is
sent: every attempt it may make, each with every image at `tokens_per_image`, the prompt text
counted generously (3 characters a token) and a full reply at the reply limit. A judge without a
price reports no cost, and the cost cap can only count what is reported.

## Budgets

| `Monitor(...)` argument | Default | Scope |
|---|---|---|
| `max_requests_per_min` | `6` | per stream: outbound attempts (retries and cascade hops included) in any 60 s |
| `max_cost_per_hour` | `None` | the whole monitor: dollars in any hour |
| `max_concurrent_requests` | `4` | the whole monitor: requests in flight |

- **Reservation.** Before a request is sent it reserves the most attempts it can make (2 for the
  built-in judges: one try and one retry; a cascade counts only the members the request can
  reach) and, with a price, its worst-case dollars. When it finishes, what it actually used stays
  charged for the window. So with the default of 6, a stream makes at most 6 attempts in any
  minute: 3 requests if every one retries, more when they don't. A request that would not fit waits.
- **Guarantee.** No 60 s span holds more attempts than `max_requests_per_min`, and no hour more
  dollars than `max_cost_per_hour`, for judges that report their attempts truthfully and have a
  price. Without a price the cost cap is best effort. A cancelled request reports no usage, so its
  whole reservation stays charged for the window.
- **Priority** when the budget is short: verifications first, then event detections, state
  detections, heartbeats and rechecks. An event detection waits at most 3 s and a state detection
  10 s before it is dropped (`dropped_budget`); heartbeats and rechecks wait for the next slot.
  A budget that refuses due looks opens a `budget_exhausted` blind interval.
- **Single flight.** Per stream, one detection request and one verification may be in flight;
  looks that become due meanwhile are merged into the next request.
- `max_requests_per_min` must be at least one judge call's attempts (2 for the built-in judges).

## Reporting cost

`Monitor.stats()` reports:

- `cost_usd`: the settled cost the judges reported, per stream and in total;
- `cost_reserved_usd`: dollars still held against the cap (requests in flight, and cancelled
  requests for the rest of their hour);
- `cost_usd_in_window`: everything the hourly cap counts now;
- `streams[id]["conditions"][condition_id]`: `requests`, `attempts` and `cost_usd` per condition.
  A request's usage is split evenly among the conditions it carried; in a cascade, each hop is
  charged to the conditions that member was asked.

Each decision's `cost` is the usage charged to its occurrence since the previous decision.

## Writing your own judge

Any object with `name`, `can_batch`, `price` and `async def judge(request) -> JudgeResult`
satisfies `percept.Judge`. It must return exactly one `Observation` per item in the request,
never raise, and report every provider request it made as a `RequestUsage`. Optional attributes
narrow the budget's reservation: `max_attempts` (default 2), `max_attempts_for(request)` and
`worst_case_cost_for(request)`. They are trusted: a judge that under-reports is under-reserved.
