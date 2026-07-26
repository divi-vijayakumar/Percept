# Percept stage-validation report

Generated: 2026-07-26T22:44:51+00:00
Runtime under test: exact launch wheel `percept 1.0.0` (`bba572a`)

Corpus: **100 cases**

| Dataset | Cases |
|---|---|
| charades | 20/20 |
| omnipro | 20/20 |
| perception_test | 20/20 |
| proactiveqa | 20/20 |
| repcount | 20/20 |

Long-horizon sources: **6**

| Goal | Duration | Expected occurrences |
|---|---|---|
| `perception_test::showing_something_to_the_camera` | 225.9s | 9 |
| `repcount::frontraise` | 97.0s | 5 |
| `repcount::situp` | 170.0s | 5 |
| `perception_test::launching_something` | 60.9s | 3 |
| `repcount::jumpjack` | 48.0s | 3 |
| `charades::someone_is_cooking_something` | 60.7s | 2 |

## Totals

PASS **16** · FAIL **7** · UNKNOWN 0 · suites MISSING 0

## STAGE 0

### Ingest + normalize (`sv0_ingest`)

Status: **PASS** (PASS 6)

| Test | Status | Detail |
|---|---|---|
| SV-0.1 | PASS |  |
| SV-0.2 | PASS |  |
| SV-0.3 | PASS |  |
| SV-0.4 | PASS |  |
| SV-0.5 | PASS |  |
| SV-0.6 | PASS |  |

### Planner + compiler (`sv0_compile`)

Status: **FAIL** (FAIL 4)

| Test | Status | Detail |
|---|---|---|
| SV-0.7 | FAIL |  |
| SV-0.8 | FAIL |  |
| SV-0.9 | FAIL |  |
| SV-0.10 | FAIL |  |

## STAGE 1

### Admission (VID/VAD, dwell) (`sv1_admission`)

Status: **FAIL** (FAIL 1, PASS 3)

| Test | Status | Detail |
|---|---|---|
| Dwell correctness — hysteresis, debounce, expiry (exact) | PASS |  |
| Dwell under a monotonous signal — does the interval hold? | FAIL |  |
| Per-Watch attribution — N Watches, one source, zero cross-dispatch | PASS |  |
| State-transition table (suppress / accumulate / admit+dwell) | PASS |  |

## STAGE 2

### Local perception (senses, reducers) (`sv2_senses`)

Status: **PASS** (PASS 4)

| Test | Status | Detail |
|---|---|---|
| ? | PASS |  |
| ? | PASS |  |
| ? | PASS |  |
| ? | PASS |  |

## STAGE 3

### Semantic judges (boundedness) (`sv3_pool`)

Status: **FAIL** (FAIL 1, PASS 3)

| Test | Status | Detail |
|---|---|---|
| ? | PASS |  |
| ? | FAIL |  |
| ? | PASS |  |
| ? | PASS |  |

## STAGE 4

### Cognition + disposition (`sv4_cognition`)

Status: **FAIL** (FAIL 1)

| Test | Status | Detail |
|---|---|---|
| SV-4.7 | FAIL |  |
