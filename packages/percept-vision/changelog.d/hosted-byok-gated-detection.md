### Added — hosted BYOK gated detection (`vision.detect.gated`, contract 1.1.0)

`percept.hosted_byok` bumps `HOSTED_BYOK_CONTRACT_VERSION` to `"1.1.0"` and adds
`HostedByokVision.detect_gated(goals, frames)` / `stream_gated_detections(...)` — zero or more
gated proposals, each carrying the temporal identity (`frame_index` / `offset_s` /
`timestamp_unix_s`) a hosted incident binds to. Contract `"1.0.0"` remains executable and `judge()`
is unchanged; `HOSTED_BYOK_SUPPORTED_CONTRACT_VERSIONS` lists both, and a request pinned to an older
version is refused for capabilities that version never carried rather than silently upgraded.

The route reuses the SDK's own tier-0 gate (`percept.harness.tier0.MotionGate`) — there is no second
gate — and is defined over `HostedByokGateFrame`, the 48x36 grayscale tier-0 downsample the capture
path already produces. A larger frame is refused at the boundary, so full frames are needed only for
admitted moments: 1,200 gate frames (600 s at 2 fps) is ~2 MB against the 32 MiB ceiling. Thresholds
default to the deployment's calibrated per-statistic cutoff (`block_max` @ 0.20 by default), matching
`compile_watchspec`; the goal's `condition` text is carried as provenance and never interpreted.

`refractory_s` (default 15.0) is the gate's own per-goal cadence, so a 600 s clip yields at most 40
proposals per goal — measured — and detection itself spends **zero** provider calls. New per-provider
capability `vision.detect.gated` is declared in `HOSTED_BYOK_CAPABILITIES` and enforced at preflight,
so an unentitled route is refused before a clip is walked. `build_info()` on the factory and on a
bound adapter reports the contract version, the capability table, and the detection bounds;
`percept.build_info().hosted_byok_contract_version` reports the version from the package accessor
without widening the frozen four-field build-stamp receipt.

Isolation: the scan runs under a context-scoped credential firewall that raises on any provider
environment key read (caller-supplied salience adapters included), constructs no provider client or
backend, and is never offloaded to a thread or executor — cancellation stops between two frames with
no child process to orphan. A run that admits nothing raises `HostedByokDetectionStarved` carrying
the observed salience distribution instead of returning an empty list indistinguishable from
"nothing happened".
