# Local Qwen2.5-VL-3B diagnostic result

Status: **INCONCLUSIVE — zero cases adjudicated**

The frozen local-only run completed all 15 unmatched calibration fires. Every response selected
`ambiguous`: 12 with reported confidence `0.0` and three with `0.5`. Eleven free-text responses
mentioned an apparent crossing and four said no crossing, but the model itself declined every
classification. Those descriptions are not converted into labels.

Measured local execution for the 14 batch-recorded cases was 275.946 seconds total with 4.612 GB
maximum reported peak memory. The separately preserved first feasibility case used the same identity
and also returned `ambiguous`. Provider calls and provider cost were zero.

The immutable evidence result is
`/Volumes/Percept/Percept-Eternal/Percept-Benchmarks/virat-fixed-watch/evidence/b4-v2-local-vlm/local-vlm-results.json`
with SHA-256 `34879de5f582e223124119db92e84bb109d0db81ee856f1d2fa7ba013526ba6d`.
All 15 fires retain provisional-FP status; this model result neither confirms nor disproves them.

