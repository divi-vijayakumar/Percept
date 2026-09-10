# Local-VLM feasibility attempt 2

Status: **LOAD FAILURE — no inference and no adjudication**

The conversion-matched `mlx-vlm 0.1.11` runtime was invoked on the frozen `B4V2-ZGT-01` ordered
16-frame sequence. It stopped before inference with `Model type qwen2_5_vl not supported`. No tokens,
label, or visual claim were generated. This attempt is not counted as a classification.

The third and final feasibility identity restores `mlx-vlm 0.7.0`, which can load this architecture,
but supplies ordered frames rather than using the native-video path that returned invalid punctuation.
Cases, frame pixels/order, prompt, model weights, and decoding remain unchanged.

