# Local-VLM feasibility attempt 1

Status: **INVALID OUTPUT — no adjudication recorded**

The first local inference used the frozen case `B4V2-ZGT-01`, prompt, model revision
`46d4cf06a06ffc1a766c214174f9cbed2f45bcab`, 16 video frames, temperature zero, and `mlx-vlm
0.7.0`. It completed with exactly 350 exclamation-mark tokens and returned no JSON object, label,
or visual evidence. The response is unusable and is not counted as a classification.

The model repository states that its weights were converted with `mlx-vlm 0.1.11`. Runtime `0.1.11`
does not expose the later native `--video` option, so the second feasibility identity supplies the
same 16 chronological video frames as an ordered image sequence. Cases, clips, question, labels,
prompt, model weights, and decoding temperature are unchanged.

