# Sources

A source brings frames. Percept reads cameras, files and streams itself, or takes frames your
application already has.

## `open_source(...)` then `watch(source)`

One place configures a source: `open_source()` owns capture (rate, frame size, reconnection, and
the `on_frame` / `on_health` callbacks); `watch()` owns monitoring and delivery.

```python
source = percept.open_source("rtsp://camera.local/stream", fps=5, on_frame=show)
async for delivery in monitor.watch(source, ack="manual"):
    ...
```

`monitor.watch("rtsp://camera.local/stream")` is shorthand: it opens the URI with
`open_source`'s defaults. Open the source yourself whenever you need anything else.

```python
percept.open_source(uri, *, stream_id=None, fps=None, max_side=640, loop=False, live=None,
                    backend="auto", epoch=0, backoff=(2.0, 30.0), stall_timeout=30.0,
                    on_health=None, on_frame=None, preroll=None, ffmpeg="ffmpeg")
```

| `uri` | Read with |
|---|---|
| `rtsp://`, `rtsps://` | OpenCV, over TCP |
| `webcam:N` | OpenCV device N |
| an HLS or DASH manifest (`.m3u8`, `.mpd`) | an `ffmpeg` subprocess (`ffmpeg` must be on `PATH`) |
| a file path, `file://` URL or other URL | OpenCV |

- OpenCV is the `video` extra (`pip install "percept-vision[video]"`). A missing OpenCV or ffmpeg
  raises in `open_source`, not later. `backend="ffmpeg"` forces the ffmpeg reader.
- **Live or file.** Every `scheme://` URL is live unless its path ends in a container suffix
  (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.m4v`); local files are not. `live=` overrides the
  guess. A file ends at its end (or restarts in a new epoch with `loop=True`).
- **Reconnection.** A live source reconnects forever, with jittered backoff from 2 s doubling to
  30 s (`backoff`). A live connection that delivers no frame for `stall_timeout` seconds (30) is
  torn down and retried.
- **Frame size.** Frames are shrunk so the longest side is at most `max_side` (640).
- **Credentials.** The default `stream_id` is the URI without `user:password@`, query string or
  fragment, and health details are scrubbed of credentials. OpenCV's own logger and the ffmpeg
  command line still see the raw URI (set `OPENCV_LOG_LEVEL=ERROR` to quiet OpenCV).
- `source.link` is `"connecting"`, `"up"`, `"down"` or `"ended"`.

## Capture rate and processing rate

Two different rates apply, set in two places:

- **Capture rate**: `open_source(fps=...)` samples the source (default: every frame). Capture
  never waits for the consumer: a slow consumer gets the newest frame, not a queue (files are the
  exception and hand over every sampled frame).
- **Processing rate**: `monitor.watch(source, fps=5)` caps how many frames a second, on the stream
  clock, go through the quality gate, scene check and triggers. `None` processes every frame.
  Skipped frames are counted as `frames_capped` in coverage. Streams started by `Monitor.push`
  use the default of 5.

The pre-roll (the last 8 s, sampled at 2 frames a second, from which event clips and
verifications are cut) is fed at the capture rate, so the processing cap never thins the
evidence of an event. An event window (`Condition.clip_s`, `clip_frames`) the pre-roll can't
hold, longer than its seconds or with more frames than its rate can give, is refused with
`ValueError` when the condition or the stream is added; a source with its own `PreRoll` sets
both. To give an event a longer or denser window, pass one to the source:

```python
from percept.clips import PreRoll

source = percept.open_source(url, preroll=PreRoll(seconds=12, fps=4))   # also PushSource(..., preroll=)
```

## Push sources and `set_link`

If your application already decodes frames, push them:

```python
camera = percept.PushSource("dock-cam")          # max_side=640, maxlen=8
decisions = monitor.watch(camera, ack="manual")
camera.push(image_bgr)                           # uint8 HxWx3 (BGR) or HxW; from any thread
camera.close()                                   # ends the stream
```

Or skip the object: `monitor.push("dock-cam", image)` starts a push stream on first use, and
`monitor.watch("dock-cam")` then delivers its decisions.

A push source has no connection of its own, so **your application reports its outages**:

- `camera.set_link(False, "decoder lost the feed")` opens a `link_down` blind interval and raises
  health `link_down`; looks that need a current frame wait.
- `camera.set_link(True)` (or simply pushing a frame again) raises `link_up` and starts a new
  epoch: answers to requests sent before the outage are then stale, and held states are rechecked
  on the new frames.
- `monitor.set_link(stream_id, up, detail="")` does the same for a stream the monitor created
  from `push`, or for a custom source.

`PushSource.push` and `PushSource.set_link` are safe to call from any thread; link reports
arrive in order and strictly alternate. `Monitor.set_link`, like the monitor's other methods
except `push`, is called from the monitor's event loop.

## Link health

`watch(source)` subscribes the monitor to every source that has `add_health_listener` (every
`open_source` source and every `PushSource`), so an outage is a `link_down` blind interval and a
health event without further wiring, even when you also pass your own `on_health`.

A frame older than 10 s is never judged as the current one: a look that needs the current frame
waits (counted as `evidence_unavailable`) instead of judging a frozen picture.

## Custom sources

Any object with a `stream_id` and an async `frames()` iterator of `percept.Frame` is a source. It
should keep `seq` and `t` increasing within an epoch, raise `stream_epoch` on every reconnect, and
report outages as health rather than raising. An optional `async aclose()` is called at shutdown.

## The static-camera assumption

Percept is built for **fixed-viewpoint cameras**. It is not for PTZ, body-worn, vehicle,
drone or phone video, or screen content.

- When most of the picture changes at once (a bumped camera, a PTZ move, an infrared day/night
  switch), Percept declares a **scene reset**, never an event: triggers pause until the picture
  holds still for 2 s, change triggers rebuild their baseline, and every condition gets one look
  at the settled scene. The reset is a `scene_reset` blind interval and health event.
- On a camera that keeps moving, change triggers fire on nearly every frame; looks queue up and
  the budget drops them. Expect poor coverage there, and see it in `coverage()`.
- **Frame quality.** Frames that look corrupt (decoder smear from packet loss) or blurry against
  the stream's own recent sharpness are not judged (`dropped_quality`). Too many bad frames open a
  `degraded_video` blind interval. A blur that lasts becomes the new normal, reported as
  `degraded` health, so a camera left out of focus is never silently blind. Known limit: on a
  low-detail scene with visible sensor noise, a defocused lens can still read as sharp; such
  frames are judged, not dropped.
