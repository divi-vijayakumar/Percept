# Third-party notices

Percept is licensed under Apache-2.0 (see LICENSE). The `percept-vision` package bundles no
third-party code; it depends on packages that are installed separately, each under its own
licence. Notable ones:

- **numpy** (BSD-3-Clause and others; its wheels include OpenBLAS and the GCC runtime under their
  own permissive or runtime-exception terms) and **pillow** (MIT-CMU): required.
- **opencv-python-headless** (Apache-2.0), with the `video` extra. Its wheels bundle FFmpeg,
  which is licensed under the LGPL-2.1; see that package for its notices and sources.
- **google-genai**, **anthropic**, **openai** and their dependencies, with the judge extras: Apache-2.0,
  MIT and BSD licences. **certifi** is MPL-2.0 and is used unmodified.

The benchmark harness in `benchmarks/` is not part of the package. Its optional detector baselines
download torchvision's COCO-trained SSDlite weights at run time. Those weights carry no licence of
their own, and their terms derive from the COCO dataset (annotations CC BY 4.0, images under
Flickr's terms). They are never redistributed here: if you run those baselines, determine for
yourself whether your use is permitted. Datasets the harness reads (such as MEVA, CC BY 4.0) are
downloaded by you and are not included.
