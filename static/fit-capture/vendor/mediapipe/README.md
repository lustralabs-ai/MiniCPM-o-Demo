# Vendored MediaPipe assets

Served locally so fit-capture does not depend on a CDN. These are plain git files (no LFS: it is
not enabled on the GitHub repo); each is under 10 MB.

| File | Source |
|---|---|
| `vision_bundle.mjs`, `wasm/*`, `package.json` | npm `@mediapipe/tasks-vision@0.10.14` (tarball sha512 `vOifgZhkndgybdvoRITzRkIueWWSiCKuEUXXK6Q4FaJsFvRJuwgg++vqFUMlL0Uox62U5aEXFhHxlhV7Ja5e3Q==`, matches the registry) |
| `pose_landmarker_full.task` | `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task` |

License: Apache-2.0 (MediaPipe, Google LLC).

To upgrade, replace the files from the new npm tarball, update this table, and test the page on
a phone.
