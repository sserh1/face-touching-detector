# Optimization notes

This documents why the original `index.py` is resource-hungry, why the
detector fired on non-touches, and what `optimized.py` does about both.

## The CPU problem

`index.py` detects when a hand touches your face and sleeps the display. It
works, but it pins a CPU core and heats the machine. Three causes, in order of
impact:

1. **`mp.solutions.Holistic` runs every frame.** Holistic bundles three heavy
   models — FaceMesh (468 landmarks), Pose, and two Hands — and runs all of them
   on every frame. The task only needs a face bounding box and hand positions,
   so nearly all of that compute is wasted.
2. **Uncapped loop.** `while True` with no sleep processes frames as fast as the
   CPU allows, holding a core at ~100%. Detecting a hand on a face does not need
   30–60 fps.
3. **Full-resolution frames.** Inference cost scales with pixel count, and the
   frame is fed in at full camera resolution.

Minor: `cv2.waitKey(1)` never returns `q` because there is no `imshow` window,
so the intended quit key does nothing — the script can only be killed with
Ctrl+C.

| Change | Effect |
| --- | --- |
| **FaceDetection (BlazeFace) + Hands** instead of Holistic | Drops FaceMesh + Pose. Provides exactly the box + landmarks the comparison needs. Main saving. |
| **Face-first gating** | The cheap face detector runs first; the expensive hand model runs only when a face is present. When you are out of frame, the hand step is skipped entirely. |
| **FPS cap (`TARGET_FPS = 6`)** | Sleeps off the remainder of each frame budget so the CPU idles between frames instead of spinning. |
| **Downscale to 480px wide** + request 640×480 camera feed | Fewer pixels per inference and less data to decode/move. |
| **Trigger cooldown** | Avoids firing `pmset` repeatedly while a touch persists. |
| **Explicit cleanup** | Releases the camera and closes both models on exit. |

## The false-positive problem

The first `optimized.py` (like `index.py`) slept the display on **any single
frame** in which **any one** of the 21 hand landmarks fell inside the face
box. That matched a lot of things that are not touches:

1. **No persistence.** One frame was enough: a hand passing the face, or a
   one-frame detection glitch.
2. **2D only.** A hand anywhere between the face and the camera overlaps the
   face in the image: gesturing, reaching for the screen.
3. **Loose region.** `FACE_PADDING = 0.05` was 5% of the *frame*, not the
   face — for a face a fifth of the frame wide, 25% of the face per side —
   and the region is a rectangle. It covered hair, neck, collar and
   background; on a test portrait it was 1.65× the face box's area.

What `optimized.py` does now:

| Change | Effect |
| --- | --- |
| **Persistence** (`TOUCH_CONFIRM_SECONDS = 0.5`) | A touch must last 0.5 s, tolerating one dropped frame. Passes and glitches are ignored. |
| **Depth check** (`MIN/MAX_DEPTH_RATIO`) | Compares the hand's apparent size with the face's to tell a hand *on* the face from one *in front of* it. See below. |
| **Face ellipse** | Ellipse from forehead to chin, sized relative to the face. Excludes hair, neck and the box corners. `MIN_POINTS_IN_FACE = 2` landmarks must be inside. |
| **Face memory** (`FACE_MEMORY_SECONDS = 1.0`) | A hand covering the face often hides it from the face detector. The last face position is reused instead of skipping the frame. |
| **Full hand model** (`model_complexity=1`), presence ≥ 0.7, detected every frame | The lite model's hand scale was too noisy for the depth check, and in the tests it missed a palm covering the face. |

### How the depth check works

MediaPipe Hands also returns *world landmarks*: the hand's 3D shape in metres,
aligned with the image axes. The landmarks' spread in pixels divided by their
spread in metres is the hand's pixels-per-metre — its apparent size with the
hand's pose factored out (a fist and an open palm give the same value). The
face box is ~14 cm wide, which gives the same for the face. The ratio is ~1
when the hand is at the face's depth and grows as the hand gets nearer the
camera: a hand 40 cm from the camera in front of a face at 60 cm reads ~1.5.

The constants are population averages (MediaPipe's world hand is ~16 cm long,
the face box ~14 cm wide), so a large hand reads a little nearer than it is.
Touches in the tests below read 1.03–1.29; the cut-off is 1.6.

### Results

No webcam footage was used. The tests are synthetic: the Grace Hopper portrait
that ships with matplotlib, placed in a 480×360 frame, with Apple emoji hands
pasted onto it. MediaPipe detects these as hands and returns world landmarks
for them. Old logic = the previous commit's `optimized.py`; each scenario is a
6 fps sequence.

| Scenario | Old | New |
| --- | --- | --- |
| 9 touches: cheek, mouth/chin, eye/forehead, left hand, two other hand shapes, two palms over the face, a brief 0.67 s touch | 8 fired (missed one palm over the face) | 9 fired |
| Hand passes over the face for 1 or 2 frames | fired | quiet |
| Detection flicker (1 frame in 3) | fired | quiet |
| Hand in front of the face, 1.7× and 2.2× nearer | fired | quiet (depth 1.94, 2.14) |
| Hand very near the camera, hiding the face | quiet (no face found) | quiet |
| Hand by the hair/hat | fired | quiet |
| Hand at the neck/collar | fired | quiet |
| Hand beside the face, fingers over the cheek edge | fired | fired |

What it still cannot tell apart: a hand at the face's depth that is *almost*
touching — holding a cup at the mouth, a phone at the ear, fingers just over
the edge of the cheek. From one camera, a 1–2 cm gap is invisible.

### CPU cost of the changes

Measured per frame on the synthetic scenes: 13.6 ms with no hand in view (old:
16.8 ms) and 34 ms with a hand in view (old: 17 ms). At 6 fps that is ~8% of
one core without a hand and ~20% while a hand is visible. The full hand model
is the difference. `model_complexity=0` brings it back to ~17 ms, but its
depth readings were 5–30% higher and less consistent, so `MAX_DEPTH_RATIO`
would need to go up to ~1.8.

## Tuning

Watch what the detector sees, without sleeping the display:

```bash
venv/bin/python optimized.py --debug --dry-run
```

The window shows the face ellipse and each hand's landmarks:

- **red**: counts as a touch.
- **orange**: over the face, but rejected by the depth check.
- **green**: away from the face.

Each hand is labelled `in N` (landmarks inside the ellipse) and `depth R`.
Every confirmed touch is also printed with those numbers.

Knobs at the top of `optimized.py`:

- `TOUCH_CONFIRM_SECONDS` — raise to ignore longer brushes past the face;
  lower to catch quicker touches.
- `MAX_DEPTH_RATIO` — your real touches should read ~1.0–1.3. If they show
  orange, raise it; if hands in front of your face still count, lower it
  toward ~1.4.
- `FACE_REGION_SCALE` — grow (>1) or shrink (<1) the face ellipse.
- `MIN_POINTS_IN_FACE` — raise to require more of the hand on the face.
- `TARGET_FPS`, `PROCESS_WIDTH`, `TRIGGER_COOLDOWN` — as before.

## Run

From the project folder:

```bash
venv/bin/python optimized.py
```

Stop with Ctrl+C, or `q` in the `--debug` window.

The decision logic (ellipse, depth ratio, persistence) has unit tests:

```bash
venv/bin/python -m unittest test_optimized
```
