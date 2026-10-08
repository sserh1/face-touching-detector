import argparse
import math
import time
import subprocess
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

# Lightweight face-touch detector.
#
# Why this is cheaper than index.py (which used mp.solutions.Holistic):
#   1. Holistic runs FaceMesh (468 landmarks) + Pose + 2 hand models on EVERY
#      frame. We don't need any of that precision. We only need a face bounding
#      box and hand positions, so we use the much lighter FaceDetection
#      (BlazeFace) + Hands.
#   2. The original loop was uncapped (`while True` with no sleep), so it pinned
#      a CPU core at 100% and ran inference as fast as possible -> heat. Here we
#      cap to TARGET_FPS; hand-to-face contact does not need 30-60 fps.
#   3. We downscale the frame before inference. Inference cost scales with pixel
#      count; a 480px-wide frame is plenty for this task.
#   4. Face detection is cheap, hand detection is not. We run the face detector
#      first and ONLY run the hand model when a face is actually visible. When
#      you're not in frame, we skip the expensive step entirely.
#
# Why it fires far less on non-touches. The old rule was "any of the 21 hand
# landmarks inside the face box (padded by 5% of the frame) in any single
# frame", which also matched:
#   1. One-frame glitches and a hand merely passing the face. A touch must now
#      last TOUCH_CONFIRM_SECONDS.
#   2. A hand between the face and the camera (gesturing, reaching for the
#      screen): a 2D overlap is not contact. The hand's size relative to the
#      face shows how far in front of it the hand is, see hand_depth_ratio().
#   3. Hair, neck and background: the padding was relative to the frame and
#      the box is a rectangle. The face region is now an ellipse sized to the
#      face (forehead to chin), and MIN_POINTS_IN_FACE landmarks must be in.
# The depth check relies on the full hand model, not lite (see Hands below).

TARGET_FPS = 6           # frames actually processed per second
PROCESS_WIDTH = 480      # frame is downscaled to this width before inference
TRIGGER_COOLDOWN = 3.0   # seconds to wait after sleeping the display

TOUCH_CONFIRM_SECONDS = 0.5  # a touch must persist this long to count
TOUCH_GRACE_SECONDS = 0.4    # detection gaps tolerated within one touch (~1 frame)
FACE_MEMORY_SECONDS = 1.0    # reuse the last face position while a hand hides it
FACE_REGION_SCALE = 1.0      # grow (>1) or shrink (<1) the face ellipse
MIN_POINTS_IN_FACE = 2       # hand landmarks (of 21) that must be inside the face
MIN_DEPTH_RATIO = 0.7        # hand vs face scale; below = hand far behind the face
MAX_DEPTH_RATIO = 1.6        # above = hand in front of the face, not touching it

FACE_WIDTH_M = 0.14          # real-world width of the face-detection box
FOREHEAD = 0.25              # the face box stops at the brows: extend it up by this
CHIN = 0.1                   # ...and down by this, as fractions of its height


@dataclass
class FaceRegion:
    cx: float     # ellipse centre and semi-axes, in pixels
    cy: float
    rx: float
    ry: float
    width: float  # face box width in pixels, the face's apparent size

    @classmethod
    def from_detection(cls, detection, frame_w, frame_h):
        box = detection.location_data.relative_bounding_box
        x, w = box.xmin * frame_w, box.width * frame_w
        top = (box.ymin - FOREHEAD * box.height) * frame_h
        bottom = (box.ymin + (1 + CHIN) * box.height) * frame_h
        return cls(
            cx=x + w / 2,
            cy=(top + bottom) / 2,
            rx=w / 2 * FACE_REGION_SCALE,
            ry=(bottom - top) / 2 * FACE_REGION_SCALE,
            width=w)

    def contains(self, points):
        """Which of the (N, 2) pixel points fall inside the face ellipse."""
        dx = (points[:, 0] - self.cx) / self.rx
        dy = (points[:, 1] - self.cy) / self.ry
        return dx * dx + dy * dy <= 1.0


def hand_depth_ratio(points, world, face_width):
    """Hand scale / face scale: ~1 at the face's depth, >1 nearer the camera.

    MediaPipe's world landmarks are the hand's 3D shape in metres, aligned with
    the image axes. The spread of the landmarks in pixels over their spread in
    metres is the hand's pixels-per-metre, independent of the hand's pose (a
    fist and an open palm give the same value). The face box, FACE_WIDTH_M
    wide, gives the same for the face, so the ratio compares their distances
    from the camera.
    """
    img = points - points.mean(axis=0)
    wld = world - world.mean(axis=0)
    hand_px_per_m = math.sqrt((img ** 2).sum() / (wld ** 2).sum())
    return hand_px_per_m / (face_width / FACE_WIDTH_M)


@dataclass
class HandCheck:
    points: np.ndarray  # 21 landmark positions in pixels
    inside: int         # how many of them are inside the face ellipse
    depth_ratio: float
    touching: bool


class TouchDebouncer:
    """Confirms a touch once it has been seen for `confirm` seconds.

    Gaps of up to `grace` seconds don't restart the clock: hand and face
    detection flicker while a hand covers the face.
    """

    def __init__(self, confirm, grace):
        self.confirm = confirm
        self.grace = grace
        self.reset()

    def reset(self):
        self.started = None
        self.last_seen = None

    def update(self, touching, now):
        if self.last_seen is not None and now - self.last_seen > self.grace:
            self.reset()
        if not touching:
            return False
        if self.started is None:
            self.started = now
        self.last_seen = now
        return now - self.started >= self.confirm

    def held(self, now):
        return 0.0 if self.started is None else now - self.started


class FaceTouchingDetector:
    def __init__(self, dry_run=False, debug=False):
        self.dry_run = dry_run
        self.debug = debug

        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        self.hands = mp.solutions.hands.Hands(
            # Detect hands afresh every frame. Tracking reuses the previous
            # frame's hand position, which at 6 fps is stale, and with fewer than
            # max_num_hands in view the palm detector runs every frame anyway.
            static_image_mode=True,
            max_num_hands=2,
            # Full model, not lite: lite's hand scale was too noisy for the
            # depth check, and in tests it missed a palm covering the face.
            # Costs about twice the CPU while a hand is in view (~33 vs ~15 ms).
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.7)   # hand-presence score threshold

        self.debouncer = TouchDebouncer(TOUCH_CONFIRM_SECONDS, TOUCH_GRACE_SECONDS)
        self.face = None
        self.face_seen_at = -math.inf
        self.face_width = None  # face size from the last frame without a hand on it
        self.last_trigger = -math.inf

    def turn_off_screen(self):
        if not self.dry_run:
            subprocess.run(["pmset", "displaysleepnow"])

    def find_face(self, rgb, now):
        result = self.face_detection.process(rgb)
        if result.detections:
            height, width = rgb.shape[:2]
            # Use the largest detected face.
            self.face = max(
                (FaceRegion.from_detection(d, width, height) for d in result.detections),
                key=lambda face: face.width)
            self.face_seen_at = now
        elif now - self.face_seen_at > FACE_MEMORY_SECONDS:
            self.face = None
        return self.face

    def check_hands(self, rgb, face, face_width):
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []

        height, width = rgb.shape[:2]
        checks = []
        for landmarks, world in zip(result.multi_hand_landmarks,
                                    result.multi_hand_world_landmarks):
            points = np.array([(lm.x * width, lm.y * height) for lm in landmarks.landmark])
            world_xy = np.array([(lm.x, lm.y) for lm in world.landmark])
            inside = int(face.contains(points).sum())
            ratio = hand_depth_ratio(points, world_xy, face_width)
            touching = (inside >= MIN_POINTS_IN_FACE and
                        MIN_DEPTH_RATIO <= ratio <= MAX_DEPTH_RATIO)
            checks.append(HandCheck(points, inside, ratio, touching))
        return checks

    def process(self, rgb, now):
        """Returns the face region, per-hand checks and whether a touch is confirmed."""
        face = self.find_face(rgb, now)
        if face is None:
            # no face -> skip the expensive hand model entirely
            self.face_width = None
            self.debouncer.update(False, now)
            return None, [], False

        hands = self.check_hands(rgb, face, self.face_width or face.width)
        if not any(h.inside for h in hands):
            # A hand over the face shrinks or grows the face box by 10-25%, so
            # depth is measured against the size seen without one.
            self.face_width = face.width
        confirmed = self.debouncer.update(any(h.touching for h in hands), now)
        return face, hands, confirmed

    def on_touch(self, hands, now):
        if now - self.last_trigger <= TRIGGER_COOLDOWN:
            return
        self.last_trigger = now
        self.debouncer.reset()

        hand = next(h for h in hands if h.touching)
        print(f"{time.strftime('%H:%M:%S')} touch: {hand.inside}/21 points on face, "
              f"depth ratio {hand.depth_ratio:.2f}"
              f"{' (dry run)' if self.dry_run else ''}", flush=True)
        self.turn_off_screen()

    def show(self, frame, face, hands, now):
        """Debug view: face ellipse, hand landmarks and the numbers behind the decision."""
        if face is not None:
            cv2.ellipse(frame, (int(face.cx), int(face.cy)), (int(face.rx), int(face.ry)),
                        0, 0, 360, (255, 255, 0), 1)
        for hand in hands:
            if hand.touching:
                color = (0, 0, 255)       # red: counts as a touch
            elif hand.inside >= MIN_POINTS_IN_FACE:
                color = (0, 165, 255)     # orange: overlaps the face, rejected by depth
            else:
                color = (0, 200, 0)       # green: away from the face
            for x, y in hand.points.astype(int):
                cv2.circle(frame, (x, y), 2, color, -1)

        # Mirror like a selfie camera, then write the text.
        frame = cv2.flip(frame, 1)
        width = frame.shape[1]
        for hand in hands:
            x, y = hand.points[0].astype(int)  # wrist
            cv2.putText(frame, f"in {hand.inside} depth {hand.depth_ratio:.2f}",
                        (width - x - 60, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        status = "no face" if face is None else (
            f"touch {self.debouncer.held(now):.1f}/{TOUCH_CONFIRM_SECONDS:.1f}s")
        cv2.putText(frame, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("face-touching-detector (q to quit)", frame)
        return cv2.waitKey(1) & 0xFF != ord("q")

    def run(self):
        webcam = cv2.VideoCapture(0)
        # Ask the camera for a modest resolution so we move/decode fewer pixels.
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_interval = 1.0 / TARGET_FPS
        try:
            while True:
                start = time.monotonic()

                ok, frame = webcam.read()
                if not ok:
                    time.sleep(frame_interval)
                    continue

                scale = PROCESS_WIDTH / frame.shape[1]
                if scale < 1.0:
                    frame = cv2.resize(
                        frame, (PROCESS_WIDTH, int(frame.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False  # lets mediapipe avoid a copy
                face, hands, confirmed = self.process(rgb, start)
                if confirmed:
                    self.on_touch(hands, start)
                if self.debug and not self.show(frame, face, hands, start):
                    break

                # Throttle: sleep off the remainder of the frame budget so we
                # don't spin the CPU at 100%.
                elapsed = time.monotonic() - start
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
        except KeyboardInterrupt:
            pass
        finally:
            webcam.release()
            cv2.destroyAllWindows()
            self.face_detection.close()
            self.hands.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sleep the display when a hand touches the face.")
    parser.add_argument("--debug", action="store_true",
                        help="show the camera with the face region and per-hand scores")
    parser.add_argument("--dry-run", action="store_true",
                        help="print touches instead of sleeping the display")
    args = parser.parse_args()
    FaceTouchingDetector(dry_run=args.dry_run, debug=args.debug).run()
