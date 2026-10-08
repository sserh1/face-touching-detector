import unittest
from types import SimpleNamespace

import numpy as np

from optimized import FaceRegion, TouchDebouncer, hand_depth_ratio


def detection(xmin, ymin, width, height):
    box = SimpleNamespace(xmin=xmin, ymin=ymin, width=width, height=height)
    return SimpleNamespace(location_data=SimpleNamespace(relative_bounding_box=box))


class FaceRegionTest(unittest.TestCase):
    def test_covers_forehead_to_chin_but_not_hair_or_neck(self):
        # 100x100 px face box at (200, 100) in a 480x360 frame
        face = FaceRegion.from_detection(detection(200 / 480, 100 / 360, 100 / 480, 100 / 360), 480, 360)
        inside = face.contains(np.array([
            [250, 150],  # centre of the face
            [250, 85],   # forehead, above the detection box
            [250, 205],  # chin
            [205, 105],  # top-left corner of the box: hair
            [295, 200],  # bottom-right corner of the box: jaw/neck
            [250, 230],  # neck, below the chin
        ]))
        self.assertEqual(inside.tolist(), [True, True, True, False, False, False])


class HandDepthRatioTest(unittest.TestCase):
    world = np.array([[0, 0], [0.02, -0.08], [0.05, -0.17], [-0.03, -0.15], [0.07, -0.05]])
    face_width = 140  # px; a 0.14 m face box -> 1000 px/m at the face's depth

    def test_hand_at_face_depth(self):
        points = self.world * 1000 + [240, 180]
        self.assertAlmostEqual(hand_depth_ratio(points, self.world, self.face_width), 1.0)

    def test_hand_twice_as_close_looks_twice_as_big(self):
        points = self.world * 2000 + [240, 180]
        self.assertAlmostEqual(hand_depth_ratio(points, self.world, self.face_width), 2.0)

    def test_ignores_hand_rotation_in_the_image(self):
        a = np.radians(40)
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        points = self.world @ rotation.T * 1000 + [240, 180]
        self.assertAlmostEqual(hand_depth_ratio(points, self.world, self.face_width), 1.0)


class TouchDebouncerTest(unittest.TestCase):
    def feed(self, flags, fps=6):
        debouncer = TouchDebouncer(confirm=0.5, grace=0.4)
        return [debouncer.update(bool(flag), i / fps) for i, flag in enumerate(flags)]

    def test_confirms_a_sustained_touch(self):
        self.assertEqual(self.feed([1, 1, 1, 1]), [False, False, False, True])

    def test_ignores_brief_contact(self):
        self.assertFalse(any(self.feed([1, 1, 1, 0, 0, 0, 1, 1])))

    def test_tolerates_one_dropped_frame(self):
        self.assertEqual(self.feed([1, 1, 0, 1, 1]), [False, False, False, True, True])

    def test_restarts_after_two_dropped_frames(self):
        self.assertFalse(any(self.feed([1, 1, 0, 0, 1, 1, 1])))


if __name__ == "__main__":
    unittest.main()
