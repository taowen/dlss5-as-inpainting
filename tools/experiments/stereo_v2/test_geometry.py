"""Geometry contract tests independent of native rendering."""

import unittest
import numpy as np
from dlss5.stereo.geometry import splat, fill_scanlines


class GeometryTest(unittest.TestCase):
    def test_zero_disparity_preserves_every_pixel(self):
        color = np.random.default_rng(12).random((8,12,3),dtype=np.float32)
        warped, valid, _, _, mv = splat(color,np.zeros((8,12),np.float32))
        np.testing.assert_array_equal(color,warped)
        self.assertTrue(valid.all())
        self.assertFalse(mv.any())

    def test_occlusion_and_backward_motion(self):
        color = np.full((4,20,3),.2,np.float32)
        color[:,8:12] = .8
        depth = np.full((4,20),2,np.float32)
        depth[:,8:12] = 6
        warped, valid, z, sx, mv = splat(color,depth)
        # Near rectangle lands at [2,6); exposed background is [6,10).
        self.assertFalse(valid[:,6:10].any())
        np.testing.assert_array_equal(warped[:,2:6],np.full((4,4,3),.8,np.float32))
        np.testing.assert_array_equal((np.arange(20)+mv[...,0])[valid],sx[valid])
        filled = fill_scanlines(warped,valid,z,True)
        np.testing.assert_array_equal(filled[:,6:10],np.full((4,4,3),.2,np.float32))
        np.testing.assert_array_equal(filled[valid],warped[valid])


if __name__ == "__main__":
    unittest.main()
