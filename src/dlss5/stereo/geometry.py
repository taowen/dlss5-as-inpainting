"""Reference rectified stereo geometry with explicit coverage and source coordinates."""

import numpy as np


def splat(color, disparity):
    """Integer reference renderer: larger disparity is nearer; holes have no MV."""
    h, w = disparity.shape
    warped = np.zeros_like(color)
    z = np.full((h, w), -np.inf, np.float32)
    source_x = np.full((h, w), -1, np.int32)
    for y in range(h):
        for x in range(w):
            target = int(np.rint(x - disparity[y, x]))
            if 0 <= target < w and disparity[y, x] >= z[y, target]:
                warped[y, target] = color[y, x]
                z[y, target] = disparity[y, x]
                source_x[y, target] = x
    valid = source_x >= 0
    motion = np.zeros((h, w, 2), np.float32)
    motion[..., 0] = np.where(valid, source_x - np.arange(w), 0)
    return warped, valid, z, source_x, motion


def fill_scanlines(warped, valid, disparity, background_only=False):
    """Fill gaps using nearest samples, or the farther of their two boundary surfaces."""
    result = warped.copy()
    for y in range(valid.shape[0]):
        known = np.flatnonzero(valid[y])
        if not known.size:
            raise ValueError("scanline has no observed samples")
        for x in np.flatnonzero(~valid[y]):
            pos = np.searchsorted(known, x)
            candidates = known[max(0, pos - 1):min(len(known), pos + 1)]
            if background_only:
                # Smaller disparity means background; no hidden reference is accessed.
                chosen = min(candidates, key=lambda p: (disparity[y, p], abs(p - x)))
            else:
                chosen = min(candidates, key=lambda p: abs(p - x))
            result[y, x] = warped[y, chosen]
    return result
