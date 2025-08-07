# Modified from scikit-image's skimage/morphology/_skeletonize_various_cy.pyx:_fast_skeletonize function
# Original copyright © 2009-2022 the scikit-image team
# License: BSD-3-Clause (see custom_skeletonization/LICENSE_BSD3.txt)
# Additional modifications © 2025 Heejun Kim

import numpy as np
from collections import deque

def find(x, parent):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b, parent):
    ra, rb = find(a, parent), find(b, parent)
    if ra != rb:
        parent[rb] = ra

def find_closest_survivor(idx, cleaned_skel, W):
    """
    Given a flat index idx in an HxW grid, and a binary matrix
    cleaned_skel of shape (H, W) marking which pixels remain,
    do a BFS outward from idx to find the nearest location
    (in Manhattan distance) where cleaned_skel[r,c]==True.
    Return that flat index.
    """
    H, W = cleaned_skel.shape
    r0, c0 = divmod(idx, W)
    if cleaned_skel[r0, c0]:
        return idx

    q = deque([(r0, c0)])
    seen = { (r0, c0) }
    # 4‐neighborhood is enough for distance
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        pass
    while q:
        r, c = q.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),
                       (-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and (nr,nc) not in seen:
                if cleaned_skel[nr, nc]:
                    return nr*W + nc
                seen.add((nr,nc))
                q.append((nr,nc))
    # fallback (shouldn’t happen unless the skeleton is empty)
    return idx

def skeletonize_custom(image):
    """Optimized parts of the Zhang-Suen [1]_ skeletonization.
    Iteratively, pixels meeting removal criteria are removed,
    till only the skeleton remains (that is, no further removable pixel
    was found).

    Performs a hard-coded correlation to assign every neighborhood of 8 a
    unique number, which in turn is used in conjunction with a look up
    table to select the appropriate thinning criteria.

    Parameters
    ----------
    image : numpy.ndarray
        A binary image containing the objects to be skeletonized. '1'
        represents foreground, and '0' represents background.

    Returns
    -------
    skeleton : ndarray
        A matrix containing the thinned image.

    References
    ----------
    .. [1] A fast parallel algorithm for thinning digital patterns,
           T. Y. Zhang and C. Y. Suen, Communications of the ACM,
           March 1984, Volume 27, Number 3.

    """

    # look up table - there is one entry for each of the 2^8=256 possible
    # combinations of 8 binary neighbors. 1's, 2's and 3's are candidates
    # for removal at each iteration of the algorithm.
    lut = np.array([0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0,
            1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0,
            3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            2, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
            0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0,
            3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0,
            2, 0, 0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3,
            0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0,
            2, 0, 0, 0], dtype=np.uint8)

    nrows, ncols = image.shape[:2]
    nrows += 2
    ncols += 2

    # Copy over the image into a larger version with a single pixel border
    # this removes the need to handle border cases below

    skeleton = np.zeros((nrows, ncols), dtype=np.uint8)
    skeleton[1:-1, 1:-1] = image
    cleaned_skeleton = skeleton.copy()

    closest_skeleton_indices = dict()

    H, W = nrows - 2, ncols - 2
    N = H*W
    # replace label_map entirely with a true UF parent
    parent = - np.ones(N, dtype=np.int32)
    

    pixel_removed = True

    # the algorithm reiterates the thinning till
    # no further thinning occurred (variable pixel_removed set)
    while pixel_removed:
        pixel_removed = False

        # there are two phases, in the first phase, pixels labeled
        # (see below) 1 and 3 are removed, in the second 2 and 3

        # nogil can't iterate through (True, False) because it is a Python
        # tuple. Use the fact that 0 is Falsy, and 1 is truthy in C
        # for the iteration instead.
        # for first_pass in (True, False):
        for pass_num in range(2):
            first_pass = (pass_num == 0)
            removed_pixels = []
            for row in range(1, nrows-1):
                for col in range(1, ncols-1):
                    # all set pixels ...

                    if skeleton[row, col]:
                        # are correlated with a kernel
                        # (coefficients spread around here ...)
                        # to apply a unique number to every
                        # possible neighborhood ...

                        # which is used with the lut to find the
                        # "connectivity type"

                        neighbors = lut[skeleton[row - 1, col - 1] +
                                        2 * skeleton[row - 1, col] +
                                        4 * skeleton[row - 1, col + 1] +
                                        8 * skeleton[row, col + 1] +
                                        16 * skeleton[row + 1, col + 1] +
                                        32 * skeleton[row + 1, col] +
                                        64 * skeleton[row + 1, col - 1] +
                                        128 * skeleton[row, col - 1]]

                        if (neighbors == 0):
                            continue
                        elif ((neighbors == 3) or
                                (neighbors == 1 and first_pass) or
                                (neighbors == 2 and not first_pass)):
                            # Remove the pixel
                            cleaned_skeleton[row, col] = 0
                            removed_pixels.append((row, col))
                            pixel_removed = True

            inner_cleaned_skeleton = cleaned_skeleton[1:-1, 1:-1]

            for row, col in removed_pixels:
                idx = (row - 1) * W + (col - 1)
                # find the closest survivor
                closest_survivor = find_closest_survivor(idx, inner_cleaned_skeleton, W)
                parent[idx] = closest_survivor
                mask = (parent == idx)
                parent[mask] = closest_survivor

            # once a step has been processed, the original skeleton
            # is overwritten with the cleaned version
            skeleton[:, :] = cleaned_skeleton[:, :]
    inner = cleaned_skeleton[1:-1,1:-1].astype(bool)
    flat_inner = inner.ravel()
    for idx in np.nonzero(flat_inner)[0]:
        parent[idx] = idx

    return skeleton[1:-1, 1:-1].astype(bool), parent.reshape(H, W)