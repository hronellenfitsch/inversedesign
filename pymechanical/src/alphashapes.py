#!/usr/bin/env python3

""" alphashapes.py

Contains functions to calculate alpha shapes (or concave hulls)
With code adapted from https://sgillies.net/2012/10/13/the-fading-shape-of-alpha.html
"""

import numpy as np
import scipy as sp
from scipy.spatial import Delaunay

from collections import defaultdict

def concave_hull(points, alpha):
    """ Return the concave hull of the set of points in 2D for given value
    of alpha.
    """
    tri = Delaunay(np.array(points))

    def add_edge(i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        i, j = sorted([i, j])
        edges[(i, j)] += 1

    edges = defaultdict(int)

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

        # Lengths of sides of triangle
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        area = np.sqrt(s*(s-a)*(s-b)*(s-c))

        # radius of the circumcircle
        circum_r = a*b*c/(4.0*area)

        # Add edges to the filtered triangulation
        if circum_r < 1.0/alpha:
            add_edge(ia, ib)
            add_edge(ib, ic)
            add_edge(ic, ia)

    # find the concave hull, which is the set of edges that appear
    # in only one triangle/only once in the edge list

    hull_idx = np.unique(np.array([k for k, v in edges.items() if v == 1], dtype=int).flatten())
    return hull_idx
