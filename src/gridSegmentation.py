import numpy as np
import cv2
import math
from UnionFind import UnionFind


class Edge:
    def __init__(self, a=None, b=None, w=None):
        self.a = a
        self.b = b
        self.w = w


def L2(p1, p2):
    total = 0
    for i in range(3):
        total += (float(p1[i]) - float(p2[i])) ** 2
    return math.sqrt(total)


def create_edges(img):
    height, width, _ = img.shape
    edges = []
    dr = [-1, 0, 1, 0]
    dc = [0, 1, 0, -1]
    for x in range(0, height):
        for y in range(0, width):
            for k in range(0, 4):
                tx = x + dr[k]
                ty = y + dc[k]
                if 0 <= tx < height and 0 <= ty < width:
                    a = x * width + y
                    b = tx * width + ty
                    w = L2(img[x, y], img[tx, ty])
                    edges.append(Edge(a, b, w))

    return edges


def find_segments(v, edges, k):
    edges.sort(key=lambda edge: edge.w)
    segments = UnionFind(v)
    thresh = k * np.ones(v)
    for edge in edges:
        a = segments.find(edge.a)
        b = segments.find(edge.b)
        if a != b:
            if edge.w <= thresh[a] and edge.w <= thresh[b]:
                segments.union(a, b)
                a = segments.find(a)
                thresh[a] = edge.w + k / segments.csize[a]

    return segments


def merge_small_components(segments, edges, min_size):
    for edge in edges:
        a = segments.find(edge.a)
        b = segments.find(edge.b)
        if a != b and (segments.csize[a] < min_size or
                       segments.csize[b] < min_size):
            segments.union(a, b)

    return segments


def color_segments(img, segments):
    colored_image = img.copy()
    height, width, _ = img.shape
    colors = np.random.random((height * width, 3)) * 255.0
    for x in range(0, height):
        for y in range(0, width):
            segment_id = segments.find(x * width + y)
            colored_image[x, y] = colors[segment_id]

    return colored_image

def image_segmentation_3(
        img, sigma=0.8, kernel_size=5, min_size=20, k=300):
    height, width, _ = img.shape
    smooth_image = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    edges = create_edges(smooth_image)
    segments = find_segments(width * height, edges, k)
    segments = merge_small_components(segments, edges, min_size)
    output_image = color_segments(smooth_image, segments)

    return segments, output_image
