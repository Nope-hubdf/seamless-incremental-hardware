import math
import numpy as np

SEED = 0
PERM = None

UNIT = 1.0 / math.sqrt(2)
VECTORS = np.array(
    [
        [UNIT, UNIT],
        [-UNIT, UNIT],
        [UNIT, -UNIT],
        [-UNIT, -UNIT],
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
    ]
)


def lerp(t, a, b):
    # Linear interpolation
    return a + t * (b - a)


def fade(t):
    # 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10)


def gradient(h, x, y):
    return x * VECTORS[h & 7][0] + y * VECTORS[h & 7][1]


def hash(x0, x1, y0, y1):
    h0 = PERM[x0]
    h1 = PERM[x1]

    h00 = PERM[h0 + y0]
    h10 = PERM[h1 + y0]
    h01 = PERM[h0 + y1]
    h11 = PERM[h1 + y1]

    return PERM[h00], PERM[h10], PERM[h01], PERM[h11]


def perlin(x, y):
    # Determine grid cell coordinates
    x0 = int(x) & 255
    x1 = (x0 + 1) & 255

    y0 = int(y) & 255
    y1 = (y0 + 1) & 255

    # Internal coordinates
    xf, yf = x - int(x), y - int(y)

    # Fade factors
    u, v = fade(xf), fade(yf)

    # 4 hashed gradient indices
    h00, h10, h01, h11 = hash(x0, x1, y0, y1)

    # Noise components
    n00 = gradient(h00, xf, yf)
    n10 = gradient(h10, xf - 1, yf)
    n01 = gradient(h01, xf, yf - 1)
    n11 = gradient(h11, xf - 1, yf - 1)

    # Combine noises
    n0 = lerp(u, n00, n10)
    n1 = lerp(u, n01, n11)
    return lerp(v, n0, n1)


def get_noise_value(
    x, y, scale=16, octaves=1, lacunarity=2.0, persistence=0.5, offset=0, seed=0
):
    global SEED
    global PERM

    if PERM is None or SEED != seed:
        SEED = seed
        np.random.seed(seed)
        # Permutation table
        PERM = np.arange(256, dtype=int)
        np.random.shuffle(PERM)
        PERM = np.stack([PERM, PERM]).flatten()

    frequency = 1
    amplitude = 1
    max = 0
    total = 0
    for _ in range(octaves):
        total += (
            perlin(
                ((x + offset) / scale) * frequency,
                ((y + offset) / scale) * frequency,
            )
            * amplitude
        )
        max += amplitude
        frequency *= lacunarity
        amplitude *= persistence

    return total / max
