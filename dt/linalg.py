"""Minimal dense linear algebra (pure Python).

Only what the UKF and the Mahalanobis detector need: Cholesky with jitter,
triangular solves, symmetric inverse, and matrix helpers. Avoiding numpy keeps
the edge deployment to a bare CPython install.
"""

from __future__ import annotations

import math
from typing import List

Matrix = List[List[float]]
Vector = List[float]


def zeros(n: int, m: int) -> Matrix:
    return [[0.0] * m for _ in range(n)]


def eye(n: int, s: float = 1.0) -> Matrix:
    return [[s if i == j else 0.0 for j in range(n)] for i in range(n)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    n, k, m = len(a), len(b), len(b[0])
    out = zeros(n, m)
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for p in range(k):
            v = ai[p]
            if v == 0.0:
                continue
            bp = b[p]
            for j in range(m):
                oi[j] += v * bp[j]
    return out


def transpose(a: Matrix) -> Matrix:
    return [list(r) for r in zip(*a)]


def matvec(a: Matrix, x: Vector) -> Vector:
    return [sum(ai[j] * x[j] for j in range(len(x))) for ai in a]


def outer(x: Vector, y: Vector) -> Matrix:
    return [[xi * yj for yj in y] for xi in x]


def add(a: Matrix, b: Matrix) -> Matrix:
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def scale(a: Matrix, s: float) -> Matrix:
    return [[v * s for v in row] for row in a]


def symmetrise(a: Matrix) -> Matrix:
    n = len(a)
    return [[0.5 * (a[i][j] + a[j][i]) for j in range(n)] for i in range(n)]


def cholesky(a: Matrix, jitter: float = 1e-10) -> Matrix:
    """Lower-triangular Cholesky factor, retrying with growing jitter if the
    matrix has lost positive-definiteness through round-off."""
    n = len(a)
    for attempt in range(8):
        eps = jitter * (10.0 ** attempt)
        l = zeros(n, n)
        ok = True
        for i in range(n):
            for j in range(i + 1):
                s = sum(l[i][k] * l[j][k] for k in range(j))
                if i == j:
                    d = a[i][i] + eps - s
                    if d <= 0.0:
                        ok = False
                        break
                    l[i][j] = math.sqrt(d)
                else:
                    l[i][j] = (a[i][j] - s) / l[j][j]
            if not ok:
                break
        if ok:
            return l
    # last resort: diagonal approximation
    return [[math.sqrt(max(a[i][i], 1e-12)) if i == j else 0.0 for j in range(n)]
            for i in range(n)]


def solve_lower(l: Matrix, b: Vector) -> Vector:
    n = len(l)
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(l[i][k] * y[k] for k in range(i))) / l[i][i]
    return y


def solve_upper(u: Matrix, b: Vector) -> Vector:
    n = len(u)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(u[i][k] * x[k] for k in range(i + 1, n))) / u[i][i]
    return x


def spd_solve(a: Matrix, b: Vector) -> Vector:
    """Solve A x = b for symmetric positive-definite A."""
    l = cholesky(symmetrise(a))
    return solve_upper(transpose(l), solve_lower(l, b))


def spd_inverse(a: Matrix) -> Matrix:
    n = len(a)
    l = cholesky(symmetrise(a))
    lt = transpose(l)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(solve_upper(lt, solve_lower(l, e)))
    return transpose(cols)


def mahalanobis(x: Vector, mean: Vector, cov_inv: Matrix) -> float:
    d = [x[i] - mean[i] for i in range(len(x))]
    return math.sqrt(max(0.0, sum(d[i] * cov_inv[i][j] * d[j]
                                  for i in range(len(d)) for j in range(len(d)))))


def jacobi_eig(a: Matrix, iters: int = 100, tol: float = 1e-10):
    """Symmetric eigendecomposition by cyclic Jacobi rotations.

    Returns (eigenvalues, eigenvectors-as-columns) sorted descending. Adequate
    and numerically sound for the small covariance matrices used by the PCA
    monitor (tens of features), with no external dependency.
    """
    n = len(a)
    m = [row[:] for row in a]
    v = eye(n)
    for _ in range(iters):
        off = math.sqrt(sum(m[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(m[p][q]) < 1e-14:
                    continue
                theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p] = c * mkp - s * mkq
                    m[k][q] = s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = m[p][k], m[q][k]
                    m[p][k] = c * mpk - s * mqk
                    m[q][k] = s * mpk + c * mqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    vals = [m[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -vals[i])
    return [vals[i] for i in order], [[v[r][i] for i in order] for r in range(n)]
