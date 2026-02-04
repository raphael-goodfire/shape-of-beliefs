import numpy as np

from scipy.spatial.distance import cdist  # or roll your own

def pairwise_distance(probs, mode="hellinger", eps=1e-12):
    if mode == "hellinger":
        sqrtP = np.sqrt(probs)
        cross = sqrtP @ sqrtP.T  # <sqrt(p_i), sqrt(p_j)>
        return 1.0 - cross  # Hellinger^2
    elif mode == "l2":
        # squared L2
        return cdist(probs, probs, metric="sqeuclidean")
    elif mode == "js":
        # Jensen–Shannon (metric). scipy’s jensenshannon returns sqrt(JS); square it.
        from scipy.spatial.distance import jensenshannon
        m = probs.shape[0]
        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i+1, m):
                d = jensenshannon(probs[i], probs[j])
                D[i, j] = D[j, i] = d * d
        return D
    elif mode == "cosine":
        return cdist(probs, probs, metric="cosine")
    else:
        raise ValueError(f"Unknown mode {mode}")

def inpca_embedding(prob_matrix: np.ndarray, dim: int = 3, eps: float = 1e-12, mode: str = "hellinger"):
    """
    prob_matrix: [m, k] rows are probability vectors (will be renormalized).
    dim: target embedding dimension.
    eps: tolerance for positive eigenvalues.
    Returns (coords [m, dim_eff], explained_variance [dim_eff]).
    """
    probs = np.asarray(prob_matrix, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)    # # safety normalize

    D = pairwise_distance(probs, mode=mode, eps=eps)
    m = D.shape[0]
    J = np.eye(m) - np.ones((m, m)) / m
    W = -0.5 * J @ D @ J                 # double-centering (MDS)

    eigvals, eigvecs = np.linalg.eigh(W)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    pos = eigvals > eps
    if not pos.any():
        raise ValueError("InPCA: no positive eigenvalues—check data or reduce eps")

    dim_eff = min(dim, pos.sum())        # filters out non-positive eigenvalues
    eigvals = eigvals[pos][:dim_eff]
    eigvecs = eigvecs[:, pos][:, :dim_eff]

    coords = eigvecs * np.sqrt(eigvals)  # U Σ^{1/2}
    explained = eigvals / eigvals.sum()
    return coords, explained
