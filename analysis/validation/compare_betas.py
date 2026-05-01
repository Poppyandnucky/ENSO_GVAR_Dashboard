# analysis/validation/compare_betas.py
#
# Utilities for comparing true and estimated coefficient paths.

import numpy as np
import matplotlib.pyplot as plt

def extract_A1_path(beta_path, p, k):
    """beta_path: (T_eff, m, k) -> A1_path: (T_eff, k, k)"""
    T_eff = beta_path.shape[0]
    A1_path = np.zeros((T_eff, k, k))
    for t in range(T_eff):
        A_blocks = extract_A_blocks_from_beta(beta_path[t], p, k)
        A1_path[t] = A_blocks[0]
    return A1_path

def plot_A1_tracking(results, entry=(0, 0)):
    """
    Plot one A1 coefficient over time.
    Works for both simulated data (with truth) and real data (no truth).
    """
    import matplotlib.pyplot as plt

    i, j = entry
    p = results["p"]
    k = results["k"]

    # Extract TVP path
    A1_path = extract_A1_path(results["beta_tvp_path"], p, k)
    series = A1_path[:, i, j]

    plt.figure(figsize=(9, 4))
    plt.plot(series, label=f"TVP A1[{i},{j}]")

    # Fixed VAR (always available)
    if "A_blocks_var" in results and results["A_blocks_var"] is not None:
        A1_var = results["A_blocks_var"][0][i, j]
        plt.axhline(A1_var, color="b", linestyle="--", label="fixed VAR")

    # True values (only for simulated data)
    if results.get("true") is not None:
        t_break = results["true"]["break_index"]
        A1_true_pre = results["true"]["A1_pre"][i, j]
        A1_true_post = results["true"]["A1_post"][i, j]

        plt.axvline(t_break - p, color="gray", linestyle=":", label="break (approx)")
        plt.axhline(A1_true_pre, color="k", linestyle="--", label="true pre")
        plt.axhline(A1_true_post, color="r", linestyle="--", label="true post")

    plt.title(f"A1[{i},{j}] tracking")
    plt.legend()
    plt.tight_layout()
    plt.show()

def summarize_pre_post(results):
    """
    Compare average estimated A1 pre/post break to true values.
    Only applicable for simulated data.
    """
    if results.get("true") is None:
        print("No true break information available (real data). Skipping pre/post summary.")
        return

    p = results["p"]
    k = results["k"]
    t_break = results["true"]["break_index"]

    A1_path = extract_A1_path(results["beta_tvp_path"], p, k)

    # beta_path index starts at t=p in original time
    tb = max(0, t_break - p)

    A1_pre_hat = A1_path[:tb].mean(axis=0)
    A1_post_hat = A1_path[tb:].mean(axis=0)

    print("=== True A1 pre ===\n", results["true"]["A1_pre"])
    print("=== True A1 post ===\n", results["true"]["A1_post"])
    print("=== TVP mean A1 pre ===\n", A1_pre_hat)
    print("=== TVP mean A1 post ===\n", A1_post_hat)
    print("=== Fixed VAR A1 ===\n", results["A_blocks_var"][0])

def extract_A_blocks_from_beta(beta_mat, p, k):
    """
    Extract VAR lag matrices A1, A2, ..., Ap from TVP beta matrix.

    beta_mat : (m, k) where m = p*k + q
    returns  : A_blocks (p, k, k)
    """
    A_blocks = np.zeros((p, k, k))

    for lag in range(p):
        rows = slice(lag * k, (lag + 1) * k)
        A_blocks[lag] = beta_mat[rows, :].T

    # returns A with rows = equations
    return A_blocks

def plot_beta_comparison(
    A1_est,
    A1_true_pre,
    A1_true_post,
    break_index,
):
    """
    Plot estimated vs true A1 coefficients.
    """
    T_eff = A1_est.shape[0]

    plt.figure(figsize=(8, 4))
    plt.plot(A1_est[:, 0, 0], label="Estimated A1[0,0]")
    plt.axhline(A1_true_pre[0, 0], color="k", linestyle="--", label="True pre-break")
    plt.axhline(A1_true_post[0, 0], color="r", linestyle="--", label="True post-break")
    plt.axvline(break_index, color="gray", linestyle=":")
    plt.legend()
    plt.title("TVP-VAR coefficient tracking")
    plt.tight_layout()
    plt.show()

def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def extract_A_blocks(beta_mat, p, k):
    """
    Extract A1, A2, ... from beta matrix.
    """
    A_blocks = beta_mat[: p * k].reshape(p, k, k, order="F")
    return A_blocks

def compare_convergence(results, p=2, k=2):
    A1_true = results["true"]["A1"]
    A2_true = results["true"]["A2"]

    # Fixed VAR (already in correct shape)
    A_blocks_var = results["A_blocks_var"]
    A1_var = A_blocks_var[0]
    A2_var = A_blocks_var[1]

    # TVP-VAR (needs extraction)
    beta_mat_final = results["beta_tvp_path"][-1]
    # take A_blocks, taking transform after extracting from beta_mat_final
    #   each row is an equation
    A_blocks_tvp_final = extract_A_blocks_from_beta(beta_mat_final, p, k)
    A1_tvp = A_blocks_tvp_final[0]
    A2_tvp = A_blocks_tvp_final[1] # not sure if this exists for p=1

    print("=== A1 comparison ===")
    print("True A1:\n", A1_true)
    print("VAR A1:\n", A1_var)
    print("TVP A1 (final):\n", A1_tvp)

    print("\n=== A2 comparison ===")
    print("True A2:\n", A2_true)
    print("VAR A2:\n", A2_var)
    print("TVP A2 (final):\n", A2_tvp)

    import matplotlib.pyplot as plt

    A1_path = results["beta_tvp_path"][:, :k, :].reshape(-1, k, k, order="F")

    plt.plot(A1_path[:, 0, 0], label="TVP A1[0,0]")
    plt.axhline(results["true"]["A1"][0, 0], color="k", linestyle="--", label="True")
    plt.axhline(A_blocks_var[0][0, 0], color="r", linestyle="--", label="VAR")
    plt.legend()
    plt.grid()
    plt.show()
