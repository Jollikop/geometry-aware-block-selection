"""Code shared by the collapsed (``sgpr``) and variational (``svgp``) stacks.

| Module                     | Contents                                          |
|----------------------------|---------------------------------------------------|
| ``jax_config``             | enables float64 in JAX, once                      |
| ``provenance``             | data hashes, git state, package and GPU manifests |
| ``datasets``               | declared sources, caches, provenance sidecars     |
| ``datasets_uci``           | raw UCI archive download and parsing              |
| ``data_split``             | deterministic split, train-only standardisation   |
| ``kernels``                | JAX kernels and the canonical name registry       |
| ``kernels_nonstationary``  | the Gibbs kernel and its warp parameterisation    |
| ``terms``                  | the packed ``(coeffs, idx, mask)`` block cover    |
| ``partitions``             | BT and SBT block selection, host and device       |
| ``predictors``             | local-cross and full-cross residual corrections   |
| ``results_io``             | per-run JSON and parallel-safe result CSVs        |
| ``stats``                  | error bars and mean/std formatting                |
| ``plot_style``             | method colours, markers and figure defaults       |

Project rule: a stack imports shared behaviour from here; no private copies.
"""

__all__ = [
    "data_split", "datasets", "datasets_uci", "jax_config", "kernels",
    "kernels_nonstationary", "partitions", "plot_style", "predictors",
    "provenance", "results_io", "stats", "terms",
]
__version__ = "5.0.0"
