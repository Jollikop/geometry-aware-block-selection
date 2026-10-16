"""Variational sparse-GP stack: SVGP, T-SVGP, RBT-SVGP, SBT-SVGP.

q(u) is kept explicit and optimised by natural gradient, so training is
minibatched and the methods differ only in the residual correction applied to
q(f|u): trace, diagonal, random block, or selected block.

| Module            | Contents                                               |
|-------------------|--------------------------------------------------------|
| ``engine``        | objectives, optimisers, banks and predictors           |
| ``plans``         | campaign plans and per-kernel learning-rate profiles   |
| ``run_benchmark`` | the official grid driver                               |
| ``analysis``      | campaign validators and post-hoc ablations             |
"""

__all__ = ["engine", "plans"]
