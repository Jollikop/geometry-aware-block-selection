"""Collapsed sparse-GP stack: SGPR, T-SGPR, RBT-SGPR, SBT-SGPR.

The marginal likelihood is collapsed analytically over q(u), so training
optimises hyperparameters and inducing points directly and the methods differ
only in how they bound the residual log-determinant.

| Module                  | Contents                                          |
|-------------------------|---------------------------------------------------|
| ``gpflow_bounds``       | TensorFlow bounds used by the Snelson figure      |
| ``gpflow_training``     | L-BFGS-B loop with structure refreshes            |
| ``jax_collapsed``       | JAX losses, training and predictors               |
| ``benchmark``           | the benchmark core both JAX runners call          |
| ``run_benchmark``       | the official Kin40k table driver                  |
| ``snelson``             | the small GPflow reference figure                 |
| ``residual_illustration`` | the residual-structure motivation figure        |
"""

__all__ = ["benchmark", "gpflow_bounds", "gpflow_training", "jax_collapsed"]
