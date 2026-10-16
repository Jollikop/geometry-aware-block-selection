# Variational SVGP campaign

The SVGP stack uses constant Adam and Gaussian natural gradients. It is kept
separate from the collapsed SGPR campaign and is not submitted by the
collapsed-only launcher.

Public methods are SVGP, T-SVGP, RBT-SVGP, and SBT-SVGP. Headline predictors
are sparse, diagonal, complete random blocks, and complete selected blocks.
