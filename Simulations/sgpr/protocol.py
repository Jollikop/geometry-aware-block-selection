"""Release-critical collapsed-SGPR optimisation protocol.

This module is intentionally JAX-free so launchers, table builders and final
suite validators can import one contract without initialising an accelerator.
"""

from __future__ import annotations


# v23 first obtains a robust inducing-point start through the deterministic
# variational-EM warm-up below.  The fitted residual structure may still move
# four times at the start of the release solve, at cumulative budgets 15, 30,
# 50 and 75.  It is then frozen for one continuous joint solve.  The last
# entry is a ceiling: a certified run normally stops before exhausting it.
LBFGS_REFRESH_SCHEDULE = (15, 15, 20, 25, 99925)
LBFGS_STEPS = sum(LBFGS_REFRESH_SCHEDULE)

# Inducing-point protocol.  Greedy conditional-variance selection is the
# primary, nested initialisation: every M uses a prefix of the M=512 selection
# on the same split.  A k-means++ data-point seeding is a single predeclared
# fallback, attempted only when the primary fit fails the final certificate.
INITIALIZATION_POLICY = ("greedy-variance", "kmeans++")
INITIALIZATION_NESTED_MAX_M = 512

# Up to three variational-EM-style warm-up cycles are performed before the
# release solve.  Each cycle optimises theta/noise with Z fixed, then refreshes
# Z using greedy conditional variance under the updated kernel.  Reselection
# stops when it buys less than the paper-scale per-datum tolerance.
WARMUP_MAX_CYCLES = 3
WARMUP_THETA_STEPS = 1000
WARMUP_RESELECT_OBJ_N_TOL = 1e-4

# The final fixed-partition phase is one continuous SciPy solve.  The windows
# below are diagnostics on that single trajectory: they must never restart
# L-BFGS and discard its inverse-Hessian history.  A release run needs two
# consecutive flat objective windows and a small full gradient at the returned
# point.  These tolerances are tighter than the downstream null-refinement
# effects while remaining meaningful for the dense E-SGPR objective at N=5000.
# Thirty thousand is a hard ceiling, not a required count: the continuous final
# solve stops as soon as the convergence certificate passes.
CONVERGENCE_WINDOW_STEPS = 100
CONVERGENCE_OBJ_N_TOL = 1e-4
# L-BFGS-B itself uses the infinity norm of the projected gradient.  Applying
# the same dimension-independent norm to loss/N keeps M=128 and M=512 under
# one meaningful gate.  L2/N remains recorded as a diagnostic only.
CONVERGENCE_GRAD_MAX_N_TOL = 1e-3
CONVERGENCE_PATIENCE = 2

# Exact SciPy solver controls, recorded in every release fingerprint.  The
# reviewer-facing decision remains the per-datum checks above; these only bound
# the internal line search and numerical relative-reduction stop.
# Do not let SciPy's scale-dependent relative objective test pre-empt the
# reviewer-facing per-datum tests above.  The explicit maximum iteration count
# and the external convergence gate own termination.
SCIPY_LBFGSB_FTOL = 0.0
SCIPY_LBFGSB_GTOL = 0.0
SCIPY_LBFGSB_MAXCOR = 100
SCIPY_LBFGSB_MAXFUN = 250000
SCIPY_LBFGSB_MAXLS = 100

# The transplant is a deliberately stringent null test after the donors have
# passed the main convergence gate.  SGPR should barely move under this extra
# Z-only refinement if its official fit is stationary with respect to Z.
TRANSPLANT_STEPS = 100000
TRANSPLANT_NULL_ELBO_TOL = 5e-4
TRANSPLANT_NULL_NLL_TOL = 1e-3
