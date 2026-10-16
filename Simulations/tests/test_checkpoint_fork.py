#!/usr/bin/env python3
"""Legacy checkpoint continuation remains deterministic.

The official Gibbs campaign starts cold and does not use this path. Checkpoint
continuation is retained for diagnostics, where the comparison is meaningful
only if the branch is a genuine continuation:

1. a same-kernel resume must reproduce the continuous run exactly -- otherwise
   the step counter, the momentum or the minibatch sequence is not carried
   over and the branch is a different experiment;
2. a gated Gibbs kernel equals the stationary kernel of lengthscale
   1.5 * ell, so log_ell must be re-expressed on load; copying it verbatim
   starts the branch where the stationary run never was;
3. the checkpoint must restore optax's step counter, so an optional scheduled
   learning rate continues instead of restarting.

Runs on CPU in a few seconds and is picked up by run_submission_checks.sh.
"""
from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

# Set before jax is imported: the checks must run identically on a login node
# with no GPU and on a compute node with one.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parent.parent

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from common import kernels_nonstationary as jns  # noqa: E402
from common import kernels as kn  # noqa: E402
from svgp import engine as sv  # noqa: E402

N, D, N_TEST, M, BLOCK, EPOCHS = 200, 3, 50, 8, 50, 20

_RNG = np.random.default_rng(0)
X_TR = _RNG.normal(size=(N, D))
Y_TR = np.sin(X_TR[:, 0]) + 0.1 * _RNG.normal(size=N)
X_TE = _RNG.normal(size=(N_TEST, D))
Y_TE = np.sin(X_TE[:, 0])


def make_args(**over):
    args = types.SimpleNamespace(
        lr=3e-3, z_lr=3e-3, natural_q=True, natural_lr=0.05,
        normalize_objective=True, grad_clip=0.0, track_grad_stats=False,
        epochs=EPOCHS, blocks_per_step=2, log_every=5, refresh_every=5,
        refresh_until=10, importance_refresh_every=5, importance_epsilon=0.05,
        warp_gate_fraction=0.6, swa_fraction=0.0, batch_coupling="split",
        penalty_sampling="gain", common_ll_batches=True, full_batch=False,
        optimizer="lion", seed=0, live=False, evaluate_during_training=False,
        record_history=False, checkpoint_at_fraction=0.0,
        phases=None, metrics_at_epochs=None,
        block=BLOCK, ll_batch_size=BLOCK,
    )
    for key, value in over.items():
        setattr(args, key, value)
    return args


def train(method, kernel, args, params=None, start_epoch=0, leaves=None,
          checkpoint_path=None):
    engine = sv.make_engine(X_TR, Y_TR, X_TE, Y_TE, jitter=1e-8,
                            kernel_name=kernel)
    if params is None:
        params = sv.init_params(X_TR, M, 0, kernel_name=kernel)

    def make_bank(current):
        return sv.build_method_bank(
            method, jnp.asarray(X_TR), current, block=args.block, seed=0,
            jitter=1e-8, kernel_name=kernel, ll_batch_size=args.ll_batch_size)

    def on_checkpoint(epoch, current, opt_state):
        if checkpoint_path:
            sv.save_training_checkpoint(
                checkpoint_path, epoch, current, opt_state,
                meta={"total_epochs": args.epochs})

    trained, _, info, _ = sv.train_svgp(
        engine, method, params, make_bank, engine["metrics"], N, args,
        checkpoint_fn=on_checkpoint, start_epoch=start_epoch,
        resume_optimizer_leaves=leaves)
    return trained, info, engine


def gated_gibbs_params(stationary):
    """Re-express stationary parameters in the gated Gibbs parameterisation."""
    gated = dict(stationary)
    gated["warp"] = jnp.zeros((D, D))
    gated["log_ell"] = jns.stationary_to_gated_log_ell(gated["log_ell"])
    return gated


class GatedKernelIdentity(unittest.TestCase):
    """With W = 0 the Gibbs kernel is the stationary kernel of scale 1.5*ell."""

    def test_shift_makes_the_kernels_identical(self):
        x = jnp.asarray(_RNG.normal(size=(60, D)))
        log_ell = jnp.asarray(_RNG.normal(size=D) * 0.3)
        sf2, warp0 = jnp.asarray(1.7), jnp.zeros((D, D))
        stationary = kn.kernel(x, x, jnp.exp(log_ell), sf2, "rbf")
        shifted = kn.kernel(
            x, x, jnp.exp(jns.stationary_to_gated_log_ell(log_ell)), sf2,
            "gibbs", warp0)
        self.assertLess(float(jnp.max(jnp.abs(stationary - shifted))), 1e-12)

    def test_without_the_shift_they_are_not(self):
        # Guards against someone "simplifying" the re-expression away: the
        # error is half the kernel amplitude, not a rounding difference.
        x = jnp.asarray(_RNG.normal(size=(60, D)))
        log_ell = jnp.asarray(_RNG.normal(size=D) * 0.3)
        sf2, warp0 = jnp.asarray(1.7), jnp.zeros((D, D))
        stationary = kn.kernel(x, x, jnp.exp(log_ell), sf2, "rbf")
        verbatim = kn.kernel(x, x, jnp.exp(log_ell), sf2, "gibbs", warp0)
        self.assertGreater(float(jnp.max(jnp.abs(stationary - verbatim))), 1e-3)


class CheckpointRoundTrip(unittest.TestCase):
    """The optimizer state, and therefore the schedule, survives the fork."""

    def setUp(self):
        steps = 100
        self.schedule = optax.cosine_decay_schedule(1.0, steps, 0.05)
        opt = optax.chain(optax.clip_by_global_norm(1.0),
                          optax.lion(self.schedule))
        params = {"log_ell": jnp.zeros(3), "z": jnp.ones((4, 3))}
        state = opt.init(params)
        grad = jax.tree.map(jnp.ones_like, params)
        for _ in range(60):
            updates, state = opt.update(grad, state, params)
            params = optax.apply_updates(params, updates)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            sv.save_training_checkpoint(path, 60, params, state,
                                        meta={"total_epochs": steps})
            loaded, leaves, self.meta = sv.load_training_checkpoint(path)
        forked = dict(loaded)
        forked["warp"] = jnp.zeros((3, 3))
        self.merged, self.restored, self.missing = sv.merge_optimizer_state(
            opt.init(forked), leaves)

    def counters(self):
        return [int(leaf) for path, leaf in
                jax.tree_util.tree_flatten_with_path(self.merged)[0]
                if "count" in jax.tree_util.keystr(path)]

    def test_epoch_is_recorded(self):
        self.assertEqual(int(self.meta["epoch"]), 60)

    def test_schedule_counter_is_restored(self):
        counters = self.counters()
        self.assertTrue(counters)
        self.assertTrue(all(c == 60 for c in counters), counters)

    def test_learning_rate_continues(self):
        self.assertEqual(float(self.schedule(60)),
                         float(self.schedule(self.counters()[0])))

    def test_only_warp_slots_stay_fresh(self):
        self.assertGreater(self.restored, 0)
        self.assertTrue(all("warp" in key for key in self.missing),
                        self.missing)

    def test_warp_momentum_starts_at_zero(self):
        warp_slots = [float(jnp.max(jnp.abs(leaf))) for path, leaf in
                      jax.tree_util.tree_flatten_with_path(self.merged)[0]
                      if "warp" in jax.tree_util.keystr(path)]
        self.assertTrue(warp_slots)
        self.assertTrue(all(v == 0.0 for v in warp_slots))


class SameKernelResumeIsExact(unittest.TestCase):
    """Resuming must reproduce the continuous run bit for bit."""

    def _check(self, method):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            continuous, _, _ = train(
                method, "rbf", make_args(checkpoint_at_fraction=0.5),
                checkpoint_path=path)
            loaded, leaves, meta = sv.load_training_checkpoint(path)
            forked, _, _ = train(method, "rbf", make_args(), params=loaded,
                                 start_epoch=int(meta["epoch"]), leaves=leaves)
        worst = max(float(jnp.max(jnp.abs(continuous[k] - forked[k])))
                    for k in continuous)
        self.assertEqual(worst, 0.0, f"{method}: divergenza {worst:.2e}")

    def test_svgp(self):
        self._check("SVGP")

    def test_sbt_svgp(self):
        self._check("SBT-SVGP")


class StationaryToGibbsFork(unittest.TestCase):
    """The branch starts exactly where the stationary run was."""

    def test_models_coincide_at_the_fork_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            train("SVGP", "rbf", make_args(checkpoint_at_fraction=0.5),
                  checkpoint_path=path)
            loaded, _, _ = sv.load_training_checkpoint(path)
        engine_rbf = sv.make_engine(X_TR, Y_TR, X_TE, Y_TE, jitter=1e-8,
                                    kernel_name="rbf")
        engine_gibbs = sv.make_engine(X_TR, Y_TR, X_TE, Y_TE, jitter=1e-8,
                                      kernel_name="gibbs")
        gap = max(abs(float(a) - float(b)) for a, b in zip(
            jax.device_get(engine_rbf["metrics"](loaded)),
            jax.device_get(engine_gibbs["metrics"](gated_gibbs_params(loaded)))))
        self.assertLess(gap, 1e-12, f"scarto al fork {gap:.2e}")

    def _drift(self, optimizer):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            continuous, _, _ = train(
                "SVGP", "rbf",
                make_args(optimizer=optimizer, checkpoint_at_fraction=0.5),
                checkpoint_path=path)
            loaded, leaves, meta = sv.load_training_checkpoint(path)
            branch, _, _ = train(
                "SVGP", "gibbs",
                make_args(optimizer=optimizer, warp_gate_fraction=0.999),
                params=gated_gibbs_params(loaded),
                start_epoch=int(meta["epoch"]), leaves=leaves)
        engine_rbf = sv.make_engine(X_TR, Y_TR, X_TE, Y_TE, jitter=1e-8,
                                    kernel_name="rbf")
        engine_gibbs = sv.make_engine(X_TR, Y_TR, X_TE, Y_TE, jitter=1e-8,
                                      kernel_name="gibbs")
        return max(abs(float(a) - float(b)) for a, b in zip(
            jax.device_get(engine_rbf["metrics"](continuous)),
            jax.device_get(engine_gibbs["metrics"](branch))))

    def test_adam_branch_stays_exact(self):
        self.assertLess(self._drift("adam"), 1e-12)

    def test_lion_branch_drift_is_bounded(self):
        # Lion updates by sign(), which turns a 1e-15 difference into a whole
        # step whenever a gradient component sits near zero. That drift is a
        # property of the optimizer rather than of the fork, and it is orders
        # of magnitude below the effects the campaign measures -- but it must
        # not grow silently, so it is bounded here.
        self.assertLess(self._drift("lion"), 2e-4)


class VariationalCovarianceCrossesTheFork(unittest.TestCase):
    """q(u)'s covariance must survive the fork, despite changing name.

    init_params creates the raw factor q_raw; train_svgp replaces it by q_cov
    once, at the start, under the natural-gradient optimizer. A saved run
    therefore stores q_cov while a freshly initialised target still holds
    q_raw, so a key-by-key copy skips the covariance in silence -- and at
    M=512 that is the largest trained object in the model. This regressed once
    already, in both the warm start and the fork.
    """

    def setUp(self):
        from svgp import run_benchmark as runner

        self.runner = runner

    def test_initialisation_and_training_use_different_names(self):
        fresh = sv.init_params(X_TR, M, 0, kernel_name="rbf")
        self.assertIn("q_raw", fresh)
        self.assertNotIn("q_cov", fresh)
        trained, _, _ = train("SVGP", "rbf", make_args())
        self.assertIn("q_cov", trained)
        self.assertNotIn("q_raw", trained)

    def test_checkpoint_stores_the_covariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ck.npz")
            train("SVGP", "rbf", make_args(checkpoint_at_fraction=0.5),
                  checkpoint_path=path)
            stored, _, _ = sv.load_training_checkpoint(path)
        self.assertIn("q_cov", stored)

    def test_adoption_swaps_the_representation(self):
        fresh = sv.init_params(X_TR, M, 0, kernel_name="rbf")
        value = np.eye(M)
        adopted = self.runner.adopt_variational_covariance(
            fresh, value, types.SimpleNamespace(q_optimizer="natural"))
        self.assertNotIn("q_raw", adopted)
        self.assertTrue(bool(jnp.allclose(adopted["q_cov"], value)))

    def test_adoption_refuses_an_incompatible_parameterisation(self):
        fresh = sv.init_params(X_TR, M, 0, kernel_name="rbf")
        with self.assertRaises(SystemExit):
            self.runner.adopt_variational_covariance(
                fresh, np.eye(M), types.SimpleNamespace(q_optimizer="lion"))

    def test_training_keeps_a_supplied_covariance(self):
        # The guard inside train_svgp is "q_cov" not in params; if that ever
        # changes, a forked run would overwrite the posterior it just loaded.
        fresh = sv.init_params(X_TR, M, 0, kernel_name="rbf")
        supplied = dict(fresh)
        supplied.pop("q_raw")
        supplied["q_cov"] = jnp.asarray(np.eye(M) * 4.0)
        args = make_args(epochs=1)
        trained, _, _ = train("SVGP", "rbf", args, params=supplied)
        default, _, _ = train("SVGP", "rbf", args)
        self.assertFalse(bool(jnp.allclose(trained["q_cov"], default["q_cov"])))


class SparseTransplantSchedule(unittest.TestCase):
    """The paper transplant freezes theta, warms q, then releases only Z."""

    def test_theta_stays_exact_and_z_moves_after_release(self):
        initial = sv.init_params(X_TR, M, 17, kernel_name="rbf")
        trained, info, _ = train(
            "SVGP",
            "rbf",
            make_args(
                epochs=20,
                optimizer="adam",
                lr=5e-3,
                z_lr=1e-3,
                phases="5:theta,z 15:theta",
                metrics_at_epochs=(0, 5, 20),
            ),
            params=initial,
        )

        for key in ("log_ell", "log_sf2", "log_s2"):
            self.assertEqual(
                float(jnp.max(jnp.abs(trained[key] - initial[key]))),
                0.0,
                key,
            )
        self.assertGreater(
            float(jnp.max(jnp.abs(trained["z"] - initial["z"]))),
            1e-6,
        )
        self.assertGreater(float(jnp.linalg.norm(trained["q_mu"])), 1e-6)
        self.assertEqual(
            info["phases"],
            [
                {"until_epoch": 5, "frozen": ["theta", "z"]},
                {"until_epoch": 20, "frozen": ["theta"]},
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
