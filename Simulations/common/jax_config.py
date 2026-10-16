#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Enable float64 in JAX, once, before any array is created.

Every bound in this project takes a Cholesky of a matrix that is near-singular
by construction (``I + D/sigma^2`` with a small noise variance), so float32 is
not an option: it does not merely lose accuracy, it loses positive
definiteness.

This used to be a side effect of importing the kernel module. Importing a
module for its side effect is easy to drop during a refactor and hard to notice
afterwards -- the symptom is a test tolerance that quietly needs relaxing -- so
it now has its own module, imported by everything that touches JAX.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
