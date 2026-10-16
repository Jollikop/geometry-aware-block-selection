"""Numerical algebra tests independent of the large Kin40k experiment."""
import unittest

import numpy as np

from Simulations.analysis.audit_sgpr_refresh import exchange_quantities, partition_quantities


class ExchangeAuditTests(unittest.TestCase):
    def test_exact_two_by_two_gain_and_ordered_pairs(self):
        d=np.array([[2.,1.,0.,0.],[1.,2.,0.,0.],[0.,0.,2.,1.],[0.,0.,1.,2.]])
        p=[np.array([0,2]),np.array([1,3])]
        q=[np.array([0,1]),np.array([2,3])]
        r=exchange_quantities(d,p,q,1.)
        self.assertAlmostEqual(r['G_plus'],4.)
        self.assertAlmostEqual(r['G_minus'],0.)
        self.assertAlmostEqual(r['kappa_Q'],4.)
        self.assertAlmostEqual(r['delta_elbo_per_datum'],np.log(9/8)/4)
        self.assertTrue(r['sufficient_condition_passed'])
        self.assertTrue(r['bound_holds_numerically'])

    def test_identical_up_to_block_labels(self):
        d=np.eye(6)+.3*np.ones((6,6))
        p=[np.array([0,1,2]),np.array([3,4,5])]
        r=exchange_quantities(d,p,p[::-1],.2)
        self.assertEqual(r['delta_elbo_per_datum'],0.)
        self.assertEqual(r['G_plus'],0.)
        self.assertEqual(r['G_minus'],0.)
        self.assertTrue(r['same_partition'])

    def test_psd_signed_and_unequal_partitions(self):
        rng=np.random.default_rng(182)
        for _ in range(12):
            a=rng.normal(size=(9,6)); d=a@a.T
            p=[np.arange(4),np.arange(4,9)]
            perm=rng.permutation(9); q=[perm[:2],perm[2:6],perm[6:]]
            r=exchange_quantities(d,p,q,.17)
            self.assertTrue(r['bound_holds_numerically'])
            lp=np.array([0]*4+[1]*5)
            lq=np.empty(9,dtype=int)
            for i,b in enumerate(q): lq[b]=i
            mask=(lq[:,None]==lq[None,:])&(lp[:,None]!=lp[None,:])
            self.assertAlmostEqual(r['G_plus'],np.sum(d[mask]**2))

    def test_separation_extrema_and_invalid_cover(self):
        d=np.array([[3.,-1.,.2,.1],[-1.,3.,.4,.3],[.2,.4,3.,-.8],[.1,.3,-.8,3.]])
        q=[np.array([0,1]),np.array([2,3])]
        s=partition_quantities(d,q,.1)
        self.assertAlmostEqual(s['u'],.8)
        self.assertAlmostEqual(s['v'],.4)
        with self.assertRaises((ValueError,AssertionError)):
            partition_quantities(d,[np.array([0,0]),np.array([2,3])],.1)


if __name__=='__main__':
    unittest.main()
