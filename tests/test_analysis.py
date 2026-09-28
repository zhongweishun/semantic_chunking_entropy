import math
import random
import unittest
import contextlib
import io
import numpy as np
from scipy.integrate import quad
from analysis.common import sizes_by_level, size_and_logprob, pooled_sizes, binned
from analysis.rtm import expected_entropy, weak_composition, scaling_densities


class AnalysisTests(unittest.TestCase):
    def test_terminal_leaves_and_levels(self):
        tree=[[1,2],0,[1,1]]
        self.assertEqual(sizes_by_level(tree),{3:[1,2,1,1],2:[3,0,2],1:[5]})
        pooled=pooled_sizes([('example',{'partition':tree})])
        np.testing.assert_allclose(pooled[2],[.6,0,.4])
        self.assertEqual(len(pooled[3]),0)

    def test_unsplit_leaf_scoring(self):
        n,logp=size_and_logprob([2,1,0,0],4)
        self.assertEqual(n,3)
        self.assertAlmostEqual(logp,-math.log(math.comb(6,3))-math.log(math.comb(5,3)))

    def test_fast_recurrence_against_direct_sum(self):
        for k in [2,3,4,6]:
            slow=np.zeros(50)
            for n in range(2,50):
                z=math.comb(n+k-1,k-1)
                slow[n]=math.log(z)+k/z*sum(math.comb(n-m+k-2,k-2)*slow[m] for m in range(2,n))
            np.testing.assert_allclose(expected_entropy(k,49),slow,rtol=2e-14)

    def test_binary_closed_form(self):
        h=expected_entropy(2,100)
        for n in [2,7,40,100]:
            exact=math.log(n+1)+2*(n+2)*sum(math.log(m+1)/((m+2)*(m+3)) for m in range(2,n))
            self.assertAlmostEqual(h[n],exact,places=10)

    def test_composition_mass(self):
        rng=random.Random(24)
        for n in [0,1,7,50]:
            for k in [2,4,8]:
                for _ in range(30):
                    parts=weak_composition(n,k,rng)
                    self.assertEqual(sum(parts),n);self.assertEqual(len(parts),k)
                    self.assertTrue(all(v>=0 for v in parts))

    def test_vectorized_markov_against_scalar_reference(self):
        from src.recompute_theory import compute
        from src.rtm_theory_utilities import node_dist_n, node_dist_0, node_dist_01
        for k in [2,4]:
            with contextlib.redirect_stdout(io.StringIO()):
                fast=compute(k,20,2,5)
            for level in range(2,6):
                for key,fn in [('distn',node_dist_n),('dist0',node_dist_0),('dist01',node_dist_01)]:
                    np.testing.assert_allclose(fast[level][key][-1],fn(20,level,k),atol=1e-14,rtol=1e-13)

    def test_fft_against_independent_mellin_integral(self):
        t,pdfs=scaling_densities(max_level=3)
        for s in [.02,.1,.4,.8]:
            exact=quad(lambda r:3*(1-r)**2*3*(1-s/r)**2/r,s,1,epsabs=1e-12)[0]
            got=np.interp(-math.log(s),t,pdfs[3])/s
            self.assertAlmostEqual(got/exact,1,delta=1e-4)
        for density in pdfs.values():
            self.assertAlmostEqual(np.trapz(density,t),1,places=5)

    def test_binning_is_population_sd_and_half_open(self):
        b=binned(np.array([1,2,3,4,5]),np.array([2,4,6,8,10]),np.array([1,5,6]),minimum=2)
        np.testing.assert_allclose(b,[[2.5,5,np.sqrt(5),4]])


if __name__=='__main__':unittest.main()
