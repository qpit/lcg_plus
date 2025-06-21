from bosonicplus.optimizer import run_opts
from bosonicplus.cost_functions import symm_effective_squeezing, symm_effective_squeezing_gradients
from itertools import product, repeat
import numpy as np
import pickle

num_opts = 20
cutoff = 10

niter = 30

inf = 1e-5
costf_lattice = 's'
setting = 'no_phase'
pPNR = False
nbars = None
etas = None

nmodes = 4

bs = ['Clements', 'cascade', 'inv_cascade']

costfs = [symm_effective_squeezing, symm_effective_squeezing_gradients]

patterns = [list(repeat(i,nmodes-1)) for i in range(cutoff)] #Just the diagonals
#patterns = list(product(range(cutoff+1), repeat=nmodes-1))

run_opts(nmodes, num_opts, cutoff, niter, bs, costfs, patterns, inf, costf_lattice, setting, pPNR, nbars, etas)


