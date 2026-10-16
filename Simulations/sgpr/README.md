# Collapsed SGPR campaign

The official experiment uses Kin40k, N=5000 training and 2000 test points,
M=256, block size 500, ten seeds, full-batch L-BFGS, and four kernels: RBF,
Matern-5/2, amplitude-RBF, and Gibbs/Paciorek.

Seven methods are fitted: Exact-GP, E-SGPR, METIS-SGPR, SGPR, T-SGPR,
RBT-SGPR, and SBT-SGPR. Their headline predictors are respectively exact,
dense residual, fitted METIS blocks, sparse, diagonal, all random blocks, and
all selected anchor blocks. Every fit saves parameters; block methods save the
final partition.

The campaign also generates the Snelson reference, same-parameter RBT/SBT/METIS
residual heatmaps, and a second table in which each donor's theta/noise are
frozen and only Z is optimized under the SGPR objective.
