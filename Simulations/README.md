# Simulations

The experiment code has one shared core (`common/`) and two stacks: collapsed
full-batch models (`sgpr/`) and minibatch variational models (`svgp/`). Public
block names are RBT for the random partition baseline and SBT for the selected
anchor partition. BT is accepted only when reading archived runs.

Every headline NLL uses the predictor represented by its fitted bound: sparse
for SGPR/SVGP, diagonal for T, complete fitted blocks for RBT/SBT, dense
residual Eq. (12) for E-SGPR, and exact prediction for Exact-GP.

The collapsed-only campaign is submitted by `submit_sgpr_campaign.sh`. The
complete SGPR+SVGP suite is submitted by `submit_final_campaign.sh`.
