Adaptive Learning with SGD and DMRG
###################################

When an MPS is initialized with `adaptive_mode` set to `True`, training proceeds by alternating between different "offsets" of the MPS cores. Each offset combines adjacent pairs of cores into effective merged cores, so that half of the bonds in the MPS are contracted over. This contraction provides a low rank decomposition of the initial merged core, but as training progresses the rank across this bond will typically increase.

After a certain number of inputs are fed to the MPS (equal to `merge_threshold`), each merged core is split in two via a singular value decomposition (SVD) across the contracted bond index. A truncation is then applied which removes all singular values less than `cutoff`, yielding a collection of split cores with half of the bonds having reduced bond dimension. These cores are then merged along a different offset and the process repeated, so that all of the bond dimensions are eventually optimized.

Throughout this process, real-time lists of all the truncated bond dimensions and singular value spectra are accessible as the attributes `my_mps.bond_list` and `my_mps.sv_list`.

This adaptive training mode was directly inspired by the ML DMRG training procedure in [Stoudenmire and Schwab 2016][S&S], which uses a similar division of training into merging and splitting steps, but with a different overall control flow.
