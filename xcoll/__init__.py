# copyright ############################### #
# This file is part of the Xcoll Package.   #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from .general import _pkg_root, __version__, citation

from .beam_elements import BlackAbsorber, EverestBlock, EverestCollimator, EverestCrystal
from .scattering_routines.everest import materials, Material, CrystalMaterial
from .manager import CollimatorManager
from .colldb import CollimatorDatabase, load_SixTrack_colldb
from .rf_sweep import RFSweep
from .initial_distribution import generate_pencil_on_collimator, generate_delta_from_dispersion, generate_4D_pencil_one_jaw
from .lossmap import LossMap

# print("If you use Xcoll in your simulations, please cite us :-)")
# print(citation)

