"""Network models.

This package provides functions to create scattering matrices for networks such
as thin films, free space, absorbing loads, etc..

.. moduleauthor:: Bertrand Delforge <b.delforge@sron.nl>

"""
from distance import Distance
from distance import Distance1
from gain import Gain
from gain import Gain1
from grid import Grid
from interface import InterfaceNormal
from interface import InterfaceOblique
from mirror import AnisotropicMirrorNormal
from mirror import MirrorNormal
from mirror import MirrorNormal1
from mirror import SemiTransparentMirrorNormal
from mirror import SemiTransparentMirrorNormal1
from mirror import TwoWayMirrorNormal
from mirror import Rooftop
from mirror import PolarizationScrambler
from mixer import Mixer
from thinfilm import ThinFilmOblique
from thinfilm import ThinFilmNormal
