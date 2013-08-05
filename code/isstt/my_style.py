#! /usr/bin/python
"""

"""
from __future__ import division
from contextlib import contextmanager
import pylab
import scipy.constants
import matplotlib as mpl
#             Black      Red        Blue       Purple     Orange.
COLORS_STD = ['#333333', '#E41A1C', '#377EB8', '#984EA3', '#FF7F00']
COLORS_10 = ['#A6CEE3', '#1F78B4', '#B2DF8A', '#33A02C', '#FB9A99', '#E31A1C', '#FDBF6F', '#FF7F00', '#CAB2D6', '#6A3D9A']
COLORS_ISCC_CBS = [
                   '#ff8b96',  # Pink.
                   '#c10430',  # Red.
                   '#f68600',  # Orange.
                   '#82430c',  # Brown.
                   '#5e55e7',  # Olive.
                   '#add8e6',  # Light blue.
                   '#abc800',  # Yellow green.
                   '#00906c',  # Green.
                   '#0069a5',  # Blue.
                   '#8f00a5',  # Purple.
                   '#e9bf00',  # Yellow.
                   '#eaeae9',  # White.
                   '#878686',  # Gray.
                   '#212121',  # Black.
                   ]
WIDTH_ARTICLE = 360
WIDTH_BEAMER = 270

def latexT(s):
    return "$\\textrm{%s}$" % s
def latexM(s):
    return "$%s$" % s

def pretty(linewidth=2):
    AXES_COLOR = '#808080'
    LABEL_COLOR = '#000000'
    mpl.rc('lines', linewidth=linewidth)
    mpl.rc('axes', edgecolor=AXES_COLOR, labelcolor=LABEL_COLOR, grid=True)
    mpl.rc('xtick', color=AXES_COLOR)
    mpl.rc('ytick', color=AXES_COLOR)
    mpl.rc('grid', color=AXES_COLOR)

INCHES_PER_PT = 1 / 72.27

def computeFigureSize(width_pt, aspect_ratio):
    width = width_pt * INCHES_PER_PT
    height = width / aspect_ratio
    return [width, height]

def startlatex(width_pt, aspect_ratio=scipy.constants.golden):
    """width pt obtained with \showthe\columnwidth in latex; ex: 345."""
    fig_size = computeFigureSize(width_pt, aspect_ratio)
    params = {'backend': 'ps',
              'text.usetex': True,
              'axes.labelsize': 10,
              'text.fontsize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'text.latex.preamble' : ['\usepackage{siunitx}'],
              'figure.figsize': fig_size}
    pylab.rcParams.update(params)

def stoplatex():
    pylab.rcParams.update(pylab.rcParamsDefault)

@contextmanager
def latex(width_pt, aspect_ratio=scipy.constants.golden):
    startlatex(width_pt, aspect_ratio)
    yield
    stoplatex()
