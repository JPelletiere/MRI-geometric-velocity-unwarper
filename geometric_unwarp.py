#!/usr/bin/env python
"""
===============================================================================
Combined Gradunwarp Script
Version: HCP-1.2.0

This script combines all parts of the gradunwarp package into a single file.
It includes:
    - globals: Global parameters and logger setup.
    - utils: Utility functions (coordinate transformations, meshgrid, etc.).
    - coeffs: Functions to parse gradient/coefficient files.
    - unwarp_resample: Core unwarping/resampling routines.
    - gradient_unwarp: The main driver that performs geometric unwarping of a 
      DICOM volume.
      
See the COPYING file distributed along with the gradunwarp package for the
copyright and license terms.
===============================================================================
"""

####################################
#              GLOBALS             #
####################################

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from __future__ import print_function
import logging
import dicom2nifti
from argparse import Namespace
import pandas as pd


VERSION = 'HCP-1.2.0'

usage = '''
gradient_unwarp infile outfile manufacturer -g <coefficient file> [optional arguments]
'''

# ---------------------- SIEMENS Configuration -----------------------------
siemens_cas = 100  # coefficient array size
siemens_fovmin = -0.18  # fov min in meters
siemens_fovmax = 0.18   # fov max in meters
siemens_numpoints = 225 # number of grid points in each direction
siemens_max_det = 10.0  # maximum Jacobian determinant for Siemens

# ------------------------- GE Configuration -------------------------------
ge_cas = 6  # coefficient array size
ge_fov_min = -0.3
ge_fov_max = 0.3
ge_resolution = 0.0075

def get_logger():
    """
    Create and return a logger for the gradunwarp package.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log = logging.getLogger('gradunwarp')
    log.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)s-%(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log

logger = get_logger()

####################################
#              UTILS               #
####################################

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import numpy as np
from collections import namedtuple
import math
from math import sqrt, cos, pi

# Import SciPy's RegularGridInterpolator and special functions
from scipy.interpolate import RegularGridInterpolator
from scipy.special import lpmv  # For associated Legendre polynomials

# This is a container class that has 3 np.arrays which contain
# the x, y and z coordinates respectively. For example, the output
# of a meshgrid belongs to this:
#   x, y, z = meshgrid(np.arange(5), np.arange(6), np.arange(7))
#   cv = CoordsVector(x=x, y=y, z=z)
CoordsVector = namedtuple('CoordsVector', 'x, y, z')

# this method is deprecated because it's slow and my suspicion that
# the matrix expressions create unnecessary temp matrices which 
# are costly for huge matrices
def transform_coordinates_old(A, M):
    ''' 4x4 matrix M operates on orthogonal coordinate arrays
    A.x, A.y, A.z to give B1, B2, B3
    '''
    A1 = A.x
    A2 = A.y
    A3 = A.z
    B1 = A1 * M[0, 0] + A2 * M[0, 1] + A3 * M[0, 2] + M[0, 3]
    B2 = A1 * M[1, 0] + A2 * M[1, 1] + A3 * M[1, 2] + M[1, 3]
    B3 = A1 * M[2, 0] + A2 * M[2, 1] + A3 * M[2, 2] + M[2, 3]
    return CoordsVector(B1, B2, B3)

def transform_coordinates(A, M, dtype=np.float32):
    ''' 
    4x4 matrix M operates on coordinate arrays A.x, A.y, A.z and returns
    transformed coordinates (B1, B2, B3).

    Parameters:
        A (CoordsVector): Named tuple containing x, y, z coordinate arrays.
        M (np.ndarray): 4x4 transformation matrix.
        dtype (data-type, optional): Desired data type (default np.float32).

    Returns:
        CoordsVector: Transformed coordinates.
    '''
    Ax = np.ascontiguousarray(A.x, dtype=dtype)
    Ay = np.ascontiguousarray(A.y, dtype=dtype)
    Az = np.ascontiguousarray(A.z, dtype=dtype)
    M = np.ascontiguousarray(M, dtype=dtype)
    B1 = Ax * M[0, 0] + Ay * M[0, 1] + Az * M[0, 2] + M[0, 3]
    B2 = Ax * M[1, 0] + Ay * M[1, 1] + Az * M[1, 2] + M[1, 3]
    B3 = Ax * M[2, 0] + Ay * M[2, 1] + Az * M[2, 2] + M[2, 3]
    return CoordsVector(B1, B2, B3)

def get_vol_affine(infile):
    """
    Load volume data and its affine transformation matrix from a NIfTI or MGH file.

    Parameters:
        infile (str): Path to the input file.

    Returns:
        tuple: (data, affine) where data is a numpy array and affine is the 4x4 matrix.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError('gradunwarp needs nibabel for I/O of mgz/nifti files. Please install.')
    nibimage = nib.load(infile)
    pixdims = nibimage.header.get_zooms()
    pixdim1 = pixdims[0]
    pixdim2 = pixdims[1]
    pixdim3 = pixdims[2]
    print(f"Pixel Dimensions: {pixdim1}, {pixdim2}, {pixdim3}")
    data = nibimage.get_fdata(dtype=np.float32)
    affine = nibimage.affine
    return data, affine

# Memoized factorial (or simply use math.factorial)
class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]

# Uncomment to use memoized factorial:
# factorial = Memoize(math.factorial)
factorial = math.factorial

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
# This is taken from a numpy ticket which adds N-d support to meshgrid
# URL : http://projects.scipy.org/numpy/ticket/966
# License : http://docs.scipy.org/doc/numpy/license.html
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
def meshgrid(*xi, **kwargs):
    """
    Generate coordinate matrices from coordinate vectors.
    
    Parameters:
        *xi: 1-D arrays representing coordinate vectors.
        copy (bool, optional): Whether to return copies (default True).
        sparse (bool, optional): Whether to return a sparse grid.
        indexing (str, optional): 'xy' (default) or 'ij'.

    Returns:
        list: List of N arrays forming the N-D grid.
    """
    copy = kwargs.get('copy', True)
    args = np.atleast_1d(*xi)
    if not isinstance(args, list):
        if args.size > 0:
            return args.copy() if copy else args
        else:
            raise TypeError('meshgrid() takes at least one argument (0 given)')
    sparse = kwargs.get('sparse', False)
    indexing = kwargs.get('indexing', 'xy')
    ndim = len(args)
    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i+1:]) for i, x in enumerate(args)]
    shape = [x.size for x in output]
    if indexing == 'xy':
        output[0].shape = (1, -1) + (1,)*(ndim-2)
        output[1].shape = (-1, 1) + (1,)*(ndim-2)
        shape[0], shape[1] = shape[1], shape[0]
    if sparse:
        return [x.copy() for x in output] if copy else output
    else:
        if copy:
            mult_fact = np.ones(shape, dtype=int)
            return [x * mult_fact for x in output]
        else:
            return np.broadcast_arrays(*output)

def ndgrid(*args, **kwargs):
    """
    Same as calling meshgrid with indexing='ij'.
    """
    kwargs['indexing'] = 'ij'
    return meshgrid(*args, **kwargs)

def odd_factorial(k):
    """
    Compute the odd factorial of k.
    """
    f = k
    while k >= 3:
        k -= 2
        f *= k
    return f

def legendre(nu, mu, x):
    """
    Compute the associated Legendre polynomial P_nu^mu(x) using SciPy's lpmv.
    
    Parameters:
        nu (int): Degree.
        mu (int): Order.
        x (np.ndarray): Input values.
        
    Returns:
        np.ndarray: Values of the Legendre polynomial.
    """
    nu = int(nu)
    mu = int(mu)
    x = np.ascontiguousarray(x, dtype=np.float32)
    if mu < 0 or mu > nu:
        raise ValueError(f"Require 0 <= mu <= nu, but mu={mu} and nu={nu}")
    P = lpmv(mu, nu, x)
    return P.astype(np.float32)

def interp3(vol, R, C, S):
    """
    Tricubic interpolation using SciPy's RegularGridInterpolator.
    
    Parameters:
        vol (np.ndarray): 3D volume.
        R, C, S (np.ndarray): Coordinate arrays.
        
    Returns:
        np.ndarray: Interpolated values.
    """
    x = np.arange(vol.shape[0])
    y = np.arange(vol.shape[1])
    z = np.arange(vol.shape[2])
    interpolator = RegularGridInterpolator((x, y, z), vol, method='cubic', bounds_error=False, fill_value=0)
    points = np.stack((R, C, S), axis=-1)
    V = interpolator(points)
    return V.astype(np.float32)

####################################
#             COEFFS               #
####################################

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from collections import namedtuple
import numpy as np
import logging
import re
siemens_cas_local = siemens_cas
ge_cas_local = ge_cas

log = logging.getLogger('gradunwarp')

Coeffs = namedtuple('Coeffs', 'alpha_x, alpha_y, alpha_z, beta_x, beta_y, beta_z, R0_m')

# Python 2/3 iterator compatibility.
try:
    advance_iterator = next
except NameError:
    def advance_iterator(it):
        return it.next()
next = advance_iterator

def get_coefficients(vendor, cfile):
    """
    Depending on the vendor and the coefficient file extension,
    return the spherical harmonics coefficients as a Coeffs namedtuple.
    """
    log.info('Parsing ' + cfile + ' for harmonics coeffs')
    if vendor == 'siemens' and cfile.endswith('.coef'):
        return get_siemens_coef(cfile)
    if vendor == 'siemens' and cfile.endswith('.grad'):
        return get_siemens_grad(cfile)
    # (For GE, additional handling would be implemented here.)

def coef_file_parse(cfile, txt_var_map):
    """
    Parses a .coef file (for GE or Siemens) and updates txt_var_map in place.
    """
    coef_re = re.compile('^[^\#]')  # Regex for a line not starting with '#'
    coef_file = open(cfile, 'r')
    for line in coef_file.readlines():
        if coef_re.match(line):
            validline_list = line.lstrip(' \t').rstrip(';\n').split()
            if validline_list:
                log.info('Parsed : %s' % validline_list)
                l = validline_list
                x = int(l[1])
                y = int(l[2])
                txt_var_map[l[0]][x, y] = float(l[3])

def get_siemens_coef(cfile):
    """
    Parse the Siemens .coef file.
    R0_m is set to a default value.
    """
    ##    R0m_map = {'sonata': 0.25,
    ##               'avanto': 0.25,
    ##               'quantum': 0.25,
    ##               'allegra': 0.14,
    ##               'as39s': 0.25,
    ##               'as39st': 0.25,
    ##               'as39t': 0.25,
    ##               'prisma': 0.25}
    ##    for rad in R0m_map.keys():
    ##        if cfile.startswith(rad):
    R0_m = 0.25
    coef_array_sz = siemens_cas_local
    if cfile.startswith('allegra'):
        coef_array_sz = 15
    ##    if cfile.startswith('prisma1'):
    coef_array_sz = 20
    ax = np.zeros((coef_array_sz, coef_array_sz))
    ay = np.zeros((coef_array_sz, coef_array_sz))
    az = np.zeros((coef_array_sz, coef_array_sz))
    bx = np.zeros((coef_array_sz, coef_array_sz))
    by = np.zeros((coef_array_sz, coef_array_sz))
    bz = np.zeros((coef_array_sz, coef_array_sz))
    ##    txt_var_map = {'Alpha_x': ax,
    ##                   'Alpha_y': ay,
    ##                   'Alpha_z': az,
    ##                   'Beta_x': bx,
    ##                   'Beta_y': by,
    ##                   'Beta_z': bz}
    ##
    ##    coef_file_parse(cfile, txt_var_map)
    # Manually assigned coefficients (example values)
    with open("coeff.grad", "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or comment lines starting with '#'
            if not line or line.startswith("#"):
                continue
            # Expected line format (for coefficient lines):
            #   <serial> <Letter>( <i, j>) <value> <axis>
            # Example: "  1 A( 3, 0)           -0.05625983                  z"
            pattern = r'^\s*\d+\s+([AB])\(\s*(\d+),\s*(\d+)\)\s+([-.\d]+)\s+([xyz])'
            match = re.match(pattern, line)
            if not match:
                continue
            coeff_letter = match.group(1)  # "A" or "B"
            i = int(match.group(2))
            j = int(match.group(3))
            value = float(match.group(4))
            axis = match.group(5)  # "x", "y", or "z"
            # For coefficients starting with A
            if coeff_letter == "A":
                if axis == "z":
                    # For axis z, assign to az; use j=0 regardless of the file j value.
                    az[i, 0] = value
                elif axis == "x":
                    ax[i, j] = value
            # For coefficients starting with B (in this file, they have axis y)
            elif coeff_letter == "B":
                if axis == "y":
                    by[i, j] = value
    print(Coeffs(ax, ay, az, bx, by, bz, R0_m))
    return Coeffs(ax, ay, az, bx, by, bz, R0_m)

def get_ge_coef(cfile):
    """
    Parse the GE .coef file.
    """
    ax = np.zeros((ge_cas_local, ge_cas_local))
    ay = np.zeros((ge_cas_local, ge_cas_local))
    az = np.zeros((ge_cas_local, ge_cas_local))
    bx = np.zeros((ge_cas_local, ge_cas_local))
    by = np.zeros((ge_cas_local, ge_cas_local))
    bz = np.zeros((ge_cas_local, ge_cas_local))
    txt_var_map = {'Alpha_x': ax,
                   'Alpha_y': ay,
                   'Alpha_z': az,
                   'Beta_x': bx,
                   'Beta_y': by,
                   'Beta_z': bz}
    coef_file_parse(cfile, txt_var_map)
    return Coeffs(ax, ay, az, bx, by, bz, R0_m)

def grad_file_parse(gfile, txt_var_map):
    """
    Parse a Siemens .grad file and update txt_var_map in place.

    Returns:
        tuple: (R0_m, (xmax, ymax))
    """
    gf = open(gfile, 'r')
    line = next(gf)
    # Skip the comments.
    while not line.startswith('#*] END:'):
        line = next(gf)
    # Get R0.
    line = next(gf)
    line = next(gf)
    line = next(gf)
    R0_m = float(line.strip().split()[0])
    # Go to the data.
    line = next(gf)
    line = next(gf)
    line = next(gf)
    line = next(gf)
    line = next(gf)
    line = next(gf)
    line = next(gf)
    xmax = 0
    ymax = 0
    while 1:
        lindex = line.find('(')
        rindex = line.find(')')
        if lindex == -1 and rindex == -1:
            break
        arrindex = line[lindex+1:rindex]
        xs, ys = arrindex.split(',')
        x = int(xs)
        y = int(ys)
        if x > xmax:
            xmax = x
        if y > ymax:
            ymax = y
        if line.find('A') != -1 and line.find('x') != -1:
            txt_var_map['Alpha_x'][x, y] = float(line.split()[-2])
        if line.find('A') != -1 and line.find('y') != -1:
            txt_var_map['Alpha_y'][x, y] = float(line.split()[-2])
        if line.find('A') != -1 and line.find('z') != -1:
            txt_var_map['Alpha_z'][x, y] = float(line.split()[-2])
        if line.find('B') != -1 and line.find('x') != -1:
            txt_var_map['Beta_x'][x, y] = float(line.split()[-2])
        if line.find('B') != -1 and line.find('y') != -1:
            txt_var_map['Beta_y'][x, y] = float(line.split()[-2])
        if line.find('B') != -1 and line.find('z') != -1:
            txt_var_map['Beta_z'][x, y] = float(line.split()[-2])
        try:
            line = next(gf)
        except StopIteration:
            break
    return R0_m, (xmax, ymax)

def get_siemens_grad(gfile):
    """
    Parse the Siemens .grad file.
    """
    coef_array_sz = siemens_cas_local
    if gfile.startswith('coef_AC44'):
        coef_array_sz = 15
    ax = np.zeros((coef_array_sz, coef_array_sz))
    ay = np.zeros((coef_array_sz, coef_array_sz))
    az = np.zeros((coef_array_sz, coef_array_sz))
    bx = np.zeros((coef_array_sz, coef_array_sz))
    by = np.zeros((coef_array_sz, coef_array_sz))
    bz = np.zeros((coef_array_sz, coef_array_sz))
    txt_var_map = {'Alpha_x': ax,
                   'Alpha_y': ay,
                   'Alpha_z': az,
                   'Beta_x': bx,
                   'Beta_y': by,
                   'Beta_z': bz}
    R0_m, max_ind = grad_file_parse(gfile, txt_var_map)
    ind = max(max_ind)
    ax = ax[:ind+1, :ind+1]
    ay = ay[:ind+1, :ind+1]
    az = az[:ind+1, :ind+1]
    bx = bx[:ind+1, :ind+1]
    by = by[:ind+1, :ind+1]
    bz = bz[:ind+1, :ind+1]
    return Coeffs(ax, ay, az, bx, by, bz, R0_m)

####################################
#          UNWARP_RESAMPLE         #
####################################

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the gradunwarp package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

import sys
import pdb
import gc
import math
import logging
from scipy import ndimage
import nibabel as nib
import subprocess
import matplotlib.pyplot as plt
from matplotlib import cm

log = logging.getLogger('gradunwarp')

class Unwarper(object):
    """
    Class for performing nonlinear unwarping/resampling of an image volume.
    """
    def __init__(self, vol, m_rcs2ras, vendor, coeffs, fileName):
        self.vol = vol
        self.m_rcs2ras = m_rcs2ras
        self.vendor = vendor
        self.coeffs = coeffs
        self.name = fileName
        self.warp = False
        self.nojac = False
        self.m_rcs2lai = None
        self.fovmin = None
        self.fovmax = None
        self.numpoints = None
        self.order = 3

    def eval_spharm_grid(self, vendor, coeffs):
        """
        Evaluate spherical harmonics on a lower-resolution grid.
        """
        fov_min = self.fovmin if self.fovmin is not None else siemens_fovmin
        fov_max = self.fovmax if self.fovmax is not None else siemens_fovmax
        num_points = self.numpoints if self.numpoints is not None else siemens_numpoints
        fov_min_mm = fov_min * 1000.0
        fov_max_mm = fov_max * 1000.0
        grid_vector = np.linspace(fov_min_mm, fov_max_mm, num_points)
        gvx, gvy, gvz = meshgrid(grid_vector, grid_vector, grid_vector)
        cf = (fov_max_mm - fov_min_mm) / num_points
        g_rcs2xyz = np.array([[0, cf, 0, fov_min_mm],
                              [cf, 0, 0, fov_min_mm],
                              [0, 0, cf, fov_min_mm],
                              [0, 0, 0, 1]], dtype=np.float32)
        g_xyz2rcs = np.linalg.inv(g_rcs2xyz)
        gr, gc, gs = meshgrid(np.arange(num_points), np.arange(num_points), np.arange(num_points), dtype=np.float32)
        log.info(f"Evaluating spherical harmonics on a {num_points}^3 grid")
        log.info(f"with extents {fov_min_mm}mm to {fov_max_mm}mm")
        gvxyz = CoordsVector(gvx, gvy, gvz)
        _dv, _dxyz = eval_spherical_harmonics(coeffs, vendor, gvxyz)
        return CoordsVector(_dv.x, _dv.y, _dv.z), g_xyz2rcs

    def run(self):
        """
        Run the unwarping/resampling process.
        """
        self.polarity = -1.0 if self.warp else 1.0
        dv, g_xyz2rcs = self.eval_spharm_grid(self.vendor, self.coeffs)
        m_ras2lai = np.array([[-1.0, 0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]], dtype=np.float)
        m_rcs2lai = np.dot(m_ras2lai, self.m_rcs2ras)
        m_rcs2lai_nohalf = m_rcs2lai.copy()
        '''
        nr, nc, ns = self.vol.shape[:3]
        vc3, vr3, vs3 = meshgrid(np.arange(nr), np.arange(nc), np.arange(ns), dtype=np.float32)
        vrcs = CoordsVector(x=vr3, y=vc3, z=vs3)
        vxyz = transform_coordinates(vrcs, m_rcs2lai)
        '''
        halfvox = np.zeros((4, 4))
        halfvox[0, 3] = m_rcs2lai[0, 0] / 2.0
        halfvox[1, 3] = m_rcs2lai[1, 1] / 2.0
        # m_rcs2lai = m_rcs2lai + halfvox
        r_rcs2lai = np.eye(4, 4)
        r_rcs2lai[:3, :3] = m_rcs2lai[:3, :3]
        ones = CoordsVector(1.0, 1.0, 1.0)
        dxyz = transform_coordinates(ones, r_rcs2lai)
        if self.vendor == 'siemens':
            self.out, self.vjacout = self.non_linear_unwarp_siemens(self.vol.shape, dv, dxyz,
                                                                     m_rcs2lai, m_rcs2lai_nohalf, g_xyz2rcs)

    def non_linear_unwarp_siemens(self, vol_shape, dv, dxyz, m_rcs2lai, m_rcs2lai_nohalf, g_xyz2rcs):
        """
        Perform Siemens-specific nonlinear unwarping.
        """
        log.info('Evaluating the jacobian multiplier')
        nr, nc, ns = self.vol.shape[:3]
        if not self.nojac:
            jim2 = np.zeros((nr, nc), dtype=np.float32)
            vjacdet_lpsw = np.zeros((nr, nc), dtype=np.float32)
            if dxyz == 0:
                vjacdet_lps = 1
            else:
                print('calculating jacobian')
                vjacdet_lps = eval_siemens_jacobian_mult(dv, dxyz)
        out_vol = np.zeros((nr, nc, ns), dtype=np.float32)
        fullWarp = np.zeros((nr, nc, ns, 3), dtype=np.float32)
        vjacout = np.zeros((nr, nc, ns), dtype=np.float32)
        im2 = np.zeros((nr, nc), dtype=np.float32)
        dvx = np.zeros((nr, nc), dtype=np.float32)
        dvy = np.zeros((nr, nc), dtype=np.float32)
        dvz = np.zeros((nr, nc), dtype=np.float32)
        im_ = np.zeros((nr, nc), dtype=np.float32)
        vc, vr = meshgrid(np.arange(nc), np.arange(nr))
        try:
            pixdim1 = 0.6957
        except ValueError:
            log.error('Failure during fslval call. Ensure fslval and related commands are in PATH.')
            sys.exit(1)
        pixdim2 = 0.6957
        pixdim3 = 0.7
        # dim1 = float((subprocess.Popen(['fslval', self.name, 'dim1'], stdout=subprocess.PIPE).communicate()[0]).strip())
        # outputOrient = subprocess.Popen(['fslorient', self.name], stdout=subprocess.PIPE).communicate()[0].strip()
        outputOrient = 'c'
        if outputOrient == b'NEUROLOGICAL':
            log.info('Input volume is NEUROLOGICAL orientation. Flipping x-axis in output fullWarp_abs.nii.gz')
            m_vox2fsl = np.array([[-pixdim1, 0.0, 0.0, pixdim1*(nr-1)],
                                   [0.0, pixdim2, 0.0, 0.0],
                                   [0.0, 0.0, pixdim3, 0.0],
                                   [0.0, 0.0, 0.0, 1.0]], dtype=np.float)
        else:
            m_vox2fsl = np.array([[pixdim1, 0.0, 0.0, 0.0],
                                  [0.0, pixdim2, 0.0, 0.0],
                                  [0.0, 0.0, pixdim3, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]], dtype=np.float)
        log.info('Unwarping slice by slice')
        for s in range(ns):
            sys.stdout.flush()
            if (s+1) % 10 == 0:
                print(s+1, end=' ')
            else:
                print('.', end=' ')
            gc.collect()
            dvx.fill(0.0)
            dvy.fill(0.0)
            dvz.fill(0.0)
            im_.fill(0.0)
            vs = np.ones(vr.shape) * s
            # Use positional arguments for CoordsVector (fields: x, y, z)
            voxel_coords = CoordsVector(vr, vc, vs)
            vxyz = transform_coordinates(voxel_coords, m_rcs2lai_nohalf)
            vrcsg = transform_coordinates(vxyz, g_xyz2rcs)
            ndimage.interpolation.map_coordinates(dv.x, vrcsg, output=dvx, order=self.order)
            ndimage.interpolation.map_coordinates(dv.y, vrcsg, output=dvy, order=self.order)
            ndimage.interpolation.map_coordinates(dv.z, vrcsg, output=dvz, order=self.order)
            vxyzw = CoordsVector(x=vxyz.x + self.polarity * dvx,
                                   y=vxyz.y + self.polarity * dvy,
                                   z=vxyz.z + self.polarity * dvz)
            vrcsw = transform_coordinates(vxyzw, np.linalg.inv(m_rcs2lai))
            vfsl = transform_coordinates(vrcsw, m_vox2fsl)
            ndimage.interpolation.map_coordinates(self.vol, vrcsw, output=im_, order=self.order)
            im_[np.where(np.isnan(im_))] = 0.0
            im_[np.where(np.isinf(im_))] = 0.0
            im2[vr, vc] = im_
            print(np.shape(im_))
            # img = nib.Nifti1Image(dvx, np.eye(4))
            # nib.save(img, "x"+str(s).zfill(3)+".nii.gz")
            # img = nib.Nifti1Image(dvy, np.eye(4))
            # nib.save(img, "y"+str(s).zfill(3)+".nii.gz")
            # img = nib.Nifti1Image(dvz, np.eye(4))
            # nib.save(img, "z"+str(s).zfill(3)+".nii.gz")
            if not self.nojac:
                print('applying jacobian')
                vjacdet_lpsw.fill(0.0)
                jim2.fill(0.0)
                if self.polarity == -1:
                    vjacdet_lps = 1.0 / vjacdet_lps
                ndimage.interpolation.map_coordinates(vjacdet_lps, vrcsg, output=vjacdet_lpsw, order=self.order)
                vjacdet_lpsw[np.where(np.isnan(vjacdet_lpsw))] = 0.0
                vjacdet_lpsw[np.where(np.isinf(vjacdet_lpsw))] = 0.0
                jim2[vr, vc] = vjacdet_lpsw
                im2 = im2 * jim2
                vjacout[..., s] = jim2
            fullWarp[..., s, 0] = vfsl.x
            fullWarp[..., s, 1] = vfsl.y
            fullWarp[..., s, 2] = vfsl.z
            out_vol[..., s] = im2
        print()
        img = nib.Nifti1Image(fullWarp, self.m_rcs2ras)
        nib.save(img, "fullWarp_abs.nii.gz")
        return out_vol, vjacout

    def write(self, outfile):
        log.info('Writing output to ' + outfile)
        if self.out.dtype == np.float64:
            self.out = self.out.astype(np.float32)
        if outfile.endswith('.nii') or outfile.endswith('.nii.gz'):
            img = nib.Nifti1Image(self.out, self.m_rcs2ras)
        elif outfile.endswith('.mgh') or outfile.endswith('.mgz'):
            img = nib.MGHImage(self.out, self.m_rcs2ras)
        else:
            raise ValueError("Unsupported file extension for output.")
        nib.save(img, outfile)

def eval_siemens_jacobian_mult(F, dxyz):
    """
    Evaluate the Jacobian determinant multiplier for Siemens unwarping.
    """
    d0, d1, d2 = dxyz.x, dxyz.y, dxyz.z
    if d0 == 0 or d1 == 0 or d2 == 0:
        raise ValueError('weirdness found in Jacobian calculation')
    dFxdx, dFxdy, dFxdz = np.gradient(F.x, d0, d1, d2)
    dFydx, dFydy, dFydz = np.gradient(F.y, d0, d1, d2)
    dFzdx, dFzdy, dFzdz = np.gradient(F.z, d0, d1, d2)
    jacdet = ((1. + dFxdx) * (1. + dFydy) * (1. + dFzdz)
              - (1. + dFxdx) * dFydz * dFzdy
              - dFxdy * dFydx * (1. + dFzdz)
              + dFxdy * dFydz * dFzdx
              + dFxdz * dFydx * dFzdy
              - dFxdz * (1. + dFydy) * dFzdx)
    jacdet = np.abs(jacdet)
    jacdet[np.where(jacdet > siemens_max_det)] = siemens_max_det
    return jacdet

def eval_spherical_harmonics(coeffs, vendor, vxyz):
    """
    Evaluate the spherical harmonics displacement field.
    """
    R0 = coeffs.R0_m * 1000.0
    x, y, z = vxyz
    print(np.shape(x), np.shape(y), np.shape(z))
    if vendor == 'siemens':
        log.info('along x...')
        bx = siemens_B(coeffs.alpha_x, coeffs.beta_x, x, y, z, R0)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x[:, :, 0], y[:, :, 0], bx[:, :, 0], cmap=cm.jet, linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # plt.show()
        log.info('along y...')
        by = siemens_B(coeffs.alpha_y, coeffs.beta_y, x, y, z, R0)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(x[:, :, 0], y[:, :, 0], by[:, :, 0], cmap=cm.jet, linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # plt.show()
        log.info('along z...')
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        bz = siemens_B(coeffs.alpha_z, coeffs.beta_z, x, y, z, R0)
        surf = ax.plot_surface(np.squeeze(x[0, :, :]), np.squeeze(z[0, :, :]), np.squeeze(bz[0, :, :]), cmap=cm.jet, linewidth=0, antialiased=False)
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        # plt.show()
    else:
        log.info('along x...')
        bx = ge_D(coeffs.alpha_x, coeffs.beta_x, x, y, z)
        log.info('along y...')
        by = ge_D(coeffs.alpha_y, coeffs.beta_y, x, y, z)
        log.info('along z...')
        bz = ge_D(coeffs.alpha_z, coeffs.beta_z, x, y, z)
    return CoordsVector(bx * R0, by * R0, bz * R0), CoordsVector(x, y, z)

def siemens_B(alpha, beta, x, y, z, R0):
    """
    Calculate displacement field from Siemens coefficients.
    """
    nmax = alpha.shape[0] - 1
    x = x + 0.0001  # Avoid singularities at R=0.
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(z / r)
    phi = np.arctan2(y / r, x / r)
    b = np.zeros(x.shape)
    for n in range(0, nmax + 1):
        f = np.power(r / R0, n)
        for m in range(0, n + 1):
            f2 = alpha[n, m] * np.cos(m * phi) + beta[n, m] * np.sin(m * phi)
            _ptemp = legendre(n, m, np.cos(theta))
            normfact = 1.0
            if m > 0:
                normfact = ((-1) ** m) * math.sqrt(float((2 * n + 1) * factorial(n - m)) / float(2 * factorial(n + m)))
            b = b + f * normfact * _ptemp * f2
    return b

def ge_D(alpha, beta, x, y, z):
    """
    Compute GE gradient warp displacement.
    """
    nmax = alpha.shape[0] - 1
    x = x + 0.0001
    r = np.sqrt(x * x + y * y + z * z)
    phi = np.arccos(z / r)
    theta = np.arctan2(y / r, x / r)
    r = r * 100.0  # Convert meters to cm for GE.
    d = np.zeros(x.shape)
    for n in range(0, nmax + 1):
        f = np.power(r, n)
        for m in range(0, n + 1):
            f2 = alpha[n, m] * np.cos(m * theta) + beta[n, m] * np.sin(m * theta)
            _p = legendre(n, m, np.cos(phi))
            d = d + f * _p * f2
    d = d / 100.0  # Convert cm back to meters.
    return d

####################################
#         GRADIENT_UNWARP          #
####################################

"""
gradient_unwarp: Corrects for geometric distortions in the scans.
Input: DICOM warped file and coefficient file.
Output: NIfTI files for xvals, yvals, zvals and HHimg.
"""

import argparse
import os
import gc
import re

def parse_arguments():
    """
    Parse command-line arguments for gradient unwarp.
    """
    try:
        parser = argparse.ArgumentParser(usage=usage)
        parser.add_argument('--version', '-v', action='version', version=VERSION)
    except Exception:
        parser = argparse.ArgumentParser(version=VERSION, usage=usage)
    parser.add_argument('infile', help='The input warped file (NIfTI or MGH)')
    parser.add_argument('outfile', help='The output unwarped file (extension .nii/.nii.gz/.mgh/.mgz)')
    parser.add_argument('vendor', choices=['siemens', 'ge'], help='Vendor: siemens or ge')
    coeff_group = parser.add_mutually_exclusive_group(required=True)
    coeff_group.add_argument('-g', '--gradfile', dest='gradfile', help='The .grad coefficient file')
    coeff_group.add_argument('-c', '--coeffile', dest='coeffile', help='The .coef coefficient file')
    parser.add_argument('-w', '--warp', action='store_true', default=False, help='Warp volume instead of unwarping')
    parser.add_argument('-n', '--nojacobian', dest='nojac', action='store_true', default=False, help='Do not perform Jacobian correction')
    parser.add_argument('--fovmin', dest='fovmin', help='Minimum FOV for evaluation grid in meters')
    parser.add_argument('--fovmax', dest='fovmax', help='Maximum FOV for evaluation grid in meters')
    parser.add_argument('--numpoints', dest='numpoints', help='Number of grid points in each direction')
    parser.add_argument('--interp_order', dest='order', help='Interpolation order (1..4)')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    if not os.path.exists(args.infile):
        raise IOError(args.infile + ' not found')
    if args.gradfile and not os.path.exists(args.gradfile):
        raise IOError(args.gradfile + ' not found')
    if args.coeffile and not os.path.exists(args.coeffile):
        raise IOError(args.coeffile + ' not found')
    return args

class GradientUnwarpRunner(object):
    """
    Runs the gradient unwarp process.
    """
    def __init__(self, args):
        self.args = args
        self.unwarper = None

    def run(self):
        if self.args.gradfile:
            self.coeffs = get_coefficients(self.args.vendor, self.args.gradfile)
        else:
            self.coeffs = get_coefficients(self.args.vendor, self.args.coeffile)
        from collections import namedtuple
        CoeffsTuple = namedtuple('Coeffs', 'alpha_x, alpha_y, alpha_z, beta_x, beta_y, beta_z, R0_m')
        R0_m = 0.25
        # (The following coefficient arrays are overwritten below.)
        ax = np.zeros((20,20))
        ay = np.zeros((20,20))
        az = np.zeros((20,20))
        bx = np.zeros((20,20))
        by = np.zeros((20,20))
        bz = np.zeros((20,20))
        with open("coeff.grad", "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines or comment lines starting with '#'
                if not line or line.startswith("#"):
                    continue
                # Expected line format (for coefficient lines):
                #   <serial> <Letter>( <i, j>) <value> <axis>
                # Example: "  1 A( 3, 0)           -0.05625983                  z"
                pattern = r'^\s*\d+\s+([AB])\(\s*(\d+),\s*(\d+)\)\s+([-.\d]+)\s+([xyz])'
                match = re.match(pattern, line)
                if not match:
                    continue
                coeff_letter = match.group(1)  # "A" or "B"
                i = int(match.group(2))
                j = int(match.group(3))
                value = float(match.group(4))
                axis = match.group(5)  # "x", "y", or "z"
                # For coefficients starting with A
                if coeff_letter == "A":
                    if axis == "z":
                        # For axis z, assign to az; use j=0 regardless of the file j value.
                        az[i, 0] = value
                    elif axis == "x":
                        ax[i, j] = value
                # For coefficients starting with B (in this file, they have axis y)
                elif coeff_letter == "B":
                    if axis == "y":
                        by[i, j] = value
        self.coeffs = CoeffsTuple(ax, ay, az, bx, by, bz, R0_m)
        self.vol, self.m_rcs2ras = get_vol_affine(self.args.infile)
        self.unwarper = Unwarper(self.vol, self.m_rcs2ras, self.args.vendor, self.coeffs, self.args.infile)
        if self.args.fovmin:
            self.unwarper.fovmin = float(self.args.fovmin)
        if self.args.fovmax:
            self.unwarper.fovmax = float(self.args.fovmax)
        if self.args.numpoints:
            self.unwarper.numpoints = int(self.args.numpoints)
        if self.args.warp:
            self.unwarper.warp = True
        if self.args.nojac:
            self.unwarper.nojac = True
        if self.args.order:
            self.unwarper.order = int(self.args.order)
        self.unwarper.run()

    def write(self):
        self.unwarper.write(self.args.outfile)

def geometric_unwarp_main():

    excel_path = 'Input_parametersheet_MRV.xlsx'
    if not os.path.exists(excel_path):
        print(f"Error: Excel configuration file '{excel_path}' not found.")
        return

    try:
        excel_data = pd.read_excel(excel_path, header=None).values
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return
    
    try:
        dataset_folders = [str(f).strip().split('.')[0] for f in excel_data[4, 1:] if not pd.isna(f) and str(f).strip() != '']
        velocity_labels = [str(f).strip().split('.')[0] for f in excel_data[6, 1:] if not pd.isna(f) and str(f).strip() != '']
        flow_folder_names = [str(f).strip().split('.')[0] for f in excel_data[5, 1:] if not pd.isna(f) and str(f).strip() != '']
        mask_folder_name = str(excel_data[8, 1]).strip() if pd.notna(excel_data[7, 1]) else ""
        
        if len(velocity_labels) != len(flow_folder_names):
            raise ValueError("Mismatch in the number of flow folders and labels")
    except Exception as e:
        print(f"Error extracting configuration parameters: {e}")
        return

    # List of dataset folders to process
    #dataset_folders = ['test dataset']

    # List to hold velocity labels (for flow files) - adjust as needed.
    #velocity_labels = ['93','97','104']

    # List of folder names corresponding to each flow series.
    #flow_folder_names = ['93_flow_pc3d_P_ND','97_flow_pc3d_P_ND','104_flow_pc3d_P_ND']

    # Folder name containing the mask DICOM series
    #mask_folder_name = '89_flow_pc3d_ND'

    for dataset in dataset_folders:
        # Convert the mask DICOM series to a NIfTI file.
        if mask_folder_name:
            dicom2nifti.dicom_series_to_nifti(
                os.path.join(dataset, mask_folder_name),
                os.path.join(dataset, 'mask.nii'),
                reorient_nifti=False
            )

            # Set up arguments for unwarping the mask.
            mask_args = Namespace(
                coeffile='prisma1.coef.txt',
                fovmax=None,
                fovmin=None,
                gradfile=None,
                infile=os.path.join(dataset, 'mask.nii'),
                nojac=True,
                numpoints=None,
                order=None,
                outfile=os.path.join(dataset, 'mask_corrt.nii'),
                vendor='siemens',
                verbose=False,
                warp=False
            )

            # Unwarp the mask.
            mask_unwarper = GradientUnwarpRunner(mask_args)
            try:
                mask_unwarper.run()
                mask_unwarper.write()
            except Exception as error:
                print(f"Error processing mask: {error}")
            del mask_unwarper  # Free resources

        # Process each flow folder.
        for idx, flow_folder in enumerate(flow_folder_names):
            # Convert the DICOM series for the current flow to a NIfTI file.
            dicom2nifti.dicom_series_to_nifti(
                os.path.join(dataset, flow_folder),
                os.path.join(dataset, f'{velocity_labels[idx]}.nii'),
                reorient_nifti=False
            )

            # Set up arguments for unwarping the current flow.
            flow_args = Namespace(
                coeffile='prisma1.coef.txt',
                fovmax=None,
                fovmin=None,
                gradfile=None,
                infile=os.path.join(dataset, f'{velocity_labels[idx]}.nii'),
                nojac=True,
                numpoints=None,
                order=None,
                outfile=os.path.join(dataset, f'{velocity_labels[idx]}_corrt.nii'),
                vendor='siemens',
                verbose=False,
                warp=False
            )

            # Unwarp the current flow.
            flow_unwarper = GradientUnwarpRunner(flow_args)
            try:
                flow_unwarper.run()
                flow_unwarper.write()
            except Exception as error:
                print(f"Error processing flow {velocity_labels[idx]}: {error}")
            del flow_unwarper  # Free resources

    # Optionally force garbage collection (if processing large datasets)
    gc.collect()


if __name__ == "__main__":
    geometric_unwarp_main()
