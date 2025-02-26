#!/usr/bin/env python3
"""
This Python script replicates the MATLAB code for computing the Legendre coefficients 
for gradient coil design (AS82), creating a displacement table over a grid and then 
computing gradients. All logic, comments, and coefficient values have been preserved.
All results are saved as .npy files.
"""

import numpy as np
import math
import scipy.special
import re

siemens_fovmin = -0.18  # fov min in meters
siemens_fovmax = 0.18   # fov max in meters

#---------------------------------------------------------------------
# Function: siemens_legendre
# Description:
#   This function replicates the MATLAB siemens_legendre.m. It calls the 
#   associated Legendre function (using scipy.special.lpmv) for degree n and 
#   orders 0,...,n, then for m>=1 multiplies each row by:
#      normfact = (-1)^m * sqrt((2*n+1)*factorial(n-m)/(2*factorial(n+m)))
#   to mimic the MATLAB code.
#---------------------------------------------------------------------
def siemens_legendre(n, X):
    """
    Compute the associated Legendre functions for degree n evaluated at X.
    Parameters:
      n : integer, the degree.
      X : array-like (typically cos(Theta)); must be in [-1,1].
    Returns:
      P : 2D numpy array of shape (n+1, len(X)), where P[m, :] is P_n^m(X)
          with the additional normalization for m>=1.
    """
    X = np.array(X)
    P = np.empty((n+1, X.size))
    for m in range(0, n+1):
        # Compute associated Legendre function P_n^m(x)
        P[m, :] = scipy.special.lpmv(m, n, X)
        if m > 0:
            normfact = ((-1)**m) * math.sqrt((2*n+1) * math.factorial(n-m) / (2 * math.factorial(n+m)))
            P[m, :] *= normfact
    return P

#---------------------------------------------------------------------
# Function: siemens_B
# Description:
#   Replicates the MATLAB function siemens_B.m.
#   It adjusts X to avoid singularities, computes spherical coordinates R, Theta, Phi,
#   and then loops over n=0:nmax and m=0:n to compute the displacement field B.
#   Finally, B is scaled by R0 and Gref.
#---------------------------------------------------------------------
def siemens_B(Alpha, Beta, X, Y, Z, R0, Gref):
    """
    Compute the non-dimensional displacement field B given coefficient matrices.
    Parameters:
      Alpha, Beta : 2D numpy arrays of size (nmax+1, nmax+1)
      X, Y, Z     : 1D numpy arrays of the grid coordinates (flattened)
      R0          : scalar reference length
      Gref        : scalar gradient reference (e.g., in mT/m)
    Returns:
      B : 1D numpy array of computed displacement field values (same length as X)
    """
    # Add a small offset to X to avoid division by zero (singularities)
    X = X + 1e-4  # 0.0001 as in MATLAB
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Theta = np.arccos(Z / R)
    Phi = np.arctan2(Y / R, X / R)
    nmax = Alpha.shape[0] - 1
    B = np.zeros_like(X)
    for n in range(0, nmax+1):
        # P: associated Legendre functions of degree n evaluated at cos(Theta)
        P = siemens_legendre(n, np.cos(Theta))  # shape: (n+1, len(X))
        F = (R / R0)**n  # radial factor
        for m in range(0, n+1):
            # Compute contribution from order m
            F2 = Alpha[n, m] * np.cos(m * Phi) + Beta[n, m] * np.sin(m * Phi)
            B += F * P[m, :] * F2
    # Scale to obtain displacement field in meters then multiplied by Gref
    B = B * R0 * Gref
    return B

#---------------------------------------------------------------------
# Function: create_displacement_table_for_siemens_coords
# Description:
#   Replicates create_diplacement_table_for_siemens_coords.m.
#   It creates a grid over the range [-0.18,0.18] with spacing 0.001, saves X, Y, Z 
#   as .npy files, reshapes the grid into 1D arrays, sets scanner-specific parameters 
#   based on the table_name substring, computes Bx, By, Bz (via siemens_B) for each gradient,
#   reshapes them back to 3D grid, computes gradients with np.gradient, and saves 
#   both B-fields and gradient components as .npy files.
#---------------------------------------------------------------------
def create_displacement_table_for_siemens_coords(Alpha_x, Alpha_y, Alpha_z,
                                                  Beta_x, Beta_y, Beta_z, table_name):
    res = 0.001
    # Create grid values from -0.18 to 0.18 (inclusive)
    coordvals = np.arange(siemens_fovmin, siemens_fovmax + res, res)
    ncoords = coordvals.size
    # Use ndgrid equivalent: meshgrid with indexing='ij'
    X_grid, Y_grid, Z_grid = np.meshgrid(coordvals, coordvals, coordvals, indexing='ij')
    # Save grid coordinates as .npy files
    np.save("X.npy", X_grid)
    np.save("Y.npy", Y_grid)
    np.save("Z.npy", Z_grid)
    # Flatten the grids into 1D arrays for displacement computation
    X_flat = X_grid.ravel()
    Y_flat = Y_grid.ravel()
    Z_flat = Z_grid.ravel()
    dx = coordvals[1] - coordvals[0]
    dy = dx
    dz = dx

    # Set parameters based on table_name
    if 'sonata.gwv' in table_name:
        R0 = 0.25
        Gref_x = 33.001346e-3  # in uT/m if desired; see comment in MATLAB
        Gref_y = 32.974642e-3
        Gref_z = 32.065914e-3
    elif 'avanto.gwv' in table_name:
        R0 = 0.25
        Gref_x = 40.0 * 10
        Gref_y = 40.0 * 10
        Gref_z = 40.0 * 10
    elif 'allegra.gwv' in table_name:
        R0 = 0.14
        Gref_x = 5.999433
        Gref_y = 6.468824
        Gref_z = 6.904833
    elif 'prisma1.gwv' in table_name:
        R0 = 0.25
        Gref_x = 80
        Gref_y = 80
        Gref_z = 80
    else:
        # Default parameters (set as in prisma1)
        R0 = 0.25
        Gref_x = 80
        Gref_y = 80
        Gref_z = 80

    print("\n\nCreating displacement table ...")
    print("start calculations ...")
    Bx_flat = siemens_B(Alpha_x, Beta_x, X_flat, Y_flat, Z_flat, R0, Gref_x)
    print("computed Bx")
    By_flat = siemens_B(Alpha_y, Beta_y, X_flat, Y_flat, Z_flat, R0, Gref_y)
    print("computed By")
    Bz_flat = siemens_B(Alpha_z, Beta_z, X_flat, Y_flat, Z_flat, R0, Gref_z)
    # Reshape back to grid shape
    shape = (ncoords, ncoords, ncoords)
    Bx = Bx_flat.reshape(shape)
    By = By_flat.reshape(shape)
    Bz = Bz_flat.reshape(shape)
    X_grid = X_flat.reshape(shape)
    Y_grid = Y_flat.reshape(shape)
    Z_grid = Z_flat.reshape(shape)
    print("computed Bz")
    # Compute gradients using numpy.gradient.
    # To mimic MATLAB's gradient(Bx, dy, dx, dz) where MATLAB returns
    # [dbxdy, dbxdx, dbxdz], we call np.gradient with spacings (dx, dy, dz)
    # so that axis0 derivative uses dx (i.e. corresponds to x), axis1 uses dy (i.e. y), etc.
    grad_Bx = np.gradient(Bx, dx, dy, dz)
    grad_By = np.gradient(By, dx, dy, dz)
    grad_Bz = np.gradient(Bz, dx, dy, dz)
    # In our grid (with indexing='ij'), axis0 corresponds to x and axis1 to y.
    # MATLAB expects dbxdy (y-derivative) as the first output and dbxdx (x-derivative) as the second.
    # So we swap the first two outputs:
    dbxdx, dbxdy, dbxdz = grad_Bx
    dbydx, dbydy, dbydz = grad_By
    dbzdx, dbzdy, dbzdz = grad_Bz
    # Save computed fields and gradients as .npy files.
    np.save("Bx.npy", Bx)
    np.save("By.npy", By)
    np.save("Bz.npy", Bz)
    np.save("dbxdx.npy", dbxdx)
    np.save("dbxdy.npy", dbxdy)
    np.save("dbxdz.npy", dbxdz)
    np.save("dbydx.npy", dbydx)
    np.save("dbydy.npy", dbydy)
    np.save("dbydz.npy", dbydz)
    np.save("dbzdx.npy", dbzdx)
    np.save("dbzdy.npy", dbzdy)
    np.save("dbzdz.npy", dbzdz)
    return Bx, By, Bz, X_grid, Y_grid, Z_grid

#---------------------------------------------------------------------
# Main function (replicates main.m)
#---------------------------------------------------------------------
def bfield_main():
    # Reference parameters (as in MATLAB main.m)
    R0 = 0.25         # m
    Gref_x = 80       # mT/m
    Gref_y = 80       # mT/m
    Gref_z = 80       # mT/m
    coef_array_size = 20

    # Initialize coefficient arrays (all zeros)
    Alpha_x = np.zeros((coef_array_size, coef_array_size))
    Alpha_y = np.zeros((coef_array_size, coef_array_size))
    Alpha_z = np.zeros((coef_array_size, coef_array_size))
    Beta_x  = np.zeros((coef_array_size, coef_array_size))
    Beta_y  = np.zeros((coef_array_size, coef_array_size))
    Beta_z  = np.zeros((coef_array_size, coef_array_size))

    # Set nonzero coefficients (MATLAB indices adjusted to Python 0-indexing)
    # For Alpha_z:
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
                    Alpha_z[i, 0] = value
                elif axis == "x":
                    Alpha_x[i, j] = value
            # For coefficients starting with B (in this file, they have axis y)
            elif coeff_letter == "B":
                if axis == "y":
                    Beta_y[i, j] = value

    # Other coefficient arrays (Alpha_y, Beta_x, Beta_z) remain zero.
    table_name = 'prisma1.gwv'
    # Call the function that creates the displacement table and computes gradients.
    Bx, By, Bz, X, Y, Z = create_displacement_table_for_siemens_coords(
        Alpha_x, Alpha_y, Alpha_z, Beta_x, Beta_y, Beta_z, table_name)
    
    # Test: print the center value of Bx to check consistency.
    center_index = (Bx.shape[0]//2, Bx.shape[1]//2, Bx.shape[2]//2)
    print("Center value of Bx:", Bx[center_index])

#---------------------------------------------------------------------
# Run main if executed as script
#---------------------------------------------------------------------
if __name__ == '__main__':
    bfield_main()
