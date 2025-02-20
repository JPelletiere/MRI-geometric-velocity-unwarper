#!/usr/bin/env python3
"""
This script performs velocity unwarping (geometry correction) for MRI data.
It is based on MATLAB code that:
  - Reads in an MRI volume (either from corrected NIfTI files or raw DICOM files),
  - Creates a mask based on an intensity threshold,
  - Loads phase images (flow on/off) and subtracts them,
  - Loads coordinate maps and gradient fields (now as .npy files),
  - Applies the Markl correction (or direct scaling if Markl is not enabled) to compute
    corrected velocity fields U, V, and W.
  - Saves the corrected coordinate maps and velocity fields as NIfTI files.

Parameters you may wish to adjust at the top of the main processing function (velocity_unwarping_main):
  • USE_CORRECTED: Set True to use corrected geometry files (NIfTI files ending with "_corrt"),
    or False to use raw DICOM data.
  • The Excel file path ('Input_parametersheet_MRV.xlsx') – this must match your configuration.
  • The intensity threshold used for mask creation (currently set to 200).
  • The file/folder patterns for the phase (flow on/off) and magnitude images remain unchanged.
  
This script is compatible with Python 3.7.
  
Required packages: os, glob, numpy, pandas, scipy.io, pydicom, nibabel, and scipy.interpolate.
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import pydicom
import nibabel as nib
from dataclasses import dataclass
from typing import List, Tuple
from scipy.interpolate import interpn
import math

# ------------------------ Data Classes ------------------------ #

@dataclass
class DicomFile:
    name: str
    folder: str = ''

# ------------------------ Utility Functions ------------------------ #

def mask_simple(imags: np.ndarray, jo: int, thresh: float) -> np.ndarray:
    """
    Creates a binary mask based on a single intensity threshold.
    """
    return imags > thresh

def dread(S: List[DicomFile], folder: str, ini: int, noofimages: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads 3D volumetric DICOM images from a specified folder.
    Inputs:
      S: list of DicomFile objects (with attribute 'name')
      folder: folder path containing the DICOM files
      ini: the first image index to read (1-indexed as in MATLAB)
      noofimages: number of slices to read
    Returns:
      I: 3D numpy array of intensities (uint16) with shape (nrows, ncols, noofimages)
      X, Y, Z: coordinate arrays (float64) with shape (nrows, ncols, noofimages)
    """
    N = [item.name for item in S]
    print(f"Total DICOM files available for reading: {len(N)}")
    if ini - 1 + noofimages > len(N):
        raise IndexError(f"Requested image index {ini - 1 + noofimages - 1} exceeds available DICOM files ({len(N)}).")

    initial_filename = os.path.join(folder, N[ini - 1])
    info = pydicom.dcmread(initial_filename)
    nrows = info.Rows
    ncols = info.Columns

    I = np.zeros((nrows, ncols, noofimages), dtype=np.uint16)
    X = np.zeros((nrows, ncols, noofimages), dtype=np.float64)
    Y = np.zeros((nrows, ncols, noofimages), dtype=np.float64)
    Z = np.zeros((nrows, ncols, noofimages), dtype=np.float64)

    for i in range(1, noofimages + 1):
        idx = ini + i - 2  # adjust for MATLAB's 1-indexing
        if idx >= len(N):
            raise IndexError(f"Requested image index {idx + 1} exceeds available DICOM files.")
        n = N[idx]
        current_filename = os.path.join(folder, n)
        info = pydicom.dcmread(current_filename)
        ps = info.PixelSpacing  # [row spacing, column spacing]
        ipp = info.ImagePositionPatient  # [x, y, z]
        orientation_cosines = info.ImageOrientationPatient  # [cos1, cos2, cos3, cos4, cos5, cos6]
        row_cosines = np.array(orientation_cosines[:3])
        col_cosines = np.array(orientation_cosines[3:])
        # Create meshgrid for pixel indices
        x_indices = np.arange(ncols)
        y_indices = np.arange(nrows)
        X_grid, Y_grid = np.meshgrid(x_indices, y_indices)
        # Calculate spatial coordinates (in meters)
        X_coord = ipp[0] + (X_grid * float(ps[0]) * row_cosines[0]) + (Y_grid * float(ps[1]) * col_cosines[0])
        Y_coord = ipp[1] + (X_grid * float(ps[0]) * row_cosines[1]) + (Y_grid * float(ps[1]) * col_cosines[1])
        Z_coord = ipp[2] + (X_grid * float(ps[0]) * row_cosines[2]) + (Y_grid * float(ps[1]) * col_cosines[2])
        X[:, :, i - 1] = X_coord
        Y[:, :, i - 1] = Y_coord
        Z[:, :, i - 1] = Z_coord
        I[:, :, i - 1] = info.pixel_array.astype(np.uint16)
    return I, X, Y, Z

#---------------------------------------------------------------------
# Mathematical Functions for Velocity Correction (Unwarping)
#---------------------------------------------------------------------
def siemens_legendre(n: int, X) -> np.ndarray:
    """
    Computes the associated Legendre functions for degree n evaluated at X.
    Mimics MATLAB's siemens_legendre.m:
      For each order m=0,...,n, uses scipy.special.lpmv.
      For m>=1, multiplies by normfact = (-1)^m * sqrt((2*n+1)*factorial(n-m)/(2*factorial(n+m))).
    X should be in [-1,1].
    Returns:
      P: array of shape (n+1, len(X))
    """
    X = np.array(X)
    P = np.empty((n+1, X.size))
    for m in range(n+1):
        P[m, :] = scipy.special.lpmv(m, n, X)
        if m > 0:
            normfact = ((-1)**m) * math.sqrt((2*n+1) * math.factorial(n-m) / (2 * math.factorial(n+m)))
            P[m, :] *= normfact
    return P

def siemens_B(Alpha: np.ndarray, Beta: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
              R0: float, Gref: float) -> np.ndarray:
    """
    Replicates MATLAB's siemens_B.m.
    Computes the non-dimensional displacement field B from the coefficient matrices.
    X, Y, Z are 1D arrays (flattened grid). The function computes spherical coordinates
    (R, Theta, Phi) and sums over degree and order.
    Finally, B is scaled by R0 and Gref.
    """
    # Offset X to avoid singularity (R==0)
    X = X + 1e-4
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Theta = np.arccos(Z / R)
    Phi = np.arctan2(Y / R, X / R)
    nmax = Alpha.shape[0] - 1
    B = np.zeros_like(X)
    for n in range(nmax+1):
        P = siemens_legendre(n, np.cos(Theta))  # shape (n+1, len(X))
        F = (R / R0)**n
        for m in range(n+1):
            F2 = Alpha[n, m] * np.cos(m * Phi) + Beta[n, m] * np.sin(m * Phi)
            B += F * P[m, :] * F2
    B = B * R0 * Gref
    return B

def markl_correction(I_all: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, v_enc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies Markl correction to compute corrected velocity fields.
    Instead of loading gradients from .mat files, this function now loads them from .npy files.
    It also loads coordinate grids (X, Y, Z) from .npy files.
    The gradients are interpolated onto the provided (X,Y,Z) grid using interpn.
    Then, for each voxel, the velocity correction is computed as:
         tempk = inv(ap * gma)' * [I_all(im,jm,km,3); I_all(im,jm,km,2); I_all(im,jm,km,1)]
    where gma = diag([4095/v_enc[3], 4095/v_enc[2], 4095/v_enc[1]]) (adjusting for 0-indexing).
    Inputs:
      I_all: 4D phase image array with shape (dim1, dim2, dim3, num_channels)
      X, Y, Z: coordinate arrays (3D) that define the target grid
      v_enc: 3-element array for velocity encoding (order: [v_enc1, v_enc2, v_enc3])
    Returns:
      U, V, W: corrected velocity fields (3D arrays)
    """
    # Load gradients from .npy files
    gxx = np.load('dbxdx.npy') / 80.0
    gxy = np.load('dbydx.npy') / 80.0
    gxz = np.load('dbzdx.npy') / 80.0
    gyx = np.load('dbxdy.npy') / 80.0
    gyy = np.load('dbydy.npy') / 80.0
    gyz = np.load('dbzdy.npy') / 80.0
    gzx = np.load('dbxdz.npy') / 80.0
    gzy = np.load('dbydz.npy') / 80.0
    gzz = np.load('dbzdz.npy') / 80.0

    # Load coordinate grids from .npy files
    XX = np.load('X.npy')
    YY = np.load('Y.npy')
    ZZ = np.load('Z.npy')
    print("Shapes (markl_correction): gxx:", gxx.shape, "XX:", XX.shape, "Target X:", X.shape)
    # Build the interpolation grid (assumed regular)
    points = (XX[:,0,0], YY[0,:,0], ZZ[0,0,:])
    # Interpolate gradients onto target grid
    gxx_interp = interpn(points, gxx, np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1),
                          method='linear', bounds_error=False, fill_value=None).reshape(X.shape)
    gxy_interp = interpn(points, gxy, np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1),
                          method='linear', bounds_error=False, fill_value=None).reshape(X.shape)
    gxz_interp = interpn(points, gxz, np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1),
                          method='linear', bounds_error=False, fill_value=None).reshape(X.shape)
    gyx_interp = interpn(points, gyx, np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1),
                          method='linear', bounds_error=False, fill_value=None).reshape(X.shape)
    gyy_interp = interpn(points, gyy, np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1),
                          method='linear', bounds_error=False, fill_value=None).reshape(X.shape)
    gyz_interp = interpn(points, gyz, np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1),
                          method='linear', bounds_error=False, fill_value=None).reshape(X.shape)
    gzx_interp = interpn(points, gzx, np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1),
                          method='linear', bounds_error=False, fill_value=None).reshape(X.shape)
    gzy_interp = interpn(points, gzy, np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1),
                          method='linear', bounds_error=False, fill_value=None).reshape(X.shape)
    gzz_interp = interpn(points, gzz, np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1),
                          method='linear', bounds_error=False, fill_value=None).reshape(X.shape)
    
    # Build encoding diagonal matrix.
    # Note: MATLAB: diag([4095/v_enc(3), 4095/v_enc(2), 4095/v_enc(1)])
    # Python (0-indexed): v_enc[2] is v_enc(3), etc.
    gma = np.diag([4095.0 / v_enc[2], 4095.0 / v_enc[1], 4095.0 / v_enc[0]])
    
    U = np.zeros(X.shape, dtype=np.float64)
    V = np.zeros(X.shape, dtype=np.float64)
    W = np.zeros(X.shape, dtype=np.float64)
    nx, ny, nz = X.shape
    for im in range(nx):
        for jm in range(ny):
            for km in range(nz):
                ap = np.array([[gxx_interp[im, jm, km], gxy_interp[im, jm, km], gxz_interp[im, jm, km]],
                               [gyx_interp[im, jm, km], gyy_interp[im, jm, km], gyz_interp[im, jm, km]],
                               [gzx_interp[im, jm, km], gzy_interp[im, jm, km], gzz_interp[im, jm, km]]])
                try:
                    tempk = np.linalg.inv(ap @ gma).T @ np.array([I_all[im, jm, km, 2],
                                                                   I_all[im, jm, km, 1],
                                                                   I_all[im, jm, km, 0]])
                except np.linalg.LinAlgError:
                    tempk = np.array([0.0, 0.0, 0.0])
                U[im, jm, km] = tempk[0]
                V[im, jm, km] = tempk[1]
                W[im, jm, km] = tempk[2]
    return U, V, W

def writeTecplotBin(output_filename: str, file_title: str, zone_name: str, **variables):
    """
    Writes data to a Tecplot Binary (.plt) file.
    (A simplified placeholder implementation.)
    """
    import struct
    num_vars = len(variables)
    var_names = list(variables.keys())
    var_data = list(variables.values())
    reference_shape = var_data[0].shape
    if not all(v.shape == reference_shape for v in var_data):
        raise ValueError("All variable arrays must have the same shape.")
    with open(output_filename, 'wb') as fid:
        def write_null_term_string(s):
            for ch in s:
                fid.write(struct.pack('<i', ord(ch)))
            fid.write(struct.pack('<i', 0))
        fid.write(b'#!TDV112')
        fid.write(struct.pack('<i', 1))
        fid.write(struct.pack('<i', 0))
        write_null_term_string(file_title)
        fid.write(struct.pack('<i', num_vars))
        for name in var_names:
            write_null_term_string(name)
        # (Zone header and data writing omitted for brevity.)
        fid.write(struct.pack('<f', 357.0))
    print(f"Tecplot Binary file '{output_filename}' has been created.")

# ------------------------ Main Processing Function ------------------------ #

def velocity_unwarping_main():
    """
    Main processing function.
    
    This function reads parameters from an Excel sheet ('Input_parametersheet_MRV.xlsx'),
    loads either corrected geometry NIfTI files or raw DICOM files (based on USE_CORRECTED flag),
    creates a binary mask from the magnitude image, loads phase images (flow on/off),
    reads coordinate maps (from .npy files), applies velocity corrections (using Markl correction if enabled),
    subtracts the no-flow data from the flow data, and saves the corrected velocity fields
    (U, V, W) and coordinates (X, Y, Z) as NIfTI files. In addition, it writes a Tecplot Binary (.plt)
    file with these variables.
    
    Adjustable parameters:
      - USE_CORRECTED (True/False): If True, uses corrected NIfTI files (filenames ending with '_corrt');
        if False, uses raw DICOM files via dread.
      - Excel file path ('Input_parametersheet_MRV.xlsx'): Change if needed.
      - Mask threshold (currently 200).
    """
    USE_CORRECTED = True
    excel_path = 'Input_parametersheet_MRV.xlsx'
    if not os.path.exists(excel_path):
        print(f"Error: Excel file '{excel_path}' not found.")
        return
    try:
        excel_data = pd.read_excel(excel_path, header=None).values
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return
    
    try:
        phase = int(excel_data[9, 1])
        fname = [str(f).strip().split('.')[0] for f in excel_data[10, 1:] if not pd.isna(f) and str(f).strip() != '']
        fnamenf = [str(f).strip().split('.')[0] for f in excel_data[11, 1:] if not pd.isna(f) and str(f).strip() != '']
        magsname = int(excel_data[12, 1])
        noofslices = int(excel_data[13, 1])
        venc = [float(v) for v in excel_data[14, 1:] if not pd.isna(v)]
        markl = int(excel_data[15, 1])
        if len(fname) != len(venc) or len(fnamenf) != len(venc):
            raise ValueError("Mismatch in the number of fname, fnamenf, or venc entries.")
    except Exception as e:
        print(f"Error extracting configuration parameters: {e}")
        return
    
    print("Configuration Parameters:")
    print(f"Phase: {phase}")
    print(f"Flow On IDs (fname): {fname}")
    print(f"Flow Off IDs (fnamenf): {fnamenf}")
    print(f"Magsum ID (magsname): {magsname}")
    print(f"Number of Slices: {noofslices}")
    print(f"Venc: {venc}")
    print(f"Markl Correction Flag: {markl}")
    
    if USE_CORRECTED:
        print("Using corrected NIfTI files.")
        base_folder = os.path.join('.', 'test dataset')
        mask_file = os.path.join(base_folder, "mask_corrt.nii")
        try:
            Imags = nib.load(mask_file).get_fdata().astype(np.float64)
            print(f"Loaded corrected mask image from '{mask_file}' with shape {Imags.shape}.")
        except Exception as e:
            print(f"Error loading corrected mask image: {e}")
            return
        flow_off_ids = ["93", "104", "97"]
        I_all_nf_list = []
        for fid in flow_off_ids:
            file_path = os.path.join(base_folder, f"{fid}_corrt.nii")
            try:
                img = nib.load(file_path).get_fdata().astype(np.float64)
                if img.shape != Imags.shape:
                    print(f"Flow off image {fid}: shape {img.shape} does not match mask shape {Imags.shape}; transposing...")
                    img = np.transpose(img, (1, 0, 2))
                I_all_nf_list.append(img)
                print(f"Loaded corrected flow off image '{file_path}' with shape {img.shape}.")
            except Exception as e:
                print(f"Error loading corrected flow off image '{file_path}': {e}")
                return
        I_all_nf = np.stack(I_all_nf_list, axis=-1)
        flow_on_ids = ["126", "130", "136"]
        I_all_list = []
        for fid in flow_on_ids:
            file_path = os.path.join(base_folder, f"{fid}_corrt.nii")
            try:
                img = nib.load(file_path).get_fdata().astype(np.float64)
                if img.shape != Imags.shape:
                    print(f"Flow on image {fid}: shape {img.shape} does not match mask shape {Imags.shape}; transposing...")
                    img = np.transpose(img, (1, 0, 2))
                I_all_list.append(img)
                print(f"Loaded corrected flow on image '{file_path}' with shape {img.shape}.")
            except Exception as e:
                print(f"Error loading corrected flow on image '{file_path}': {e}")
                return
        I_all = np.stack(I_all_list, axis=-1)
        try:
            X_img = nib.load(os.path.join(base_folder, "xvals.nii"))
            Y_img = nib.load(os.path.join(base_folder, "yvals.nii"))
            Z_img = nib.load(os.path.join(base_folder, "zvals.nii"))
            X_corr = X_img.get_fdata()
            Y_corr = Y_img.get_fdata()
            Z_corr = Z_img.get_fdata()
            print("Original coordinate shapes:", X_corr.shape, Y_corr.shape, Z_corr.shape)
        except Exception as e:
            print(f"Error loading coordinate NIfTI files: {e}")
            return
        # Swap first two axes to match mask orientation.
        X_corr_final = np.transpose(X_corr, (1, 0, 2))
        Y_corr_final = np.transpose(Y_corr, (1, 0, 2))
        Z_corr_final = np.transpose(Z_corr, (1, 0, 2))
        print("Permuted coordinate shapes:", X_corr_final.shape, Y_corr_final.shape, Z_corr_final.shape)
        
        try:
            U = I_all[:, :, :, 0] * np.array(venc)[0] / 4095.0
            V = I_all[:, :, :, 1] * np.array(venc)[1] / 4095.0
            W = I_all[:, :, :, 2] * np.array(venc)[2] / 4095.0
            U_nf = I_all_nf[:, :, :, 0] * np.array(venc)[0] / 4095.0
            V_nf = I_all_nf[:, :, :, 1] * np.array(venc)[1] / 4095.0
            W_nf = I_all_nf[:, :, :, 2] * np.array(venc)[2] / 4095.0
            print("Flow data scaled without Markl correction.")
        except Exception as e:
            print(f"Error during scaling: {e}")
            return
        U = U - U_nf
        V = V - V_nf
        W = W - W_nf
        print("No-flow data subtracted.")
        slices, cols, rows = Imags.shape
        bith = np.zeros((slices, cols, rows), dtype=np.float64)
        for j in range(slices):
            bith[j, :, :] = mask_simple(Imags[j, :, :], j+1, 200).astype(float)
        
        try:
            mask_nii = nib.load(mask_file)
            affine = mask_nii.affine
            os.makedirs('result', exist_ok=True)
            nib.save(nib.Nifti1Image(X_corr_final, affine), os.path.join('result', 'X_phase1.nii'))
            nib.save(nib.Nifti1Image(Y_corr_final, affine), os.path.join('result', 'Y_phase1.nii'))
            nib.save(nib.Nifti1Image(Z_corr_final, affine), os.path.join('result', 'Z_phase1.nii'))
            print("Coordinate NIfTI files saved.")
            nib.save(nib.Nifti1Image(U, affine), os.path.join('result', 'U_phase1.nii'))
            nib.save(nib.Nifti1Image(V, affine), os.path.join('result', 'V_phase1.nii'))
            nib.save(nib.Nifti1Image(W, affine), os.path.join('result', 'W_phase1.nii'))
            print("Velocity component NIfTI files saved.")
            nib.save(nib.Nifti1Image(bith, affine), os.path.join('result', 'Mask_phase1.nii'))
            print("Mask NIfTI file saved.")
        except Exception as e:
            print(f"Error saving NIfTI files: {e}")
            return
        
        # Write Tecplot Binary (.plt) file as well.
        try:
            # Here we use the coordinate maps before permutation (or after, as desired)
            # For example, we use X_corr_final, Y_corr_final, Z_corr_final, and U, V, W.
            writeTecplotBin(os.path.join('result', 'velocity_data.plt'),
                            'MRI test section',
                            'Phase 1 Zone',
                            X=X_corr_final, Y=Y_corr_final, Z=Z_corr_final,
                            U=U, V=V, W=W)
        except Exception as e:
            print(f"Error writing Tecplot file: {e}")
        
        print("Corrected mode processing completed successfully. NIfTI and Tecplot files are saved in 'result'.")
    
    else:
        # ------------------ RAW DICOM Mode (using dread) ------------------ #
        print("Using raw DICOM files with dread.")
        for ph in range(1, phase + 1):
            print(f"\nProcessing Phase {ph}/{phase} (RAW DICOM)...")
            mags_pattern = os.path.join('test dataset', f"{magsname}_*")
            smags_dirs = glob.glob(mags_pattern)
            if not smags_dirs:
                print(f"Error: No directories found matching pattern '{mags_pattern}'.")
                return
            smags1 = smags_dirs[0]
            print(f"Found magsum directory: '{smags1}'")
            sfile_pattern = os.path.join(smags1, '**', '*')
            sfile_paths = glob.glob(sfile_pattern, recursive=True)
            sfile_paths = [f for f in sfile_paths if os.path.isfile(f)]
            if not sfile_paths:
                print(f"Error: No files found in directory '{smags1}'.")
                return
            sfile_dicom = [DicomFile(name=os.path.basename(f), folder=os.path.dirname(f)) for f in sfile_paths]
            available_images = len(sfile_dicom)
            noofimages = min(available_images, noofslices)
            try:
                Imags, X, Y, Z = dread(sfile_dicom, smags1 + os.sep, 1, noofimages)
                print("Magsum images loaded successfully using dread.")
            except Exception as e:
                print(f"Error reading magsum images: {e}")
                return
            
            X_new = X
            Y_new = Y
            Z_new = Z
            slices, cols, rows = Imags.shape
            bith = np.zeros((slices, cols, rows), dtype=np.float64)
            for j in range(slices):
                bith[j, :, :] = mask_simple(Imags[j, :, :], j+1, 200).astype(float)
            print("Binary mask created.")
            num_components = len(fname)
            I_all = np.zeros((slices, cols, rows, num_components), dtype=np.int16)
            I_all_nf = np.zeros((slices, cols, rows, num_components), dtype=np.int16)
            for i in range(num_components):
                flow_on_pattern = os.path.join('test dataset', f"{fname[i]}_*")
                ffolder_on_dirs = glob.glob(flow_on_pattern)
                if not ffolder_on_dirs:
                    print(f"Warning: No directories found matching '{flow_on_pattern}'. Skipping component {i+1}.")
                    continue
                ffolder_on = ffolder_on_dirs[0]
                print(f"Found flow on directory: '{ffolder_on}'")
                ffile_on_pattern = os.path.join(ffolder_on, '**', '*')
                ffile_on_paths = glob.glob(ffile_on_pattern, recursive=True)
                ffile_on_paths = [f for f in ffile_on_paths if os.path.isfile(f)]
                if not ffile_on_paths:
                    print(f"Warning: No files in '{ffolder_on}'. Skipping component {i+1}.")
                    continue
                ffile_on_dicom = [DicomFile(name=os.path.basename(f), folder=os.path.dirname(f)) for f in ffile_on_paths]
                current_ini_on = 1 + (ph - 1) * noofimages
                try:
                    emni_on, _, _, _ = dread(ffile_on_dicom, ffolder_on + os.sep, current_ini_on, noofimages)
                    I_all[:, :, :, i] = (emni_on.astype(np.int16) * 2) - 4095
                    print(f"Flow on images loaded for component {i+1}.")
                except Exception as e:
                    print(f"Error reading flow on images for component {i+1}: {e}")
                    continue
                flow_off_pattern = os.path.join('test dataset', f"{fnamenf[i]}_*")
                ffolder_off_dirs = glob.glob(flow_off_pattern)
                if not ffolder_off_dirs:
                    print(f"Warning: No directories found matching '{flow_off_pattern}'. Skipping component {i+1}.")
                    continue
                ffolder_off = ffolder_off_dirs[0]
                print(f"Found flow off directory: '{ffolder_off}'")
                ffile_off_pattern = os.path.join(ffolder_off, '**', '*')
                ffile_off_paths = glob.glob(ffile_off_pattern, recursive=True)
                ffile_off_paths = [f for f in ffile_off_paths if os.path.isfile(f)]
                if not ffile_off_paths:
                    print(f"Warning: No files in '{ffolder_off}'. Skipping component {i+1}.")
                    continue
                ffile_off_dicom = [DicomFile(name=os.path.basename(f), folder=os.path.dirname(f)) for f in ffile_off_paths]
                try:
                    emni_off, _, _, _ = dread(ffile_off_dicom, ffolder_off + os.sep, 1, noofimages)
                    I_all_nf[:, :, :, i] = (emni_off.astype(np.int16) * 2) - 4095
                    print(f"Flow off images loaded for component {i+1}.")
                except Exception as e:
                    print(f"Error reading flow off images for component {i+1}: {e}")
                    continue
            try:
                x_np = np.load('xvals.npy')
                y_np = np.load('yvals.npy')
                z_np = np.load('zvals.npy')
                # Mimic MATLAB's permute(X, [3,2,1])
                X_corr = np.transpose(x_np, (2, 1, 0))
                Y_corr = np.transpose(y_np, (2, 1, 0))
                Z_corr = np.transpose(z_np, (2, 1, 0))
                print("Coordinate maps loaded from .npy files.")
            except Exception as e:
                print(f"Error loading coordinate maps: {e}")
                return
            
            if markl == 1:
                try:
                    U, V, W = markl_correction(I_all, X_corr, Y_corr, Z_corr, np.array(venc))
                    U_nf, V_nf, W_nf = markl_correction(I_all_nf, X_corr, Y_corr, Z_corr, np.array(venc))
                    print("Markl correction applied.")
                except Exception as e:
                    print(f"Error during Markl correction: {e}")
                    return
            else:
                try:
                    if I_all.shape[3] < 3:
                        raise IndexError(f"Expected at least 3 flow components, got {I_all.shape[3]}.")
                    U = I_all[:, :, :, 0] * np.array(venc)[0] / 4095.0
                    V = I_all[:, :, :, 1] * np.array(venc)[1] / 4095.0
                    W = I_all[:, :, :, 2] * np.array(venc)[2] / 4095.0
                    U_nf = I_all_nf[:, :, :, 0] * np.array(venc)[0] / 4095.0
                    V_nf = I_all_nf[:, :, :, 1] * np.array(venc)[1] / 4095.0
                    W_nf = I_all_nf[:, :, :, 2] * np.array(venc)[2] / 4095.0
                    print("Flow data scaled without Markl correction.")
                except Exception as e:
                    print(f"Error during scaling: {e}")
                    return
            U = U - U_nf
            V = V - V_nf
            W = W - W_nf
            print("No-flow data subtracted.")
            try:
                mask_nii = nib.load(mask_file)
                affine = mask_nii.affine
                os.makedirs('result', exist_ok=True)
                nib.save(nib.Nifti1Image(X_corr, affine), os.path.join('result', 'X_phase1.nii'))
                nib.save(nib.Nifti1Image(Y_corr, affine), os.path.join('result', 'Y_phase1.nii'))
                nib.save(nib.Nifti1Image(Z_corr, affine), os.path.join('result', 'Z_phase1.nii'))
                print("Coordinate NIfTI files saved.")
                nib.save(nib.Nifti1Image(U, affine), os.path.join('result', 'U_phase1.nii'))
                nib.save(nib.Nifti1Image(V, affine), os.path.join('result', 'V_phase1.nii'))
                nib.save(nib.Nifti1Image(W, affine), os.path.join('result', 'W_phase1.nii'))
                print("Velocity component NIfTI files saved.")
                nib.save(nib.Nifti1Image(bith, affine), os.path.join('result', 'Mask_phase1.nii'))
                print("Mask NIfTI file saved.")
            except Exception as e:
                print(f"Error saving NIfTI files: {e}")
                return
            
            # Write Tecplot Binary (.plt) file as well.
            try:
                writeTecplotBin(os.path.join('result', 'velocity_data.plt'),
                                'MRI test section',
                                'Phase 1 Zone',
                                X=X_corr, Y=Y_corr, Z=Z_corr,
                                U=U, V=V, W=W)
            except Exception as e:
                print(f"Error writing Tecplot file: {e}")
            print(f"Phase {ph}: Raw DICOM processing completed successfully. NIfTI and Tecplot files are saved in 'result'.")
    
if __name__ == "__main__":
    velocity_unwarping_main()