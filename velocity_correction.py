import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import pydicom
import nibabel as nib
from dataclasses import dataclass
from typing import List, Tuple

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
    """
    N = [item.name for item in S]
    print(f"Total DICOM files available for reading: {len(N)}")

    if ini - 1 + noofimages > len(N):
        raise IndexError(f"Requested image index {ini - 1 + noofimages - 1} exceeds available DICOM files ({len(N)}).")

    # Read the initial DICOM file
    initial_filename = os.path.join(folder, N[ini - 1])
    info = pydicom.dcmread(initial_filename)

    nrows = info.Rows
    ncols = info.Columns

    I = np.zeros((nrows, ncols, noofimages), dtype=np.uint16)
    X = np.zeros((nrows, ncols, noofimages), dtype=np.float64)
    Y = np.zeros((nrows, ncols, noofimages), dtype=np.float64)
    Z = np.zeros((nrows, ncols, noofimages), dtype=np.float64)

    for i in range(1, noofimages + 1):
        idx = ini + i - 2  # Adjust for 1-based indexing
        if idx >= len(N):
            raise IndexError(f"Requested image index {idx + 1} exceeds available DICOM files.")
        n = N[idx]
        current_filename = os.path.join(folder, n)
        info = pydicom.dcmread(current_filename)

        ps = info.PixelSpacing  # [PixelSpacing_x, PixelSpacing_y]
        ipp = info.ImagePositionPatient  # [x, y, z]
        orientation_cosines = info.ImageOrientationPatient  # [cos1, cos2, cos3, cos4, cos5, cos6]

        row_cosines = np.array(orientation_cosines[:3])
        col_cosines = np.array(orientation_cosines[3:])
        # (A normal vector could be computed if needed)
        # Create a meshgrid for pixel indices
        x_indices = np.arange(ncols)
        y_indices = np.arange(nrows)
        X_grid, Y_grid = np.meshgrid(x_indices, y_indices)

        # Calculate spatial coordinates (a simplified version)
        X_coord = ipp[0] + (X_grid * ps[0] * row_cosines[0]) + (Y_grid * ps[1] * col_cosines[0])
        Y_coord = ipp[1] + (X_grid * ps[0] * row_cosines[1]) + (Y_grid * ps[1] * col_cosines[1])
        Z_coord = ipp[2] + (X_grid * ps[0] * row_cosines[2]) + (Y_grid * ps[1] * col_cosines[2])

        X[:, :, i - 1] = X_coord
        Y[:, :, i - 1] = Y_coord
        Z[:, :, i - 1] = Z_coord

        I[:, :, i - 1] = info.pixel_array.astype(np.uint16)

    return I, X, Y, Z

def markl_correction(I_all: np.ndarray, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, v_enc: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies Markl correction to the velocity fields.
    """
    gradient_files = {
        'gxx': 'dbxdx.mat',
        'gxy': 'dbydx.mat',
        'gxz': 'dbzdx.mat',
        'gyx': 'dbxdy.mat',
        'gyy': 'dbydy.mat',
        'gyz': 'dbzdy.mat',
        'gzx': 'dbxdz.mat',
        'gzy': 'dbydz.mat',
        'gzz': 'dbzdz.mat'
    }

    gradients = {}
    for key, filename in gradient_files.items():
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Gradient file '{filename}' not found.")
        mat_data = scipy.io.loadmat(filename)
        if key not in mat_data:
            raise KeyError(f"Variable '{key}' not found in '{filename}'.")
        gradients[key] = mat_data[key]

    if not all(gradients[key].shape == X.shape for key in gradient_files.keys()):
        raise ValueError("All gradient arrays must have the same shape as the coordinate arrays (X, Y, Z).")

    gma = np.diag([4095.0 / v_enc[2], 4095.0 / v_enc[1], 4095.0 / v_enc[0]])
    U = np.zeros_like(I_all[:, :, :, 0], dtype=np.float64)
    V = np.zeros_like(I_all[:, :, :, 0], dtype=np.float64)
    W = np.zeros_like(I_all[:, :, :, 0], dtype=np.float64)

    slices, cols, rows, phases = I_all.shape
    for im in range(slices):
        for jm in range(cols):
            for km in range(rows):
                ap = np.array([
                    [gradients['gxx'][im, jm, km], gradients['gxy'][im, jm, km], gradients['gxz'][im, jm, km]],
                    [gradients['gyx'][im, jm, km], gradients['gyy'][im, jm, km], gradients['gyz'][im, jm, km]],
                    [gradients['gzx'][im, jm, km], gradients['gzy'][im, jm, km], gradients['gzz'][im, jm, km]]
                ])
                cond_number = np.linalg.cond(ap @ gma)
                if cond_number < 1 / np.finfo(ap.dtype).eps:
                    try:
                        tempk = np.linalg.inv(ap @ gma).T @ np.array([
                            I_all[im, jm, km, 2],
                            I_all[im, jm, km, 1],
                            I_all[im, jm, km, 0]
                        ])
                        U[im, jm, km] = tempk[0]
                        V[im, jm, km] = tempk[1]
                        W[im, jm, km] = tempk[2]
                    except np.linalg.LinAlgError:
                        U[im, jm, km] = 0.0
                        V[im, jm, km] = 0.0
                        W[im, jm, km] = 0.0
                else:
                    U[im, jm, km] = 0.0
                    V[im, jm, km] = 0.0
                    W[im, jm, km] = 0.0

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
    for data in var_data:
        if data.shape != reference_shape:
            raise ValueError("All variable arrays must have the same shape.")

    if len(reference_shape) == 3:
        slices, cols, rows = reference_shape
    elif len(reference_shape) == 4:
        slices, cols, rows, phases = reference_shape
    else:
        raise ValueError("Variable arrays must be 3D or 4D.")

    with open(output_filename, 'wb') as fid:
        fid.write(b'#!TDV112')
        fid.write(struct.pack('<i', 1))
        fid.write(struct.pack('<i', 0))
        for char in file_title:
            fid.write(struct.pack('<i', ord(char)))
        fid.write(struct.pack('<i', 0))
        fid.write(struct.pack('<i', num_vars))
        for name in var_names:
            for char in name:
                fid.write(struct.pack('<i', ord(char)))
            fid.write(struct.pack('<i', 0))
        fid.write(struct.pack('<i', 1))
        fid.write(struct.pack('<i', 0))
        fid.write(struct.pack('<i', slices))
        fid.write(struct.pack('<i', cols))
        fid.write(struct.pack('<i', rows))
        fid.write(struct.pack('<i', 0))
        fid.write(struct.pack('<i', 0))
        fid.write(struct.pack('<i', 0))
        fid.write(struct.pack('<i', 1))
        fid.write(struct.pack('<i', 0))
        fid.write(struct.pack('<i', 1))
        for data in var_data:
            data_flat = data.flatten(order='F')
            fid.write(data_flat.astype(np.float32).tobytes())
        fid.write(struct.pack('<f', 357.0))
    print(f"Tecplot Binary file '{output_filename}' has been created.")

# ------------------------ Main Processing Function ------------------------ #

def velocity_unwarping_main():
    # ------------------------------------------------------------------
    # Set this flag to True to use geometrically corrected (NIfTI) files;
    # if False, the code will use raw DICOM files via dread.
    USE_CORRECTED = True
    # ------------------------------------------------------------------

    excel_path = 'Input_parametersheet_MRV.xlsx'
    if not os.path.exists(excel_path):
        print(f"Error: Excel configuration file '{excel_path}' not found.")
        return

    try:
        excel_data = pd.read_excel(excel_path, header=None).values
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    print(excel_data.shape)

    try:
        phase = int(excel_data[9, 1])
        fname = [str(f).strip().split('.')[0] for f in excel_data[10, 1:] if not pd.isna(f) and str(f).strip() != '']
        fnamenf = [str(f).strip().split('.')[0] for f in excel_data[11, 1:] if not pd.isna(f) and str(f).strip() != '']
        magsname = int(excel_data[12, 1])
        noofslices = int(excel_data[13, 1])
        venc = [float(v) for v in excel_data[14, 1:] if not pd.isna(v)]
        markl = int(excel_data[15, 1])
        if len(fname) != len(venc) or len(fnamenf) != len(venc):
            raise ValueError("Mismatch in the number of 'fname', 'fnamenf', or 'venc' entries.")
    except Exception as e:
        print(f"Error extracting configuration parameters: {e}")
        return

    print("Configuration Parameters Extracted:")
    print(f"Phase: {phase}")
    print(f"Flow On IDs (fname): {fname}")
    print(f"Flow Off IDs (fnamenf): {fnamenf}")
    print(f"Magsum ID (magsname): {magsname}")
    print(f"Number of Slices (noofslices): {noofslices}")
    print(f"Velocity Encoding (venc): {venc}")
    print(f"Markl Correction Flag (markl): {markl}")

    # ------------------------------------------------------------------
    # Depending on the USE_CORRECTED flag, we branch here.
    if USE_CORRECTED:
        print("Using corrected NIfTI files from the test dataset folder.")
        base_folder = os.path.join('.', 'test dataset')
        
        # ------------------- Load Corrected Mask ------------------- #
        mask_file = os.path.join(base_folder, "mask_corrt.nii")
        try:
            Imags = nib.load(mask_file).get_fdata().astype(np.float64)
            print(f"Loaded corrected mask image from '{mask_file}' with shape {Imags.shape}.")
        except Exception as e:
            print(f"Error loading corrected mask image: {e}")
            return

        # ------------------- Load Corrected Flow Off Images ------------------- #
        # Flow off folders: 93_flow_pc3d_P_ND, 104_flow_pc3d_P_ND, 97_flow_pc3d_P_ND
        # Expect corrected files: "93_corrt.nii", "104_corrt.nii", "97_corrt.nii"
        flow_off_ids = ["93", "104", "97"]
        I_all_nf_list = []
        for fid in flow_off_ids:
            file_path = os.path.join(base_folder, f"{fid}_corrt.nii")
            try:
                img = nib.load(file_path).get_fdata().astype(np.float64)
                if img.shape != Imags.shape:
                    print(f"Flow off image {fid}: shape {img.shape} does not match mask shape {Imags.shape}, transposing...")
                    img = np.transpose(img, (1, 0, 2))
                I_all_nf_list.append(img)
                print(f"Loaded corrected flow off image from '{file_path}' with shape {img.shape}.")
            except Exception as e:
                print(f"Error loading corrected flow off image '{file_path}': {e}")
                return
        I_all_nf = np.stack(I_all_nf_list, axis=-1)  # Shape: (264, 384, 64, 3)

        # ------------------- Load Corrected Flow On Images ------------------- #
        # Flow on folders: 126_flow_pc3d_P_ND, 130_flow_pc3d_P_ND, 136_flow_pc3d_P_ND
        # Expect corrected files: "126_corrt.nii", "130_corrt.nii", "136_corrt.nii"
        flow_on_ids = ["126", "130", "136"]
        I_all_list = []
        for fid in flow_on_ids:
            file_path = os.path.join(base_folder, f"{fid}_corrt.nii")
            try:
                img = nib.load(file_path).get_fdata().astype(np.float64)
                if img.shape != Imags.shape:
                    print(f"Flow on image {fid}: shape {img.shape} does not match mask shape {Imags.shape}, transposing...")
                    img = np.transpose(img, (1, 0, 2))
                I_all_list.append(img)
                print(f"Loaded corrected flow on image from '{file_path}' with shape {img.shape}.")
            except Exception as e:
                print(f"Error loading corrected flow on image '{file_path}': {e}")
                return
        I_all = np.stack(I_all_list, axis=-1)  # Shape: (264, 384, 64, 3)

        # ------------------- Load Corrected Coordinate Maps ------------------- #
        # Expect coordinate files named: xvals.nii, yvals.nii, zvals.nii
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

        # Based on your output, the mask shape is (264, 384, 64) and the original coordinate shapes are (384, 264, 64).
        # To match the mask, swap the first two axes:
        X_corr_final = np.transpose(X_corr, (1, 0, 2))
        Y_corr_final = np.transpose(Y_corr, (1, 0, 2))
        Z_corr_final = np.transpose(Z_corr, (1, 0, 2))
        print("Permuted coordinate shapes:", X_corr_final.shape, Y_corr_final.shape, Z_corr_final.shape)

        # ------------------- Process the Data ------------------- #
        # (Markl flag is 0, so we scale directly.)
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

        # Subtract no-flow data from flow on data
        U = U - U_nf
        V = V - V_nf
        W = W - W_nf
        print("No-flow data subtracted.")

        # ------------------- Create the Mask ------------------- #
        # Use the corrected mask (Imags) already loaded.
        slices, cols, rows = Imags.shape
        bith = np.zeros((slices, cols, rows), dtype=np.float64)
        for j in range(slices):
            bith[j, :, :] = mask_simple(Imags[j, :, :], j, 200).astype(float)

        # ------------------- Write Output Files as NIfTI Files ------------------- #
        try:
            # Get the affine from the mask NIfTI file
            mask_nii = nib.load(mask_file)
            affine = mask_nii.affine
            os.makedirs('result', exist_ok=True)
            
            # Save coordinate maps using the permuted (final) coordinate arrays
            nib.save(nib.Nifti1Image(X_corr_final, affine), os.path.join('result', 'X_phase1.nii'))
            nib.save(nib.Nifti1Image(Y_corr_final, affine), os.path.join('result', 'Y_phase1.nii'))
            nib.save(nib.Nifti1Image(Z_corr_final, affine), os.path.join('result', 'Z_phase1.nii'))
            print("Coordinate NIfTI files saved.")

            # Save velocity components U, V, W
            nib.save(nib.Nifti1Image(U, affine), os.path.join('result', 'U_phase1.nii'))
            nib.save(nib.Nifti1Image(V, affine), os.path.join('result', 'V_phase1.nii'))
            nib.save(nib.Nifti1Image(W, affine), os.path.join('result', 'W_phase1.nii'))
            print("Velocity component NIfTI files saved.")

            # Save the binary mask
            nib.save(nib.Nifti1Image(bith, affine), os.path.join('result', 'Mask_phase1.nii'))
            print("Mask NIfTI file saved.")

        except Exception as e:
            print(f"Error saving NIfTI files: {e}")
            return

        print("Corrected mode processing completed successfully and NIfTI files are saved in the 'result' folder.")


    else:
        # ------------------ RAW DICOM Mode (using dread) ------------------
        print("Using raw DICOM files with dread.")
        
        # Loop over each phase (ph)
        for ph in range(1, phase + 1):
            print(f"\nProcessing Phase {ph}/{phase} (RAW DICOM)...")
            # --- Read the magsum image for mask creation ---
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
            if available_images < noofslices:
                print(f"Warning: Available DICOM files ({available_images}) less than 'noofslices' ({noofslices}). Adjusting 'noofimages' to {available_images}.")
                noofimages = available_images
            else:
                noofimages = noofslices
            try:
                Imags, X, Y, Z = dread(sfile_dicom, smags1 + os.sep, 1, noofimages)
                print("Magsum images loaded successfully using dread.")
            except Exception as e:
                print(f"Error reading magsum images: {e}")
                return

            # Here we assume the coordinate arrays from dread are in the correct orientation.
            X_new = X
            Y_new = Y
            Z_new = Z

            # --- Create the binary mask from the magsum images ---
            slices, cols, rows = Imags.shape
            bith = np.zeros((slices, cols, rows), dtype=np.float64)
            for j in range(slices):
                bith[j, :, :] = mask_simple(Imags[j, :, :], j, 200).astype(float)
            print("Binary mask created.")

            # --- Read the phase (flow) images ---
            num_components = len(fname)  # number of flow on components
            I_all = np.zeros((slices, cols, rows, num_components), dtype=np.int16)
            I_all_nf = np.zeros((slices, cols, rows, num_components), dtype=np.int16)
            for i in range(num_components):
                # --- Flow On ---
                flow_on_pattern = os.path.join('test dataset', f"{fname[i]}_*")
                ffolder_on_dirs = glob.glob(flow_on_pattern)
                if not ffolder_on_dirs:
                    print(f"Warning: No directories found matching pattern '{flow_on_pattern}'. Skipping component {i+1}.")
                    continue
                ffolder_on = ffolder_on_dirs[0]
                print(f"Found flow on directory: '{ffolder_on}'")
                ffile_on_pattern = os.path.join(ffolder_on, '**', '*')
                ffile_on_paths = glob.glob(ffile_on_pattern, recursive=True)
                ffile_on_paths = [f for f in ffile_on_paths if os.path.isfile(f)]
                if not ffile_on_paths:
                    print(f"Warning: No files found in directory '{ffolder_on}'. Skipping component {i+1}.")
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

                # --- Flow Off ---
                flow_off_pattern = os.path.join('test dataset', f"{fnamenf[i]}_*")
                ffolder_off_dirs = glob.glob(flow_off_pattern)
                if not ffolder_off_dirs:
                    print(f"Warning: No directories found matching pattern '{flow_off_pattern}'. Skipping component {i+1}.")
                    continue
                ffolder_off = ffolder_off_dirs[0]
                print(f"Found flow off directory: '{ffolder_off}'")
                ffile_off_pattern = os.path.join(ffolder_off, '**', '*')
                ffile_off_paths = glob.glob(ffile_off_pattern, recursive=True)
                ffile_off_paths = [f for f in ffile_off_paths if os.path.isfile(f)]
                if not ffile_off_paths:
                    print(f"Warning: No files found in directory '{ffolder_off}'. Skipping component {i+1}.")
                    continue
                ffile_off_dicom = [DicomFile(name=os.path.basename(f), folder=os.path.dirname(f)) for f in ffile_off_paths]
                try:
                    emni_off, _, _, _ = dread(ffile_off_dicom, ffolder_off + os.sep, 1, noofimages)
                    I_all_nf[:, :, :, i] = (emni_off.astype(np.int16) * 2) - 4095
                    print(f"Flow off images loaded for component {i+1}.")
                except Exception as e:
                    print(f"Error reading flow off images for component {i+1}: {e}")
                    continue

            # --- Load coordinate maps from .mat files (if available) ---
            try:
                x_mat = scipy.io.loadmat('xvals.mat')['xvals']
                y_mat = scipy.io.loadmat('yvals.mat')['yvals']
                z_mat = scipy.io.loadmat('zvals.mat')['zvals']
                # Mimic MATLAB's permute(X, [3,2,1])
                X_corr = np.transpose(x_mat, (2, 1, 0))
                Y_corr = np.transpose(y_mat, (2, 1, 0))
                Z_corr = np.transpose(z_mat, (2, 1, 0))
                print("Coordinate maps loaded from .mat files.")
            except Exception as e:
                print(f"Error loading coordinate maps: {e}")
                return

            # --- Process flow data ---
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
                        raise IndexError(f"Expected at least 3 flow components, but got {I_all.shape[3]}.")
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

            # Subtract no-flow data from flow data
            U = U - U_nf
            V = V - V_nf
            W = W - W_nf
            print("No-flow data subtracted.")

                        # ------------------- Write Output Files as NIfTI Files ------------------- #
            # Instead of saving a MATLAB .mat file, we now save each variable as a .nii file.
            # Use the affine from the corrected mask file for all outputs.
            try:
                # Get the affine from the mask NIfTI file
                mask_nii = nib.load(mask_file)
                affine = mask_nii.affine
                os.makedirs('result', exist_ok=True)
                
                # Save coordinate maps using the permuted (final) coordinate arrays
                nib.save(nib.Nifti1Image(X_corr_final, affine), os.path.join('result', 'X_phase1.nii'))
                nib.save(nib.Nifti1Image(Y_corr_final, affine), os.path.join('result', 'Y_phase1.nii'))
                nib.save(nib.Nifti1Image(Z_corr_final, affine), os.path.join('result', 'Z_phase1.nii'))
                print("Coordinate NIfTI files saved.")

                # Save velocity components U, V, W
                nib.save(nib.Nifti1Image(U, affine), os.path.join('result', 'U_phase1.nii'))
                nib.save(nib.Nifti1Image(V, affine), os.path.join('result', 'V_phase1.nii'))
                nib.save(nib.Nifti1Image(W, affine), os.path.join('result', 'W_phase1.nii'))
                print("Velocity component NIfTI files saved.")

                # Save the binary mask
                nib.save(nib.Nifti1Image(bith, affine), os.path.join('result', 'Mask_phase1.nii'))
                print("Mask NIfTI file saved.")

            except Exception as e:
                print(f"Error saving NIfTI files: {e}")
                return

            print("Corrected mode processing completed successfully and NIfTI files are saved in the 'result' folder.")


    

if __name__ == "__main__":
    velocity_unwarping_main()
