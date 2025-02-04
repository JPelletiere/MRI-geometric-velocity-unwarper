"""
===========================================================================
Script: calc_pixel_loc.py
Description:
    This script computes the physical (patient) coordinates of each pixel in a
    series of DICOM images representing an MRI phantom. It uses metadata from 
    the DICOM files—such as PixelSpacing, SliceThickness, ImagePositionPatient,
    and ImageOrientationPatient—to calculate the x, y, and z coordinates for 
    every pixel in each slice. The computed coordinate arrays (in meters) along 
    with the original image pixel data are then saved as NIfTI files.
    
Input:
    - A directory containing DICOM (.dcm) files.

Output:
    - NIfTI files saved in the specified output directory:
        - xvals.nii: 3D array of x-coordinate values (in meters)
        - yvals.nii: 3D array of y-coordinate values (in meters)
        - zvals.nii: 3D array of z-coordinate values (in meters)
        - HHimg.nii: 3D array of image pixel intensities
        
Usage:
    Configure the input and output directories in the __main__ block and run the script.
===========================================================================
"""

import os
import numpy as np
import pydicom
import nibabel as nib
import pandas as pd

def calc_pixel_loc(input_directory, output_directory):
    """
    Compute the spatial coordinates of each pixel in a series of DICOM images.
    
    This function reads DICOM files from the specified input directory, computes the 
    physical (patient) coordinates (x, y, z in meters) for each pixel based on DICOM 
    metadata (PixelSpacing, SliceThickness, ImagePositionPatient, ImageOrientationPatient), 
    and saves the computed coordinate arrays along with the original image data as NIfTI files.
    
    Parameters:
        input_directory (str): Path to the directory containing DICOM (.dcm) files.
        output_directory (str): Path to the directory where NIfTI files will be saved.
        
    Returns:
        Saves four NIfTI files (xvals.nii, yvals.nii, zvals.nii, HHimg.nii) in the output directory.
    """
    
    # Verify that the input directory exists
    if not os.path.isdir(input_directory):
        print(f"Error: The input directory '{input_directory}' does not exist.")
        return

    # Gather all files with the '.dcm' extension (case-insensitive)
    dicom_filenames = [filename for filename in os.listdir(input_directory) if filename.lower().endswith('.dcm')]
    if len(dicom_filenames) == 0:
        print('Error: No DICOM .dcm files found in the specified input directory.')
        return

    # Sort the filenames alphabetically (assumes order corresponds to slice order)
    dicom_filenames.sort()

    # Create full file paths for each DICOM file
    dicom_file_paths = [os.path.join(input_directory, filename) for filename in dicom_filenames]

    # Read the first DICOM file to extract essential metadata for all slices
    try:
        first_dicom = pydicom.dcmread(dicom_file_paths[0])
    except Exception as error:
        print(f"Error: Could not read the first DICOM file '{dicom_filenames[0]}'. Error: {error}")
        return

    # Extract pixel spacing and slice thickness from the first DICOM file
    try:
        pixel_spacing_xy = first_dicom.PixelSpacing  # [PixelSpacing_x, PixelSpacing_y] in mm
        slice_thickness = first_dicom.SliceThickness    # in mm
        # Combine into a list: [pixel_spacing_x, pixel_spacing_y, slice_thickness]
        pixel_spacing = [float(pixel_spacing_xy[0]), float(pixel_spacing_xy[1]), float(slice_thickness)]
    except AttributeError as error:
        print(f"Error: Missing necessary DICOM metadata in '{dicom_filenames[0]}'. Error: {error}")
        return

    # Retrieve image dimensions (rows and columns) and determine number of slices
    num_rows = int(first_dicom.Rows)       # Number of pixels in the vertical direction
    num_cols = int(first_dicom.Columns)    # Number of pixels in the horizontal direction
    num_slices = len(dicom_file_paths)       # Total number of slices

    # Initialize arrays to hold computed coordinates and image data
    x_coords = np.zeros((num_rows, num_cols, num_slices))
    y_coords = np.zeros((num_rows, num_cols, num_slices))
    z_coords = np.zeros((num_rows, num_cols, num_slices))
    image_data = np.zeros((num_rows, num_cols, num_slices), dtype=np.float32)

    # Create a meshgrid of pixel indices (zero-indexed)
    pixel_indices_x = np.arange(num_cols)  # Column indices
    pixel_indices_y = np.arange(num_rows)  # Row indices
    meshgrid_x, meshgrid_y = np.meshgrid(pixel_indices_x, pixel_indices_y)

    # Process each DICOM file (each corresponding to one image slice)
    for slice_index, dicom_file_path in enumerate(dicom_file_paths):
        try:
            dicom_info = pydicom.dcmread(dicom_file_path)
        except Exception as error:
            print(f"Warning: Could not read '{dicom_file_path}'. Skipping. Error: {error}")
            continue

        # Extract the image position (origin) and orientation from the DICOM metadata
        try:
            image_position = [float(coord) for coord in dicom_info.ImagePositionPatient]  # [x, y, z] in mm
            image_orientation = [float(val) for val in dicom_info.ImageOrientationPatient]  # [cos1, cos2, cos3, cos4, cos5, cos6]
        except AttributeError as error:
            print(f"Warning: Missing metadata in '{dicom_file_path}'. Skipping. Error: {error}")
            continue

        # Construct the transformation (rotation + translation) matrix.
        # This matrix maps pixel indices to patient coordinates (in mm).
        rotation_matrix = np.array([
            [image_orientation[0] * pixel_spacing[0], image_orientation[3] * pixel_spacing[1], 0, image_position[0]],
            [image_orientation[1] * pixel_spacing[0], image_orientation[4] * pixel_spacing[1], 0, image_position[1]],
            [image_orientation[2] * pixel_spacing[0], image_orientation[5] * pixel_spacing[1], 0, image_position[2]],
            [0, 0, 0, 1]
        ])

        # Prepare homogeneous coordinates for each pixel in the slice.
        # This involves flattening the meshgrid and adding rows for z (initialized to 0) and the homogeneous coordinate (1).
        homogeneous_pixel_coords = np.vstack((
            meshgrid_x.flatten(),              # x pixel indices
            meshgrid_y.flatten(),              # y pixel indices
            np.zeros(meshgrid_x.size),         # z coordinate (0 for the current slice)
            np.ones(meshgrid_x.size)           # homogeneous coordinate
        ))

        # Apply the transformation matrix to obtain patient coordinates for each pixel.
        patient_coords = rotation_matrix @ homogeneous_pixel_coords  # Shape: (4, total_pixels)

        # Reshape the transformed coordinates back to the 2D image shape
        x_slice = patient_coords[0, :].reshape((num_rows, num_cols))
        y_slice = patient_coords[1, :].reshape((num_rows, num_cols))
        z_slice = patient_coords[2, :].reshape((num_rows, num_cols))

        # Convert from millimeters to meters and store in the corresponding 3D arrays
        x_coords[:, :, slice_index] = x_slice / 1000.0
        y_coords[:, :, slice_index] = y_slice / 1000.0
        z_coords[:, :, slice_index] = z_slice / 1000.0

        # Read the image pixel data from the DICOM file and store it
        image_slice = dicom_info.pixel_array.astype(np.float32)
        image_data[:, :, slice_index] = image_slice

    # Ensure that the output directory exists (create if necessary)
    os.makedirs(output_directory, exist_ok=True)

    # Create NIfTI images for the coordinate arrays and image data.
    # An identity affine matrix is used because the arrays already represent physical space.
    nifti_x = nib.Nifti1Image(x_coords, affine=np.eye(4))
    nifti_y = nib.Nifti1Image(y_coords, affine=np.eye(4))
    nifti_z = nib.Nifti1Image(z_coords, affine=np.eye(4))
    nifti_image = nib.Nifti1Image(image_data, affine=np.eye(4))

    # Save the NIfTI files to the output directory.
    nib.save(nifti_x, os.path.join(output_directory, 'xvals.nii'))
    nib.save(nifti_y, os.path.join(output_directory, 'yvals.nii'))
    nib.save(nifti_z, os.path.join(output_directory, 'zvals.nii'))
    nib.save(nifti_image, os.path.join(output_directory, 'HHimg.nii'))

    print('\nCoordinate value files have been successfully saved to:', output_directory)

def calc_pixel_loc_main():
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
        #if excel_data.shape[0] < 7:
            #raise ValueError("Excel file does not contain enough rows for all parameters.")
        input_directory = str(excel_data[1, 1]).strip()
        output_directory = str(excel_data[2, 1]).strip()
        # Optionally, normalize the paths (this converts forward/backward slashes appropriately)
        input_directory = os.path.normpath(input_directory)
        output_directory = os.path.normpath(output_directory)
        print("Input Directory:", input_directory)
        print("Output Directory:", output_directory)
    except Exception as e:
        print(f"Error extracting configuration parameters: {e}")
        return
    
    # Validate that the input directory exists before proceeding.
    if not os.path.isdir(input_directory):
        print(f"Error: The input directory '{input_directory}' does not exist.")
    else:
        calc_pixel_loc(input_directory, output_directory)


if __name__ == "__main__":
    calc_pixel_loc_main()