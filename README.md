# MRI Unwarping and Processing Suite

This project is a collection of Python scripts designed to process and correct MRI images by unwarping them. The codebase includes modules for calculating magnetic field (B-field) distortions, computing pixel locations from DICOM images, performing geometric unwarping, applying velocity corrections, and quickly viewing NIfTI images for verification.

---

## Installation and Setup

### Python 3.7 Installation

This codebase is developed for **Python 3.7**. To download Python 3.7, follow these steps:
1. Visit the official Python downloads page and select Python 3.7 for your operating system.
2. **Important:** If you do not want this installation to interfere with your existing Python setup, do not add Python 3.7 to your system PATH. Instead, use the full path to call the interpreter or a different method. For example: &"C:\Users\*your user*\AppData\Local\Programs\Python\Python37\python.exe" main_code_runner.py.
Another easy way to do this is to create a virtual environment that uses python 3.7.

### Required pip Packages

Before running the project, install the following packages (using Python 3.7’s pip):

- **numpy**
- **scipy**
- **pydicom**
- **nibabel**
- **pandas**
- **matplotlib**

You can install all dependencies with: "C:\Users\*your user*\AppData\Local\Programs\Python\Python37\python.exe" -m pip install numpy scipy pydicom nibabel pandas matplotlib

## File Descriptions

### main_code_runner.py
This is the central driver script. It calls the various processing modules in the intended order. You can select which files you want to run by changing run_(script) to true or false within the code. What each function does is described in more detail below. This should be ran in python 3.7.
- bfields.py: (Optional) Calculate B-fields from gradient coefficients. **Important** run this before the rest of the code is needed for on the spot analysis, it is very computationally heavy and can take over an hour. 
- calc_pixel_loc.py: Compute the physical (patient) coordinates of each pixel from DICOM files.
- geometric_unwarp.py: Perform geometric unwarping using methods derived from the gradunwarp research code.
- velocity_correction.py: Apply velocity unwarping corrections based on phase image differences and gradient fields.

### bfields.py
**Important**, if you are planning to use this code for on the spot data analysis during testing, make sure to run this section of the code before. It is very computationally heavy and can take over an hour. These calculated files will always be the same for a specific scanner so this only needs to be ran once then the .npy files can be repeatedly reused.

#### Overview
This is designed for computing the Legendre coefficients used in gradient coil design and for generating a displacement (B-field) table over a 3D grid. It computes a non-dimensional displacement field based on gradient coil coefficients and scales this field using a reference length (`R0`) and gradient reference (`Gref`). This field is later used in unwarping procedures. After generating the displacement field, the script calculates its spatial gradients using numerical differentiation (`np.gradient`), which are then saved for further processing in the MRI unwarping pipeline. The script comes pre-configured with constants tailored for Siemens scanners. For example, it sets field-of-view limits (`siemens_fovmin` and `siemens_fovmax`), along with different `R0` and `Gref` values depending on the Siemens model (such as sonata, avanto, allegra, and prisma1). Edit `table_name` if you want to change it to one of these. 

#### Customization
If you are using a Siemens scanner with different specifications or if you need to support another scanner type, you can modify the constants directly in the script. For instance, update the FOV limits, `R0`, and `Gref` values in the section of the code that checks the `table_name` (e.g., within the `create_displacement_table_for_siemens_coords` function). This allows you to adapt the displacement and gradient calculations to match your scanner’s hardware. To extend support to scanners other than Siemens, you would need to add new conditional branches in the scanner-specific parameter section. This involves specifying the correct FOV, reference length, and gradient reference values for your scanner, ensuring the displacement field is computed correctly for your system. The script is set up to parse a coefficient file with a specific format (using regular expressions to extract values). If your coefficient file format differs, you may need to adjust the parsing logic accordingly. It reads a coefficient file (typically named `coeff.grad`) that contains the gradient coil coefficients. The file is parsed to extract the values which populate the coefficient matrices (e.g., `Alpha_x`, `Alpha_z`, and `Beta_y`). These coefficients are critical for the accurate calculation of the displacement field.

#### Outputs
When you run **bfields.py**, the script computes both the displacement fields (B-fields) and their spatial gradients over a 3D grid. The outputs are saved as NumPy binary files (`.npy`) and include:

1. **Coordinate Grids:**
   - **X.npy, Y.npy, Z.npy**  
     These files store the 3D grid coordinates that span the defined field-of-view (typically from -0.18 to 0.18 meters). They represent the spatial locations corresponding to each point in the computed displacement field.

2. **Displacement (B-field) Arrays:**
   - **Bx.npy, By.npy, Bz.npy**  
     These files contain the computed displacement field components for the x, y, and z directions, respectively. The displacement fields are calculated using the gradient coil coefficients and scaled by the reference length (`R0`) and gradient reference (`Gref`).

3. **Gradient Arrays:**
   - **dbxdx.npy, dbxdy.npy, dbxdz.npy**  
     These files store the partial derivatives of the Bx field with respect to the x, y, and z axes.
   - **dbydx.npy, dbydy.npy, dbydz.npy**  
     These files contain the corresponding partial derivatives of the By field.
   - **dbzdx.npy, dbzdy.npy, dbzdz.npy**  
     These files represent the spatial gradients of the Bz field.



### calc_pixel_loc.py
You should only need to run this once per scanning session, in the parameter sheet input the path to any one of your flow folders that contains the DICOM images for your scan. You can also input a path to where you want the output files to be put.

#### Overview
Reads DICOM images, extracts metadata (PixelSpacing, SliceThickness, ImagePositionPatient, and ImageOrientationPatient), and computes the x, y, and z coordinates for each pixel. The resulting coordinate maps (and the original image data) are saved as NIfTI files. It processes a series of DICOM images to compute the physical (patient) coordinates for each pixel. It extracts critical metadata from the DICOM headers—such as PixelSpacing, SliceThickness, ImagePositionPatient, and ImageOrientationPatient—to map each pixel to its correct spatial location. In addition to generating coordinate maps, the script also preserves the original image data by saving it as a NIfTI file.

#### Customization
You can modify the path to the directory containing your DICOM files. The script looks for files with the `.dcm` extension. The destination folder where the NIfTI files (for x, y, z coordinates and image data) will be saved can be adjusted. The script assumes that all DICOM files in the directory contain the necessary metadata (such as PixelSpacing, SliceThickness, and ImageOrientationPatient). If your data uses different tags or requires special handling, you might need to modify the code to extract the correct values. By default, the files are sorted alphabetically, which should correspond to the correct slice order. If your DICOM files are named or sorted differently, you may need to adjust the sorting logic.

#### Outputs
The script produces the following NIfTI files as outputs:

1. **xvals.nii**  
   - Contains a 3D array of the x-coordinate values (in meters) for every pixel in the DICOM series.

2. **yvals.nii**  
   - Contains a 3D array of the y-coordinate values (in meters).

3. **zvals.nii**  
   - Contains a 3D array of the z-coordinate values (in meters).

4. **HHimg.nii**  
   - Contains the original image pixel data from the DICOM files, stored as a 3D array.

These outputs provide a complete spatial mapping of the DICOM images, making them suitable for further processing, such as geometric unwarping and other corrections in the MRI processing pipeline.

### geometric_unwarp.py
To use this, put in the file name of whatever folder is containing all your testing data into the input parameter sheet. It is expected that within whatever folder you put into this, you will have all your image reading folders, each of which contains the DICOM files for the read. In the flow folders row, put in each of the read folders you want to be ran by this (the files within dataset folder). In the row below, input the labels for each of these files, the labels will be used to name the outputs like label.nii and label_corrt.nii. There is also a spot for a mask folder, leave this blank if you don't want a mask image made (this doesn't mask the other images only creates a seperate mask image). The code is configured to use the precomputed b-fields from bfields.py, if you want to change this to calculate the b-fields in the code you can change the call to eval_precomputed_bfields to eval_spherical_harmonics.

#### Overview
This script implements a comprehensive approach to correcting geometric distortions in MRI volumes using spherical harmonic techniques. It is an integrated version of the gradunwarp package, combining several components into a single workflow. The script first sets global parameters and utility functions for handling coordinate transformations, grid creation, and interpolation. It then parses gradient or coefficient files to obtain the spherical harmonic coefficients that describe the distortions introduced by the scanner hardware.

Once the coefficients are obtained, the script builds a high-resolution evaluation grid over the specified field-of-view. It evaluates the spherical harmonics on this grid to compute a displacement field that represents the geometric distortion present in the MRI data. Using this displacement field, the script then performs a slice-by-slice correction of the input volume. This involves interpolating the displacement values onto the voxel grid, applying the corrections (with an option for Jacobian adjustment), and ultimately resampling the warped image to produce an unwarped volume.

The script is designed to be flexible and configurable. It supports command-line arguments for specifying input/output files, vendor options (e.g., Siemens or GE), and additional unwarping parameters such as field-of-view limits, number of grid points, and interpolation order. Additionally, it can integrate with external configuration files (like an Excel sheet) to streamline processing for large datasets, including multiple flow series and mask images.

#### Customization
Modify default values such as `siemens_fovmin`, `siemens_fovmax`, `siemens_numpoints`, and `siemens_max_det` to suit your specific scanner hardware. For vendors other than Siemens, similar parameter blocks can be added. The script reads spherical harmonic coefficients from a `.grad` or `.coef` file. If your coefficient file has a different format, you may need to adjust the parsing functions (`get_siemens_coef`, `get_siemens_grad`) accordingly. You can adjust the evaluation grid parameters (`fovmin`, `fovmax`, `numpoints`) to control the resolution and spatial extent of the grid where the spherical harmonics are evaluated. This directly influences the precision of the displacement field computation. Change the interpolation order (via the `order` parameter) to control the smoothness of the resampling process. You also have the option to disable Jacobian correction with the `nojac` flag or to switch between warping and unwarping modes using the `warp` flag. The script leverages command-line arguments for quick configuration and can also read additional parameters from an Excel file (`Input_parametersheet_MRV.xlsx`). This allows for batch processing of multiple datasets, including separate processing for masks and flow data.

#### Outputs
-**Unwarped Volume:**  
  The main output is an unwarped MRI volume, saved as a NIfTI (or MGH) file. This volume has been corrected for the geometric distortions identified by the displacement field and is ready for further analysis or processing. In a full processing pipeline, the script produces additional outputs such as unwarped masks and corrected flow series. These files typically include a suffix (e.g., `_corrt`) to indicate that they have undergone correction.



### velocity_correction.py
Below is an explanation of the parameters shown in your input parameter sheet. These values are read by **velocity_correction.py** to configure the velocity unwarping process.

1. **phase = 1**  
   - This indicates that you have only one “phase” or time point to process. In some 4D flow MRI datasets, multiple phases (time points) are acquired over the cardiac cycle. If `phase` were larger than 1, the script would process multiple phases in a loop. Here, a value of `1` means there is only a single time point (or single “phase” of flow) to handle.

2. **fname = 126, 136, 130**  
   - These are the **flow-on image identifiers** (sometimes referred to as “flow IDs”).  
   - Each ID (e.g., `126`, `136`, `130`) typically corresponds to a particular direction of flow encoding in the MRI acquisition. For instance, one ID might represent the flow in the x-direction, another in y, and another in z.  
   - In the script, these identifiers are used to locate the DICOM (or NIfTI) files associated with flow-on scans, which contain phase information related to blood flow.

3. **fnamenf = 93, 104, 97**  
   - These are the **flow-off image identifiers** (sometimes referred to as “no-flow IDs”).  
   - Similar to `fname`, each ID corresponds to a direction of flow encoding, but in this case, they represent acquisitions where the flow encoding is “off.” These scans are often used as reference images to subtract background phase offsets or other artifacts.  
   - The script will load these images and subtract them from the corresponding flow-on images to isolate true flow-induced phase changes.

4. **magsname = 89**  
   - This indicates the identifier for the **magnitude image** (often used to create a binary mask).  
   - A typical 4D flow MRI acquisition stores not only phase information (flow-on/off) but also a magnitude image that shows the anatomical structure. The script uses this magnitude image to generate a mask (by applying an intensity threshold) that isolates relevant regions (e.g., vessels) and excludes background noise.

5. **noofslice = 64**  
   - This tells the script how many **slices** you expect in your dataset.  
   - If your acquisition consists of 64 axial slices, for example, this parameter ensures the script only attempts to read or process that many slices from the data. If the script is working in raw DICOM mode, it will loop over 64 slices. In corrected NIfTI mode, it expects the 3D volume to have 64 slices in one dimension.

6. **venc = 1.25, 1.25, 0.8**  
   - These are the **velocity encoding (v_enc) values** for each of the three flow directions.  
   - In many 4D flow MRI protocols, the velocity encoding parameter sets the maximum velocity that can be measured without aliasing (e.g., 1.25 m/s). Here, the first value (1.25) might correspond to flow in the x-direction, the second (1.25) to flow in y, and the third (0.8) to flow in z.  
   - When the script processes the phase images, it uses these v_enc values to convert the raw phase data (often stored as integers) into meaningful velocity units (e.g., meters/second).

7. **markl = 0**  
   - This is a **flag** indicating whether the **Markl correction** is applied (1) or not (0).  
   - The Markl correction typically involves a more advanced correction step that uses the gradient fields (loaded from `.npy` files) to refine the velocity data.  
   - With `markl = 0`, the script will not apply that additional correction. Instead, it will directly scale and subtract the flow-off data from the flow-on data to produce the velocity fields.

#### Overview
The **velocity_correction.py** script implements a velocity unwarping procedure to correct for flow-related distortions in MRI data. It does so by reading in MRI volumes (either from corrected NIfTI files or raw DICOM series), creating a binary mask based on a defined intensity threshold, and loading phase images for both flow-on and flow-off conditions. The script then computes corrected velocity fields using either the Markl correction (which involves interpolating precomputed gradient fields onto the coordinate maps) or a direct scaling approach if the Markl method is not enabled. Finally, the corrected velocity components along with the coordinate maps are saved as NIfTI files, and a Tecplot binary file is generated for visualization and further analysis.

#### Customization
The script can operate in two modes:
- **Corrected Mode (USE_CORRECTED = True):** Uses pre-corrected geometry files (NIfTI files with the `_corrt` suffix).
- **Raw DICOM Mode (USE_CORRECTED = False):** Loads raw DICOM images using a custom DICOM reader (`dread`).

An Excel file (`Input_parametersheet_MRV.xlsx`) is used to specify key parameters such as (further description is above):
- Flow image identifiers for both flow-on and flow-off conditions.
- The magnitude image identifier used to generate the binary mask.
- The number of slices to process.
- Velocity encoding (v_enc) values for each direction.
- A flag to indicate whether to apply the Markl correction.

The intensity threshold for creating the binary mask is set at 200 by default. This can be modified if your data requires a different threshold. The script assumes specific naming conventions and folder structures for the DICOM or NIfTI files. Adjust the file/folder patterns if your data is organized differently. If the Markl correction is enabled (via the configuration), the script uses gradient data (loaded from precomputed `.npy` files) to interpolate and apply a correction. Otherwise, it scales the phase data directly based on the provided velocity encoding values.

#### Outputs
1. **Corrected Coordinate Maps:**  
   - **X_phase1.nii, Y_phase1.nii, Z_phase1.nii:**  
     These NIfTI files contain the corrected coordinate maps (in meters) for the MRI volume. The coordinates are adjusted to match the unwarped geometry.

2. **Corrected Velocity Fields:**  
   - **U_phase1.nii, V_phase1.nii, W_phase1.nii:**  
     These files store the corrected velocity components for each spatial direction (U, V, W). The corrections are computed by subtracting the scaled flow-off data from the flow-on data (with or without applying the Markl correction).

3. **Binary Mask:**  
   - **Mask_phase1.nii:**  
     This NIfTI file contains the binary mask generated from the magnitude image. It is used to isolate the regions of interest and exclude background noise.

4. **Tecplot Binary File:**  
   - **velocity_data.plt:**  
     In addition to the NIfTI outputs, the script writes a Tecplot binary file that includes the corrected coordinate maps and velocity fields. This file format can be used for further visualization and analysis in Tecplot-compatible software.


### viewing_nii.py
This a simple Python script that loads a specified NIfTI file using NiBabel and displays one of its slices with Matplotlib. It’s meant as a quick way to visually inspect the data—helping you confirm whether the unwarping or velocity corrections produced the expected results. By default, it shows an axial slice (usually the middle slice along the z-axis) in a chosen colormap (e.g., “jet” or "grey"), allowing for an immediate visual check of the file’s contents.


## Copyright
Both NiBabel and gradunwarp licenses ensure that this project can freely incorporate, adapt, and distribute their code as long as the original copyright and license notices remain intact.

### NiBabel
This project uses [NiBabel](https://nipy.org/nibabel/) for reading and writing NIfTI files. NiBabel is provided under the MIT license, which allows free use, distribution, and modification with minimal restrictions. For more details, see NiBabel’s own documentation and license files.

### Gradunwarp
Portions of this codebase are derived from the [gradunwarp](https://github.com/Washington-University/gradunwarp) package maintained by the WU-Minn Human Connectome Project (HCP). This gradunwarp code is also licensed under the MIT License, as stated below:

Copyright (c) 2012-2014 [WU-Minn Human Connectome Project consortium]
Copyright (c) 2009-2011 Matthew Brett
Copyright (c) 2010-2011 Stephan Gerhard
Copyright (c) 2006-2010 Michael Hanke
Copyright (c) 2010-2011 Jarrod Millman

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
