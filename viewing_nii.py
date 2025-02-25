# Basic code used to quickly see a cross section of your .nii file, just enter path into img

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load the NIfTI file using nibabel
img = nib.load(r"C:\Old_Testing\fullWarp_abs.nii")

# Get the image data as a NumPy array
data = img.get_fdata()  # for newer nibabel versions; use get_data() for older ones

# Print the shape so you know the dimensions:
print("Data shape:", data.shape)

# Choose a slice index. Here, we use the middle slice along the axial axis.
slice_index = data.shape[2] // 2  # using the third dimension for axial view

# Extract the slice
slice_data = data[:, :, slice_index]

# Create a high-definition figure
plt.figure(figsize=(10, 10), dpi=150)  # larger figure and higher DPI

# Display the slice with a colorful colormap and smooth interpolation
im = plt.imshow(slice_data, cmap='grey', interpolation='bilinear')
plt.title(f"Axial Slice {slice_index}")
plt.axis('off')

# Add a colorbar with a label for scale
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label('Velocity', rotation=270, labelpad=15)

plt.show()
