from patchify import patchify
import numpy as np
import os
from PIL import Image
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def create_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)
  return path

hist_dir =r"C:\Users\wdgst\Data\ShiData\WDG\HistogramFinal" 
mask_dir = r"C:\Users\wdgst\Data\ShiData\WDG\SegmentedFinal/masks"
new_dir_hist = create_dir(os.path.join(hist_dir, "HistPatches"))
new_dir_mask = create_dir(os.path.join(mask_dir, "MaskPatches"))

# Get a list of all image files in the directory
hist_files = sorted([file for file in os.listdir(hist_dir) if file.endswith(".tif")])
mask_files = sorted([file for file in os.listdir(mask_dir) if file.endswith(".tif")])


# Process each image
x=0
p = 0
for b in range(len(hist_files)):
    hist_path = os.path.join(hist_dir, hist_files[x])
    mask_path = os.path.join(mask_dir, mask_files[x])
    print(hist_path)
    print(mask_path)
    hist = Image.open(hist_path)
    mask = Image.open(mask_path)
    target_height = 1536
    target_width = 2048
    # Resize the image
    resized_hist = hist.resize((target_width, target_height))
    resized_mask = mask.resize((target_width, target_height))
    imH = np.asarray(resized_hist)
    #imH = imH[..., 0]
    imM = np.asarray(resized_mask)
    # Create patches
    total = (6*8)*len(hist_files)
    patchesH = patchify(imH, (256, 256,3), step=256)
    patchesM = patchify(imM, (256, 256), step=256)
    # Save patches to the new_dir_hist directory
    for i in range(patchesH.shape[0]):
        for j in range(patchesH.shape[1]):
            patchH = patchesH[i, j]
            patch_filenameH = f"hist_patch_{hist_files[x]}_{i}_{j}.tif"
            print(patch_filenameH)
            patch_pathH = os.path.join(new_dir_hist, patch_filenameH)
            reshaped_array = np.reshape(patchH, (256, 256, 3))
            patch_imageH = Image.fromarray(reshaped_array)
            patch_imageH.save(patch_pathH)
            patchM = patchesM[i, j]
            patch_filenameM = f"mask_patch_{mask_files[x]}_{i}_{j}.tif"
            print(patch_filenameM)
            patch_pathM = os.path.join(new_dir_mask, patch_filenameM)
            patch_imageM = Image.fromarray(patchM)
            patch_imageM.save(patch_pathM)
            p+=1
    x+=1
    if(p%100 == 0):
        print((str) (x/total * 100) + "%")

