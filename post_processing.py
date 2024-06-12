import os.path
from skimage import io, measure, morphology, color
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np
from data_preprocessing import resize_with_aspect_ratio
import pandas as pd

def measure_rois(mask_path, mask):
    mask_number = os.path.basename(mask_path)
    # Fill the holes in the binary mask
    mask_filled = ndimage.binary_fill_holes(mask)

    # Label each connected component (region of interest) in the image
    labels = measure.label(mask_filled, connectivity=1)

    # Compute region properties and extract area of each region
    properties = measure.regionprops(labels)

    # Ignore regions that are too small
    min_region_area = 5  
    labels_filtered = np.copy(labels)
    for region in properties:
        if region.area < min_region_area:
            labels_filtered[labels == region.label] = 0

    # Update properties after filtering
    properties_filtered = measure.regionprops(labels_filtered)
    areas_filtered = [prop.area for prop in properties_filtered]

    # Prepare the image for plotting
    image_label_overlay = color.label2rgb(labels_filtered, image=mask, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_label_overlay)
    measurements = []
    # Loop over the regions and add text annotation for each region
    for idx, region in enumerate(properties_filtered):
        # Draw min-rectangle around the region and label it with area
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        region_label = f'Area: {round(region.area * 90.81,3)}'
        ax.text(minc, minr, region_label, color='white')
        measurements.append({"Region": idx, "Area": region.area * 90.81})

    measurements_df = pd.DataFrame(measurements)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"results/measurements_image_{mask_number}.png")
    plt.close(fig)
    measurements_df.to_csv(f"results/measurements_file_{mask_number}.csv", index =False )
    print("------------------------")
    print("Measurements Saved")
    print("------------------------")


def measure_rois_no_save(mask_path, mask):
    total = 0
    mask_number = os.path.basename(mask_path)
    # Fill the holes in the binary mask
    mask_filled = ndimage.binary_fill_holes(mask)

    # Label each connected component (region of interest) in the image
    labels = measure.label(mask_filled, connectivity=1)

    # Compute region properties and extract area of each region
    properties = measure.regionprops(labels)

    # Ignore regions that are too small
    min_region_area = 5  # Set this value based on your knowledge of the problem
    labels_filtered = np.copy(labels)
    for region in properties:
        if region.area < min_region_area:
            labels_filtered[labels == region.label] = 0

    # Update properties after filtering
    properties_filtered = measure.regionprops(labels_filtered)
    areas_filtered = [prop.area for prop in properties_filtered]

    # Prepare the image for plotting
    image_label_overlay = color.label2rgb(labels_filtered, image=mask, bg_label=0)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_label_overlay)
    measurements = []
    # Loop over the regions and add text annotation for each region
    for idx, region in enumerate(properties_filtered):
        # Draw min-rectangle around the region and label it with area
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        region_label = f'Area: {round(region.area * 90.81,3)}'
        ax.text(minc, minr, region_label, color='white')
        measurements.append({"Region": idx, "Area": region.area * 90.81})
        total = total + (region.area *90.81)

    measurements_df = pd.DataFrame(measurements)
    ax.set_axis_off()
    return total

if __name__ == "__main__":
    mask_path = r"C:\Users\wdgst\Data\ShiData\WDG\UNET_12.21.23\SegmentedFinal\masks\8627-284 4-3 4X_mask.tif"
    mask = io.imread(mask_path)
    mask = resize_with_aspect_ratio("mask", mask, (256, 256), 'down')
    measure_rois(mask_path, mask)

