from skimage import io, measure, morphology, color
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np
from data_preprocessing import resize_with_aspect_ratio

# Load the binary mask


def measure_rois(mask):
    # Fill the holes in the binary mask
    mask_filled = ndimage.binary_fill_holes(mask)

    # Label each connected component (region of interest) in the image
    labels = measure.label(mask_filled, connectivity=1)

    # Compute region properties and extract area of each region
    properties = measure.regionprops(labels)
    areas = [prop.area for prop in properties]
    mean = []
    #a = [64561.03, 36830.696, 2707.418, 61519.404, 1333.779]
    #[print(f"{area2} / {area} = {area2/area}") for area2, area in zip(sorted(a),sorted(areas))]
    #[mean.append(area2/area) for area2, area in zip(sorted(a),sorted(areas))]
    #ratio = sum(mean)/ len(mean)
    #print(ratio)

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

    # Loop over the regions and add text annotation for each region
    for region in properties_filtered:
        # Draw min-rectangle around the region and label it with area
        minr, minc, maxr, maxc = region.bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        region_label = f'Area: {round(region.area * 90.81,3)}'
        print(region.area* 90.81)
        ax.text(minc, minr, region_label, color='white')

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mask_path = r"C:\Users\wdgst\Data\ShiData\WDG\UNET_12.21.23\SegmentedFinal\masks\8627-284 4-3 4X_mask.tif"
    mask = io.imread(mask_path)
    mask = resize_with_aspect_ratio("mask", mask, (256, 256), 'down')
    measure_rois(mask)

# 1	64561.033	255	255	255
# 2	36830.696	255	255	255
# 3	2707.418	255	255	255
# 4	61519.404	255	255	255
# 5	1333.779	255	255	255

