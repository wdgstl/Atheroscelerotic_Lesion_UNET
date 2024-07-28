#Use ImageJ to get masks and measurements from raw histology images and segmentations
import imagej

ij = imagej.init('sc.fiji:fiji')

convert_czi_to_tif = """
print(2)
"""


get_mask_script = """


"""

result = ij.py.run_script("Groovy", convert_czi_to_tif)

print(result)