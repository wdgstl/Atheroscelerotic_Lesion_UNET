// Set the directory path where your CZI images are located
dir = "C:\\Users\\wdgst\\Data\\ShiData\\WDG\\UNET_12.21.23\\test_seg\\";

// Create a subfolder to save the masks
File.makeDirectory(dir + "masks");

setBatchMode("true");
setBatchMode("hide");

// Get the list of CZI files in the directory
list = getFileList(dir);
nFiles = list.length;

// Loop through each CZI file
for (i = 0; i < nFiles; i++) {
  // Get the file name
  fileName = list[i];

  // Check if the file is a CZI image
  if (endsWith(fileName, ".czi")) {
  	
  	// Construct the full file path
    filePath = dir + fileName;

    // Open the CZI image
    run("Bio-Formats Importer", "open=[" + filePath + "] color_mode=Default view=Hyperstack stack_order=XYCZT display_rois" );

    // Run ROI Manager
    run("ROI Manager...");
	
	roiManager("combine");
	
	run("Create Mask");
	
    // Save the mask
    saveAs("Tiff", dir + "masks/" + fileName.replace(".czi", "_mask.tif"));

    roiManager("Deselect");
    
    // Clear the ROI Manager
    roiManager("Reset");

    // Close the CZI image
    close();
  }
  close();
}
