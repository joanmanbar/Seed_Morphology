# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:23:33 2020

@author: jbarreto@umn.edu
"""


# Festuca Seed Size Touching (FSST)


def FSST(Images_Folder, Image_format='.tif', Output_Folder='same', color_data = False, plots = False):
    
    print('\n' * 2)
    
    # Import dependencies
    import os
    
    import numpy as np
    
    from scipy import ndimage
    
    from skimage import io
    from skimage import morphology
    from skimage.measure import label, regionprops    
    from skimage.morphology import erosion, white_tophat
    from skimage.morphology import reconstruction
    from skimage.morphology import disk
    # from skimage import filters
      
    from PIL import Image, ImageEnhance 
    
    import matplotlib.pyplot as plt
    
    import pandas as pd
    
    import time
    # Start the timer
    start_time = time.time()
    
    
    # Change woring directory 
    if not Output_Folder == 'same':
        os.chdir(str(Output_Folder))
    else:
        os.chdir(str(Images_Folder))
    
    # Verify that output directory exists
    if not os.path.exists("FSST_Output"):
        os.mkdir("FSST_Output")
        
        
    # Define function 
    def FSS_image(img_name):
        
        # Read image
        img0 = io.imread(img_name)  
        
        # Get image name as string
        Image_Name = img_name.split('\\')[-1] 
        Image_Name = Image_Name[:-4]
        
        # Make sure image has only 3 channels
        img0 = img0[:, :, 0:3]
        
        # Convert to gray
        gray0 = img0 @ [0.2126, 0.7152, 0.0722]
        
        
        #############################################
        #####-----   initial segmentation      
        
        ## WARNING: 
        # The following segmentation and enhancing parameters are only applicable to our images. 
        # The parameters need to be adjusted to other images.
        
        # Set image threshold
        # T = filters.threshold_otsu(gray0)
        
        # Segment gray image based on T
        # bw0 = gray0 > T
        
        # Or... Segment image based on Color Thresholder (Matlab)
        bw0 = img0[:,:,2] > 125     # Values in blue channel > 125
        
        # Filter out objects whose area < 1000
        bw1 = morphology.remove_small_objects(bw0, min_size=1000)           # 600dpi
        
        # Apply first mask to gray
        img1 = bw1 * gray0
        img1 = np.asarray(img1, dtype='uint8')
        # plt.imshow(img1, cmap = 'gray')
        # ...or ... Apply first mask to rgb
        # img1 = np.where(bw1[...,None], img0, 0)
        
        
        
        #############################################
        #####-----   Enhancement              
        
        # Enhance image based on sharpness and color
        img1 = Image.fromarray(img1)
        Enh_I = ImageEnhance.Sharpness(img1)
        Enh_I = Enh_I.enhance(50)
        Enh_I = ImageEnhance.Color(Enh_I)
        Enh_I = Enh_I.enhance(5)
        # plt.imshow(Enh_I)
        
        
        #############################################
        #####-----   Morphological operations              
        
        # Convert to gray
        # gray0 = np.asarray(Enh_I) @ [0.2126, 0.7152, 0.0722]
        gray0 = np.asarray(Enh_I)
        
        # Erosion
        bw2 = erosion(gray0, selem = disk(2))
        
        # White top hat: the image minus its morphological opening (erosion + dilation). 
        bw2 = white_tophat(bw2, selem = disk(7))
        # plt.imshow(bw2, cmap = 'gray')
       
        # Define binary
        bw2 = bw2[:,:] > 0
        
        # Reconstruct the binary
        bw3 = reconstruction(bw0, bw2, method='erosion')        
        
        # Fill up holes
        bw3 = ndimage.binary_fill_holes(bw3)
        
        # Filter out objects whose area < 400
        bw3 = morphology.remove_small_objects(bw3, min_size=400)
        # plt.imshow(bw3)        
        
        
        
        #############################################
        #####-----   Measurements 
        
        
        # Label seeds
        labeled_seeds, num_spikes = label(bw3, return_num = True)
        labels = bw3 * labeled_seeds        
              
        # Determine regions properties
        Reg_Props = regionprops(labeled_seeds)
        
        # Plots?
        if plots == True:
            
            plt.ioff()
            # Apply mask to RGB
            RGB = np.asarray(img0)
            RGB = np.where(bw3[...,None], RGB, 0)
            
            f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
            ax0.imshow(RGB)
            ax1.imshow(labels)
            ax0.title.set_text("Thresholded RGB")
            ax1.title.set_text("Labeled Image")
            # im = Image.fromarray(img2)
            # im.save(out_image)
            plot_name = 'FSST_Output\\' + Image_Name + '.png'
            f.savefig(plot_name)
         
        
        # measuring color?
        if color_data == True:
            # Assign each color channel to a different variable
            red = img0[:, :, 0]
            green = img0[:, :, 1]
            blue = img0[:, :, 2]
            
            # Mean colors
            red_props = regionprops(labeled_seeds, intensity_image=red)
            red_means = [rp.mean_intensity for rp in red_props]
            
            green_props = regionprops(labeled_seeds, intensity_image=green)
            green_means = [gp.mean_intensity for gp in green_props]
            
            blue_props = regionprops(labeled_seeds, intensity_image=blue)
            blue_means = [bp.mean_intensity for bp in blue_props]
        
        
        # Create lists
        Images_Names = []
        Seeds = []
        Areas = []
        Lengths = []
        Widths = []
        Orientations = []
        Circularitys = []
        Eccentricitys = []
        Rs = []
        Gs = []
        Bs = []
        
        # Loop through the seeds in image     
        for ind,props in enumerate(Reg_Props):
            Seed = props.label
            Area = props.area
            Length = props.major_axis_length
            Width = props.minor_axis_length
            Orientation = props.orientation
            Circularity = (4 * np.pi * props.area) / (props.perimeter ** 2)
            Eccentricity = props.eccentricity
            # Image_Name = i
            # Image_Name = img_name.split('\\')[-1]        
            
            Images_Names.append(Image_Name)
            Seeds.append(Seed)
            Areas.append(Area)
            Lengths.append(Length)
            Widths.append(Width)
            Orientations.append(Orientation)
            Circularitys.append(Circularity)
            Eccentricitys.append(Eccentricity)
            
            # If user wants to measure color...
            if color_data == True:
                R =  red_means[ind]
                G =  green_means[ind]
                B =  blue_means[ind]
                Rs.append(R)
                Gs.append(G)
                Bs.append(B)
                
                 # Dataframe with single obervations per image
                Seeds_per_image = pd.DataFrame(list(zip(Images_Names, Seeds, Areas, Lengths, Widths, Orientations, Circularitys, Eccentricitys, Rs, Gs, Bs)), columns = ['Image_Name', 'Seed', 'Area', 'Length', 'Width', 'Orientation', 'Circularity', 'Eccentricity', 'Red_mean', 'Green_mean', 'Blue_mean'])
                
            else:
                # Dataframe with single obervations per image 
                Seeds_per_image = pd.DataFrame(list(zip(Images_Names, Seeds, Areas, Lengths, Widths, Orientations, Circularitys, Eccentricitys)), columns = ['Image_Name', 'Seed', 'Area', 'Length', 'Width', 'Orientation', 'Circularity', 'Eccentricity'])
                
        # Return dataset for current photo
        return(Seeds_per_image)
            



    # Gather the image files (change path)
    Images = Images_Folder + '\*' + Image_format
    Images = io.ImageCollection(Images)

    # Create a dataset for images in folder
    Seeds_data = pd.DataFrame()
    
    # Loop through images in folder
    for i in Images.files:
        
        # Set the initial time per image
        image_time = time.time()
        
        # Return the dataset from the function
        Seeds = FSS_image(i)
        
        # How long did it take to run this image?
        print('\n')
        print("The image", i.split('\\')[-1], "took", time.time() - image_time, "seconds to run.")
        print('\n')
        
        # Append to each dataset       
        Seeds_data = Seeds_data.append(Seeds)
        
        
    
    
    # Export data to output directory
    Seeds_data.to_csv (r'FSST_Output\Seeds_data_restuls.csv', header=True, index=False)
    
    
    # How long did it take to run the whole code?
    print('\n' * 2)
    print("This entire code took", time.time() - start_time, "seconds to run.")
    print('\n' * 2)



# Coming soon...
#   Subset false seeds based on area, length:width proportion and others...
    
    







