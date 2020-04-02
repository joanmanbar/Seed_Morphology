# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:23:33 2020

@author: jbarreto@umn.edu
"""


# Festuca Seed Size Function


def Festuca_Seed_Size(Images_Folder, Image_format='.tif', Output_Folder='same', color_data = False, plots = False):
    
    
    # Import dependencies
    import os
    
    import numpy as np
    
    from skimage import io
    from skimage import filters
    from skimage import morphology
    from skimage.measure import label, regionprops
    
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
    if not os.path.exists("FSS_Output"):
        os.mkdir("FSS_Output")
        
        
    # Define function 
    def FSS(img_name):
        
        # Read image
        img0 = io.imread(img_name)  
        
        # Get image name as string
        Image_Name = img_name.split('\\')[-1] 
        Image_Name = Image_Name[:-4]
        
        # Make sure image has only 3 channels
        img0 = img0[:, :, 0:3]
        
        # Convert to gray
        # gray0 = img0 @ [0.2126, 0.7152, 0.0722]
        
        # Set image threshold
        # T = filters.threshold_otsu(gray0)
        
        # Segment gray image based on T
        # bw0 = gray0 > T
        
        # Or... Segment image based on Color Thresholder (Matlab)
        bw0 = img0[:, :, 1] > 65
         
        
        # Filter out objects whose area < 1000
        bw1 = morphology.remove_small_objects(bw0, min_size=1000)
        
        # Label seeds
        labeled_seeds, num_spikes = label(bw1, return_num = True)
        labels = bw1 * labeled_seeds
        
        RGB = np.asarray(img0)
        RGB = np.where(bw1[...,None], RGB, 0)
    
        labels = bw1 * labeled_seeds
        plt.imshow(labels)
        
        # Determine regions properties
        Reg_Props = regionprops(labeled_seeds)
        
        if plots == True:
            f, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
            # plt.ioff()
            ax0.imshow(RGB)
            ax1.imshow(labels)
            ax0.title.set_text("Thresholded RGB")
            ax1.title.set_text("Labeled Image")
            # im = Image.fromarray(img2)
            # im.save(out_image)
            plot_name = 'FSS_Output\\' + Image_Name + '.png'
            f.savefig(plot_name)
         
        
        # If user wants to measure color...
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
        Seeds = FSS(i)
        
        # How long did it take to run this image?
        print("The image", i.split('\\')[-1], "took", time.time() - image_time, "seconds to run.")
         
        # Append to each dataset       
        Seeds_data = Seeds_data.append(Seeds)
        
        
    
    
    # Export data to output directory
    Seeds_data.to_csv (r'FSS_Output\Seeds_data_test.csv', header=True, index=False)
    
    
    # How long did it take to run the whole code?
    print("This entire code took", time.time() - start_time, "seconds to run.")





