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
        bw1 = morphology.remove_small_objects(bw0, min_size=1000)
        
        # Apply first mask to rgb
        img1 = np.where(bw1[...,None], img0, 0)
        # plt.imshow(img1)
        
        
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
        gray0 = np.asarray(test0) @ [0.2126, 0.7152, 0.0722]
        
        # Erosion
        bw2 = erosion(gray0, selem = disk(2))
        
        # White top hat: the image minus its morphological opening (erosion + dilation). 
        bw2 = white_tophat(bw2, selem = disk(7))
        plt.imshow(bw2, cmap = 'gray')
       
        # Define binary
        bw2 = bw2[:,:] > 0
        
        # Reconstruct the binary
        bw3 = reconstruction(bw0, bw2, method='erosion')        
        
        # Fill up holes
        bw3 = ndimage.binary_fill_holes(bw3)
        
        # Filter out objects whose area < 400
        bw3 = morphology.remove_small_objects(bw3, min_size=400)
        plt.imshow(bw3)        
        
        
        
        #############################################
        #####-----   Measurements 
        
        
        # Label seeds
        labeled_seeds, num_spikes = label(bw3, return_num = True)
        labels = bw3 * labeled_seeds        
              
        # Determine regions properties
        Reg_Props = regionprops(labeled_seeds)
        
        # Plots?
        if plots == True:
            
            # Apply mask to RGB
            RGB = np.asarray(img0)
            RGB = np.where(bw3[...,None], RGB, 0)
            
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
        Seeds = FSS(i)
        
        # How long did it take to run this image?
        print("The image", i.split('\\')[-1], "took", time.time() - image_time, "seconds to run.")
         
        # Append to each dataset       
        Seeds_data = Seeds_data.append(Seeds)
        
        
    
    
    # Export data to output directory
    Seeds_data.to_csv (r'FSS_Output\Seeds_data_test.csv', header=True, index=False)
    
    
    # How long did it take to run the whole code?
    print("This entire code took", time.time() - start_time, "seconds to run.")



# Coming soon...
#   Subset false seeds based on are, length width proportion and others...
    
    












##################################################################
###                 Wathershed
##################################################################
    
    
import numpy as np
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import random_walker
import matplotlib.pyplot as plt
from scipy import ndimage

# Generate an initial image with two overlapping circles
x, y = np.indices((80, 80))
x1, y1, x2, y2 = 28, 28, 44, 52
r1, r2 = 16, 20
mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1 ** 2
mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2 ** 2
image = np.logical_or(mask_circle1, mask_circle2)
# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance
# to the background
distance = ndimage.distance_transform_edt(image)
local_maxi = peak_local_max(
    distance, indices=False, footprint=np.ones((3, 3)), labels=image)
markers = measure.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=image)

markers[~image] = -1
labels_rw = random_walker(image, markers)

plt.figure(figsize=(12, 3.5))
plt.subplot(141)
plt.imshow(image, cmap='gray', interpolation='nearest')
plt.axis('off')
plt.title('image')
plt.subplot(142)
plt.imshow(-distance, interpolation='nearest')
plt.axis('off')
plt.title('distance map')
plt.subplot(143)
plt.imshow(labels_ws, cmap='nipy_spectral', interpolation='nearest')
plt.axis('off')
plt.title('watershed segmentation')
plt.subplot(144)
plt.imshow(labels_rw, cmap='nipy_spectral', interpolation='nearest')
plt.axis('off')
plt.title('random walker segmentation')

plt.tight_layout()
plt.show()
    
    


# Another approach

IN_folder = r"C:\Users\jbarreto\Documents\GitHub\Seed_Morphology\Python\IN_folder"

OUT_folder = r"C:\Users\jbarreto\Documents\GitHub\Seed_Morphology\Python\OUT_folder"

Image_format = ".tif"

# Gather the image files (change path)
Images = IN_folder + '\*' + Image_format
Images = io.ImageCollection(Images)

from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage import morphology
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage import img_as_float
from skimage import data
from skimage import io, color



# Read image
# img0 = io.imread(img_name)  
img0 = Images[1]
# plt.imshow(img0)

# Get image name as string
# Image_Name = img_name.split('\\')[-1] 
# Image_Name = Image_Name[:-4]

# Make sure image has only 3 channels
rgb = img0[:, :, 0:3]

# Convert to gray
gray0 = rgb @ [0.2126, 0.7152, 0.0722]


# Threshold based on blue channel
bw0 = img0[:,:,2] > 125
# Filter out objects whose area < 1000
bw1 = morphology.remove_small_objects(bw0, min_size=1000)

# lab-based segmentation
lab = color.rgb2lab(rgb)
a_channel = lab[:,:,1]
a_threshold = 20
bw0 = a_channel < a_threshold


fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 3))

ax0.hist(a_channel.ravel(), 1024)
ax0.set_title("Histogram of the a* channel with threshold")
ax0.axvline(x=a_threshold, color='r', linestyle='dashed', linewidth=2)
ax0.set_xbound(-10, 100)
ax1.imshow(bw0)
ax1.set_title("a*-thresholded image")
ax1.axis('off')

fig.tight_layout()


# Convert image to a float that lie in [-1,1]
gray1 = np.asarray(gray0, dtype=np.uint8)

# Apply mask (bw0) to gray image)
gray1 = gray1 * bw1
plt.imshow(gray1, cmap="gray")

# Apply mask to rgb
img1 = np.where(bw1[...,None], img0, 0)
plt.imshow(img1)

# local gradient (disk(2) is used to keep edges thin)
# gradient = rank.gradient(gray1, disk(2))
# plt.imshow(gradient)



# Let's apply filters

import matplotlib.pyplot as plt
from skimage import filters
from skimage import feature

edges = filters.sobel(gray2)
plt.imshow(edges)


# Enhance image and anpply filter
from PIL import Image, ImageEnhance 
I = Image.fromarray(img1)
# I = Image.fromarray(gray1)

# Color (best around 3.5 or higher)
I_col = ImageEnhance.Color(I)
I_shp = ImageEnhance.Sharpness(I)
I_brg = ImageEnhance.Brightness(I)
I_con = ImageEnhance.Contrast(I)


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(I_col.enhance(3.5))
ax[0].set_title('Color = 3.5')
ax[1].imshow(I_shp.enhance(50))
ax[1].set_title('Sharpness = 50')
ax[2].imshow(I_brg.enhance(1.5))
ax[2].set_title('Brightness = 1.6')
ax[3].imshow(I_con.enhance(1.5))
ax[3].set_title('Contrast = 1.55')


test0 = I_shp.enhance(50)
# test0 = I_col.enhance(3.5)
test0 = ImageEnhance.Color(test0)
# test0 = ImageEnhance.Sharpness(test0)
test0 = test0.enhance(5)
plt.imshow(test0)


# Convert to gray
gray0 = np.asarray(test0) @ [0.2126, 0.7152, 0.0722]


# Morphological operations
from skimage.morphology import erosion, dilation, opening, closing, white_tophat

# Selem
selem = disk(2)
eroded = erosion(gray0, selem)
eroded = white_tophat(eroded, disk(7))
plt.imshow(eroded, cmap = 'gray')



bw2 = eroded[:,:] > 0
plt.imshow(bw2)



from skimage.morphology import reconstruction
filled = reconstruction(bw0, bw2, method='erosion')
plt.imshow(filled)

filled2 = ndimage.binary_fill_holes(filled)
plt.imshow(filled2)

f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
ax[0].imshow(filtered)
ax[1].imshow(filled2)



# Filter out objects whose area < 1000
filtered = morphology.remove_small_objects(filled2, min_size=400)
plt.imshow(filtered)

# Label seeds
labeled_seeds, num_seeds = label(filtered, return_num = True)
labels = filtered * labeled_seeds
plt.imshow(labels)


RGB = np.asarray(img0)
RGB = np.where(filtered[...,None], RGB, 0)

# Determine regions properties
Reg_Props = regionprops(labeled_seeds)



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
    
    # Images_Names.append(Image_Name)
    Seeds.append(Seed)
    Areas.append(Area)
    Lengths.append(Length)
    Widths.append(Width)
    Orientations.append(Orientation)
    Circularitys.append(Circularity)
    Eccentricitys.append(Eccentricity)
    
    # If user wants to measure color...
    # if color_data == True:
    #     R =  red_means[ind]
    #     G =  green_means[ind]
    #     B =  blue_means[ind]
    #     Rs.append(R)
    #     Gs.append(G)
    #     Bs.append(B)



img_enh = rgb2gray()
plt.imshow(img_enh, cmap='gray')

img2 = np.asarray(img_enh)
img2 = Image.fromarray(img2)
plot_name = "pic_01"
plot_name = plot_name + '.tiff'
os.chdir(str(OUT_folder))
img2.save(plot_name)





# Compute the Canny filter for two values of sigma
edges1 = feature.canny(im)
edges2 = feature.canny(im, sigma=3)

# display results
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                    sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.axis('off')
ax1.set_title('noisy image', fontsize=20)

ax2.imshow(edges1, cmap=plt.cm.gray)
ax2.axis('off')
ax2.set_title(r'Canny filter, $\sigma=1$', fontsize=20)

ax3.imshow(edges2, cmap=plt.cm.gray)
ax3.axis('off')
ax3.set_title(r'Canny filter, $\sigma=3$', fontsize=20)

fig.tight_layout()






from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
graytest = gray1
info = np.iinfo(graytest.dtype) # Get the information of the incoming image type
graytest = graytest.astype(np.float64) / info.max # normalize the data to 0 - 1

# astro = color.rgb2gray(data.astronaut())
psf = np.ones((5, 5)) / 25

deconvolved, _ = restoration.unsupervised_wiener(graytest, psf)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

ax[0].imshow(graytest, vmin=deconvolved.min(), vmax=deconvolved.max())
ax[0].axis('off')
ax[0].set_title('Data')

ax[1].imshow(deconvolved)
ax[1].axis('off')
ax[1].set_title('Self tuned restoration')

fig.tight_layout()

plt.show()



from skimage.filters.rank import entropy
entr_img = entropy(graytest, disk(2))
plt.imshow(entr_img)



img2 = np.asarray(img_enh)

lab = color.rgb2lab(img2)
L_channel = lab[:,:,0]
a_channel = lab[:,:,1]
b_channel = lab[:,:,2]
L_threshold = 99
a_threshold = range(-3, 4)
bw1 = L_channel > L_threshold
# bw2 = bw1 * (a_channel > -3 and a_channel < 3)
# bw2 = bw1 * (a_channel = a_threshold)

# bw2 = morphology.remove_small_objects(bw1, min_size=5)
gray2 = gray1 * bw1





bw3 = morphology.binary_opening(bw1)

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
ax = axes.ravel()
ax[0].imshow(bw2)
ax[1].imshow(bw3)

bw3 = morphology.area_opening(gray1, 8, connectivity = 1)

plt.imshow(bw2)


img3 = np.where(bw2[...,None], img1, 0)
plt.imshow(img3)

# Filter out objects whose area < 1000
bw2 = morphology.remove_small_objects(bw1, min_size=5)
        
# Apply mask to rgb
img3 = np.where(bw2[...,None], img0, 0)
plt.imshow(img3)

































img_name = r"C:\Users\jbarreto\Desktop\gray.tif"

plt.imshow(bw0)

# Read image
img0 = io.imread(img_name)  
img0 = img0[:, :, 0:3]

gray0 = img0 @ [0.2126, 0.7152, 0.0722]

# Contour finding
contours = measure.find_contours(gray2, 0.2)

# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(gray2)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

# Not bad, but not good enough
















