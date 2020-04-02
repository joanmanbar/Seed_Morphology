%% Detect seed color and dimensions in Festuca ovina

% Information:
% This was written to measure the size of Festuca seeds using RGB images
% 
%
% Written by:
% Joan M. Barreto Ortiz
% jbarreto@umn.edu | jmbarretoo@gmail.com
% 
% Dec 2, 2019



% Close all windows; Clear the Workspace; Clear the Command Window
close all; clear; clc;

% Call function to create 'data'
data = cssm();

% Call function to create output table; takes 'data' as argument
out_tbl = cssm2(data);

% Write table as .csv
writetable(out_tbl, 'Festuca_seed_size.csv');


% Start function
function    data = cssm()
    %#ok<*AGROW>
    %#ok<*NASGU>
    %#ok<*ASGLU>


    % Define images' directory by using pattern 
    imagefiles = dir('*.tif');
    
    % Get file names
    filenames = {imagefiles.name};
    
    % Number of files
    nfiles = length(imagefiles);
    
    % Create cell array with n rows + row with colnames by number of columns
    data = cell( nfiles+1, 8 );     % +1 --> row with colnames
    
    % Define column names
    data(1,:) = { 'Image_Name', 'Seed', 'Mean_Red', 'Mean_Green',...
        'Mean_Blue' ,'Area', 'Length' ,'Width' };
    
    % Start iteration
    for i = 1:nfiles
        
        % Read the current image
        currentimage = imread( imagefiles(i).name );
               
        % Define RGB channels
        Red_ch   = currentimage(:,:,1);
        Green_ch = currentimage(:,:,2);
        Blue_ch  = currentimage(:,:,3);
        
        % Threshold based on the red channel
        binary = Red_ch > 93;
        binary = Green_ch > 65;
        
        % Filter noise
        binary = bwareafilt(binary, 10);
        
        % Fill holes
        binary = imfill(binary, 4, 'holes');
        
        % Define the object's morphological shape
        se     = strel('disk',5,0);
        
        % Erode the objects based on shape
        binary = imerode(binary, se);
        
        % Get current file's name as string
        a = char(filenames(i));
        
        % Create BW mask' name (string)
        masks = (['mask_',a(1:end-4),'.jpg']);
        
        % Write BW mask
        imwrite(binary, masks);
        
        % Convert binary to uint8 so it can be multiplied by RGB
        binary2 = uint8(binary);
        
        % Two images appear to have 4 dimensions, keept just the RGB
        currentimage = currentimage(:,:,1:3);
        
        % Create new image with black background
%         I2 = currentimage .* repmat( binary2, [1 1 3] );
        I2 = bsxfun(@times, currentimage, cast(binary2, 'like', currentimage));
        
        
        % Save new RGB image
        RGB_name = (['BBG_',a(1:end-4),'.jpg']);
        imwrite(I2, RGB_name);
        
        
        % identify object in image (LabeledI); determine number of objects
        % (numSeeds)
        [labeledI, numSeeds] = bwlabel(binary);
        
        % Choose dimensional properties of objects in image (Areas, Lengths, Widths)
        props = regionprops(labeledI, 'Area', 'MajorAxisLength', 'MinorAxisLength');
        
        % Take each variable property and turn them into an array of size
        % n-by-1
        Areas  = reshape([props.Area]           , [], 1 );
        Length = reshape([props.MajorAxisLength], [], 1 );
        Width  = reshape([props.MinorAxisLength], [], 1 );
        
        % Choose Pixel Value Measurements of objects in each color channel (mean intensity and pixels in object) 
        propsR = regionprops(labeledI, Red_ch, 'MeanIntensity', 'PixelValues');
        propsG = regionprops(labeledI, Green_ch, 'MeanIntensity', 'PixelValues');
        propsB = regionprops(labeledI, Blue_ch, 'MeanIntensity', 'PixelValues');
        
        % Take each variable property and turn them into an array of size
        % n-by-1
        MeansR = reshape([propsR.MeanIntensity], [], 1 );
        MeansG = reshape([propsG.MeanIntensity], [], 1 );
        MeansB = reshape([propsB.MeanIntensity], [], 1 );
        
        % Fill in variables to 'data'
        data(i+1,:)  = { imagefiles(i).name, numSeeds, MeansR,...
            MeansG, MeansB, Areas, Length, Width };
        
        % Convert cell to table and provide variable names
        mytable = cell2table( data(2:end,:), 'VariableNames', data(1,:) );
        
               
    % End iteration
    end         

% End function
end             
  
    
%% ............


% Start function
function out_tbl = cssm2(data)
    
    % Define colnames
    colhead = data(1,:);
    
    % Call function 
    out_tbl = one_batch_of_seeds_to_table( data(2,:), colhead );
    
    %
    for j = 3:size(data, 1)        % number 1 means ROWS (2 -> columns)
        
        % Create a table per image (from the second image) 
        tbl = one_batch_of_seeds_to_table( data(j,:), colhead );
        
        % Concatenate vertically (just like 'cbind' in R) to 'out_tbl'
        out_tbl = vertcat( out_tbl, tbl );  
        
    % End loop    
    end
    
% End function   
end


% Start function
 function tbl = one_batch_of_seeds_to_table( body, colhead )  % body is 'data'
    
    % Get number of objects in first image
    len = length( body{3} );
    
    % Repeat image name len-by-1 times
    pic = repmat( body(1), len, 1 );
    
    % Add rep number (object number)
    sno = reshape( (1:len), [], 1 );
    
    % Create table with image name, rep number, variables. Add colnames.
    tbl = table( pic, sno, body{3:end}, 'VariableNames', colhead );
    
    
% End function    
 end







