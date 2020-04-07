%% Seed morphology for touching seeds



rgb = imread('C:\Users\jbarreto\Desktop\Pic_test002.tif');
% I = rgb2gray(rgb);
% imshow(I);
% bw1 = rgb(:,:,3) > 65;
% bw2 = bwareafilt(bw1, [100, inf]);
% imshow(bw2);

% [BW, I] = rgb2mask(rgb);
[BW, ~] = rgb2mask(rgb);
BW2 = bwareafilt(BW, [100, inf]);
% imshow(BW2);

RGB = bsxfun(@times, rgb, cast(BW2, 'like', rgb));
% imshow(RGB);

I = rgb2gray(RGB);

imshow(imsharpen(I, 'Radius', 10)); % Good at 10
imshow(imsharpen(I, 'Amount', 30));  % Great around 30


I2 = imsharpen(I, 'Amount', 10);
% BW = imbinarize(I2);
BW3 = imfill(I2,'holes');
I3 = bsxfun(@times, I, cast(BW3, 'like', I));
I4 = imsharpen(I3, 'Amount', 30);
BW = imbinarize(I2);
BW2 = imfill(I4,'holes');
BW3 = imbinarize(J);
BW4 = imfill(BW3,8,'holes');


% I2 = imerode(I,strel('square',2));
I2 = imsharpen(I, 'Amount', 10);
J = imerode(I2,strel('square',2));
J = imopen(J,strel('disk',1));
% I2 = imbinarize(I2);
I3 = imopen(I2,strel('disk',1));
% I3 = imbinarize(I3);
I4 = imopen(I3,strel('disk',2));
I5 = imopen(I4,strel('disk',4));
% I5 = imbinarize(I5);
% I4 = imfill(I3,'holes');
% I5 = imerode(I4,strel('square',1));
I5 = imsharpen(I5, 'Amount', 5);
I5 = imbinarize(I5);
I5 = imfill(I5,'holes');
imshow(I5);

bw2 = bwareafilt(bw1, [100, inf]);


% J = imsubtract(imadd(I3,imtophat(I3,se)),imbothat(I3,se));
% J = imclose(I3,strel('disk',4));
% J = imopen(I2,strel('square',2));
J = imerode(I2,strel('square',2));
% J = imerode(J,strel('square',2));
J = imopen(J,strel('disk',1));
% J = imsharpen(J, 'Amount', 5);
% J = imfill(J,'holes');
% J = imbinarize(J);
% J = imfill(J,'holes', 8);

figure; 
subplot(1,3,1);imshow(I4);
subplot(1,3,2); imshow(I5);
subplot(1,3,3); imshow(J);
linkaxes;