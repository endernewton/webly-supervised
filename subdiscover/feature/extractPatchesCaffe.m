function images = extractPatchesCaffe(im, image_mean, options)
% This is now actually the Caffe style classification
% convert image to BGR and single
im = single(im(:,:,[3 2 1]));
img_size = size(image_mean,1);
im = imresize(im, [img_size img_size], 'bilinear') - image_mean;
images = zeros(img_size, img_size, 3, 10, 'single');
indices = [0 img_size-options.inputsize] + 1;
curr = 1;
for i = indices
  for j = indices
    images(:, :, :, curr) = ...
        permute(im(i:i+options.inputsize-1, j:j+options.inputsize-1, :), [2 1 3]);
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
    curr = curr + 1;
  end
end
center = floor(indices(2) / 2)+1;
images(:,:,:,5) = ...
    permute(im(center:center+options.inputsize-1,center:center+options.inputsize-1,:), ...
        [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);

end