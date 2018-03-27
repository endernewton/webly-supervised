function ims = extractPatches(im, boxes, image_mean, options)
% modified from RCNN
% convert image to BGR and single
im = single(im(:,:,[3 2 1]));
num_boxes = size(boxes, 1);
crop_size = size(image_mean,1);
ims = zeros(crop_size, crop_size, 3, num_boxes, 'single');

for j=1:num_boxes
  bbox = boxes(j,:);
  crop = imageCrop(im, bbox, options.cropmethod, crop_size, options.croppadding, image_mean);
  % swap dims 1 and 2 to make width the fastest dimension (for caffe)
  ims(:,:,:,j) = permute(crop, [2 1 3]);
end

end