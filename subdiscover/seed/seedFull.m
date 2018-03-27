function bboxes = seedFull(impath, imidx, options)
% return the full image
% by xinleic

i = imfinfo(impath);
bboxes = [1,1,i.Width,i.Height,0,0,imidx]; % x1, y1, x2, y2, score, detector_id, image_id

end