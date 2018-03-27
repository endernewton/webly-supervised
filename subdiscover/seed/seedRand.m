function bboxes = seedRand(impath, imidx, options)
% return the edge box proposals
% by xinleic

im = color(imread(impath));
h = size(im,1);
w = size(im,2);
thisbox = randomBoxGen(0.2, 40, 10^(-5), options);
clear im
thisbox(:,1) = max(round(thisbox(:,1) * w),1);
thisbox(:,3) = min(round(thisbox(:,3) * w),w);
thisbox(:,2) = max(round(thisbox(:,2) * h),1);
thisbox(:,4) = min(round(thisbox(:,4) * h),h);
l = size(thisbox,1);
bboxes = zeros(l,7,'single');
bboxes(:,1:2) = thisbox(:,1:2);
bboxes(:,3:4) = thisbox(:,3:4);
bboxes(:,5) = 0;
bboxes(:,7) = imidx; % x1, y1, x2, y2, score_edge_box, detector_id, image_id

end