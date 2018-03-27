function bboxes = seedEboxThres(impath, imidx, options)
% return the edge box proposals
% by xinleic

persistent model;
persistent opts;

if isempty(model)
	load(options.eboxmodel,'model'); 
	model.opts.multiscale=0; 
	model.opts.sharpen=2; 
	model.opts.nThreads=4;
end

if isempty(opts)
	opts = edgeBoxes;
	opts.alpha = options.eboxalpha;     % step size of sliding window search
	opts.beta  = options.eboxbeta;     % nms threshold for object proposals
	opts.minScore = options.eboxminScore;  % min score of boxes to detect
	opts.maxBoxes = options.eboxmaxBoxes;  % max number of boxes to detect
end

im = color(imread(impath));
h = size(im,1);
w = size(im,2);
s = h * w;
bbs = edgeBoxes(im,model,opts);
l = size(bbs,1);
ratios = bbs(:,3) .* bbs(:,4) / s;
bboxes = zeros(l,7,'single');
bboxes(:,1:2) = bbs(:,1:2);
bboxes(:,3:4) = bbs(:,1:2) + bbs(:,3:4) - 1; % do the tranformation
bboxes(:,5) = bbs(:,5);
bboxes(:,7) = imidx; % x1, y1, x2, y2, score_edge_box, detector_id, image_id
bboxes = bboxes(ratios > options.eboxSizeThres,:);

end