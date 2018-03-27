function [scores, bindex, features] = scoreMultipleThres(classifiers, thres, bboxes, imagepath, options)
% extract features for each of the bboxes, and score them with the classifiers,
% when return the results, return the ones that have a detection score higher than the thres
% classifiers, weight vectors
% bboxes, the bboxes for the image
% imagepath, the path to the image
% by xinleic

% image mean load
persistent image_mean;
if isempty(image_mean)
    load(options.meanpath,'image_mean');
    off = floor((size(image_mean,1) - options.inputsize)/2)+1;
	image_mean = image_mean(off:off+options.inputsize-1, off:off+options.inputsize-1, :);
end

% read the image
im = color(imread(imagepath));

% do multiple batches
l = size(bboxes,1);
num_batch = ceil(l / options.batchsize);
num_cls = size(classifiers,1);
crop_size = size(image_mean,1);
input_caffe = cell(1);
input_caffe{1} = zeros(crop_size, crop_size, 3, options.batchsize, 'single');
feat_dim = size(classifiers,2);

raw_scores = zeros(num_cls, l, 'single');
raw_feats = zeros(feat_dim, l, 'single');

st = 1;
bar = createProgressBar();
for i=1:num_batch
	bar(i,num_batch);
	ed = min(st + options.batchsize - 1, l);
	num = ed - st + 1;
	% construct the batch
	input_caffe{1}(:,:,:,1:num) = extractPatches(im, bboxes(st:ed,:), image_mean, options);
	feat = caffe('forward', input_caffe);
	feat = feat{1};
	feat = feat(:);
	feat = reshape(feat, [feat_dim, options.batchsize]);
	raw_feats(:, st:ed) = feat(:,1:num);
	raw_scores(:, st:ed) = classifiers * feat(:,1:num);
	st = st + options.batchsize;
end

scores = cell(num_cls, 1);
bindex = cell(num_cls, 1);

% should have a ranked list after NMS
for i=1:num_cls
	bboxes(:,5) = raw_scores(i,:)'; % get the raw_scores
    ind = Nms(bboxes(:,1:5), options.eldaoverlap);
    score = raw_scores(i,ind); % should already been sorted
    lthis = sum(score >= thres(i));
    scores{i} = score(1:lthis);
    bindex{i} = ind(1:lthis);
end

selected = [];
for i=1:num_cls
	selected = union(selected,bindex{i});
end
% selected = unique(cell2mat(bindex))';
features = raw_feats(:,selected)';

end