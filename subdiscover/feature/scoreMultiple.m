function [scores, bindex] = scoreMultiple(classifiers, bboxes, imagepath, options)
% extract features for each of the bboxes, and score them with the classifiers
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

if l == 0
	scores = [];
	bindex = [];
end

num_batch = ceil(l / options.batchsize);
num_cls = size(classifiers,1);
crop_size = size(image_mean,1);
input_caffe = cell(1);
input_caffe{1} = zeros(crop_size, crop_size, 3, options.batchsize, 'single');

raw_scores = zeros(num_cls, l, 'single');

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
	if i == 1
		feat_dim = length(feat) / options.batchsize;
	end
	feat = reshape(feat, [feat_dim, options.batchsize]);
	raw_scores(:, st:ed) = classifiers * feat(:,1:num);
	st = st + options.batchsize;
end

minnum = min(options.eldatop,l);
scores = -inf(num_cls, options.eldatop, 'single');
bindex = zeros(num_cls, options.eldatop, 'int32');

% should have a ranked list after NMS
for i=1:num_cls
	bboxes(:,5) = raw_scores(i,:)'; % get the raw_scores
    ind = Nms(bboxes(:,1:5), options.eldaoverlap);
    score = raw_scores(i,ind); % should already been sorted
    li = length(ind);
    lthis = min(minnum,li);
    scores(i,1:lthis) = score(1:lthis);
    bindex(i,1:lthis) = ind(1:lthis);
end

end