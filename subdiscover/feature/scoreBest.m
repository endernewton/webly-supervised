function [bscore, bindex, avgscores] = scoreBest(classifiers, thisi, lengths, bboxes, imagepath, options)
% extract features for each of the bboxes, and score them with the classifiers
% classifiers, weight vectors
% thisi, index of this image, so that the score should not count
% lengths, to multiply the lengths back
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

for i=1:l
	raw_scores(:,i) = raw_scores(:,i) .* lengths;
end
% zero out the current image detector so that it does not affect
raw_scores(thisi,:) = 0;

% then sort based on the top average distance
raw_scores = sort(raw_scores,1,'descend');
options.densetop = min(num_cls,options.densetop);
raw_scores = raw_scores(1:options.densetop,:);

avgscores = mean(raw_scores,1);
[bscore,bindex] = max(avgscores);

end