function feats = featuresMultiple(bboxes, imagepath, options)
% extract features for each of the bboxes, and store them
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
crop_size = size(image_mean,1);
input_caffe = cell(1);
input_caffe{1} = zeros(crop_size, crop_size, 3, options.batchsize, 'single');

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
		feats = zeros(feat_dim, l, 'single');
	end
	feat = reshape(feat, [feat_dim, options.batchsize]);
	feats(:, st:ed) = feat(:,1:num);
	st = st + options.batchsize;
end

end