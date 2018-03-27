function feats = featureSimple(bboxes, imagelist, options)
% return the features, this only works when the number of patches for each image is less than the batch size 
% rcnn style: crop the centre, do not do data augmentation
% by xinleic

% image mean load
persistent image_mean;
if isempty(image_mean)
    load(options.meanpath,'image_mean');
    off = floor((size(image_mean,1) - options.inputsize)/2)+1;
	image_mean = image_mean(off:off+options.inputsize-1, off:off+options.inputsize-1, :);
end

% reconfigure the batch to get the features
l = size(bboxes,1);
sizes = cellfun(@(x)size(x,1),bboxes);
if any(sizes > options.batchsize)
	error('Cannot process files with number of boundinb boxes more than the batch size!');
end

ll = sum(sizes);
num_batches = ceil(ll / options.batchsize);
feats = cell(num_batches,1);

crop_size = size(image_mean,1);
input_caffe = cell(1);
input_caffe{1} = zeros(crop_size, crop_size, 3, options.batchsize, 'single');

st = 1;
ic = 1;
bar = createProgressBar();
for i=1:l
	im = color(imread(imagelist{i}));
	boxes = bboxes{i};
	if isempty(boxes)
		continue;
	end
	bar(i,l);
	ed = st + sizes(i) - 1;
	if ed > options.batchsize
		% process the data
		% disp(['Batch ', int2str(ic)]);
		feat = caffe('forward', input_caffe);
		feat = feat{1};
		feat = feat(:);
		if ic == 1
			feat_dim = length(feat) / options.batchsize;
		end
		feat = reshape(feat, [feat_dim, options.batchsize]);
		feats{ic} = feat(:,1:st-1)';
		ic = ic + 1;
		% get the new starting point
		st = 1;
		ed = st + sizes(i) - 1;
	end
	% the new one is added
	input_caffe{1}(:,:,:,st:ed) = extractPatches(im, boxes, image_mean, options);
	st = ed + 1;
end

% the last one
disp(['Batch ', int2str(ic)]);
feat = caffe('forward', input_caffe);
feat = feat{1};
feat = feat(:);
if ic == 1
	feat_dim = length(feat) / options.batchsize;
end
feat = reshape(feat, [feat_dim, options.batchsize]);
feats{ic} = feat(:,1:st-1)';

% then final processing
feats = cell2mat(feats);
feats = mat2cell(feats,sizes,feat_dim);

end