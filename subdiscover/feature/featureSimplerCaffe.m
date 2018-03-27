function feats = featureSimplerCaffe(imagelist, options)
% Just get full image results, RCNN style
% Now with Crop at the corners
% by xinleic

% image mean load
persistent image_mean;
if isempty(image_mean)
    load(options.meanpath,'image_mean');
    off = floor((size(image_mean,1) - options.inputsize)/2)+1;
	image_mean = image_mean(off:off+options.inputsize-1, off:off+options.inputsize-1, :);
end

% reconfigure the batch to get the features
l = size(imagelist,1);

num_batches = ceil(l * 10 / options.batchsize);
feats = cell(num_batches,1);

crop_size = size(image_mean,1);
input_caffe = cell(1);
input_caffe{1} = zeros(crop_size, crop_size, 3, options.batchsize, 'single');

st = 1;
ic = 1;
bar = createProgressBar();
for i=1:l
	im = color(imread(imagelist{i}));
	h = size(im,1);
	w = size(im,2);
	bar(i,l);
	ed = st + 9;
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
		ed = 10;
	end
	% the new one is added
	input_caffe{1}(:,:,:,st:ed) = extractPatchesCaffe(im, image_mean, options);
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
% then do the average
i = 1;
st = 1;
ed = 10;
for i=1:l
	feats(i,:) = mean(feats(st:ed,:),1);
	st = st + 10;
	ed = ed + 10;
end

feats(l+1:end,:) = [];

end