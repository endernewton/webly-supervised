function boxes = randomBoxGen( thres, nGrid, overlap, options )
%randomBoxGen by Ender, xinleic@cs.cmu.edu
%   Nov 2nd, 2012

if nargin < 4
    options = [];
end

maxNum = 1000;
if isfield(options,'rBoxNum')
    maxNum = options.rBoxNum;
end

distort = 2;
if isfield(options,'distort')
    distort = options.distort;
end

minGrid = 1/nGrid;
grids = minGrid:minGrid:1;

boxes = zeros(maxNum,4);
% box = cell(maxNum,1);
opts.distance = 'euclidean';

for iter = 1:maxNum
    done = 0;   
    while ~done       
        span = 0;
        dis = inf;
        
        while span < thres || dis > distort
            x = randperm(nGrid,2);
            x = sort(x);
            x = grids(x);
            
            y = randperm(nGrid,2);
            y = sort(y);
            y = grids(y);
            
            ar = (x(2) - x(1) + minGrid) / (y(2) - y(1) + minGrid);
            dis = max(ar, 1/ar);
            span = (x(2) - x(1) + minGrid) * (y(2) - y(1) + minGrid);
        end
        
        Box = [x(1),y(1),x(2),y(2)];
        
        dist = distanceToSet(Box',boxes',opts);
        if all(dist > overlap)
            done  = 1;
            boxes(iter,:) = Box;
            % box{iter} = Box;
        end
    end
end

end

