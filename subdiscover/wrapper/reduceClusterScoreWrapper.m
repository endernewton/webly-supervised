function options = reduceClusterScoreWrapper(iid, options)
% wrapper for getting the instances in the clusters, just keep the big images

start = tic;
pause(mod(iid,5) + 1);

datasets = dir(options.datapath);
datasets = datasets(3:end);

overlap = options.overlapDouble;

dindex = 1:length(datasets);
for d=dindex
    if ~datasets(d).isdir
        continue;
    end
    
    datasetname = datasets(d).name;
    disp(datasetname);
    datasetpath = [options.datapath,datasets(d).name,'/'];
    datasetcache = [options.cachepath,datasets(d).name,'/'];
    makeDirOrFail(datasetcache);
    
    classes = dir(datasetpath);
    classes = classes(3:end);
    
    for i=1:length(classes)
        if ~classes(i).isdir
            continue;
        end

        classname = classes(i).name;
        classpath = [datasetpath,classes(i).name,'/'];
        disp(classname);

        if ~fileExists([classpath,'info.mat'])
            continue;
        end

        disp(['Seed selection: ', options.seedmethod]);
        disp(['Candidate selection: ', options.canditmethod]);

        pclasspath = [datasetcache,classname,'/paths.mat'];
        tclasspath = [datasetcache,classname,'/',options.seedmethod,'/'];

        detfile = [tclasspath,'merges-dets-',options.canditmethod,'.mat'];
        matfile = [tclasspath,'merges-reds-',options.canditmethod,'.mat'];
        lockfile = [tclasspath,'merges-reds-',options.canditmethod,'.lock'];
        
        if fileExists(detfile) && ~fileExists(matfile) && makeDirOrFail(lockfile)
            clear idx counts clusters detections
            load(detfile,'idx','counts','clusters','detections');
            if isempty(idx)
                newclusters = [];
                newdetections = [];
                save(matfile,'newclusters','newdetections');
                rmdir(lockfile);
                continue;
            end
            % then just find the big patches in each image
            l = length(detections);
            lc = length(clusters);
            imageindexes = cellfun2(@(x)x(:,5),detections);
            imageindexes = cell2mat(imageindexes);

            newdetections = cell(l,1);
            newclusters = cell(lc,1);

            for k=1:lc
                newclusters{k} = cell(1,l);
            end

            for k=1:l
                newdetections{k} = cell(lc,1);
            end

            C = 0;
            for j=1:l
                dets = detections{j};
                imind = find(imageindexes == dets(1,5)); % just get the image index
                init = imind(1);
                % intersect it with clusters
                for k=1:lc
                    thiscls = intersect(imind,clusters{k}) - init + 1; % should be continuous
                    detects = dets(thiscls,:);
                    % compute the size of each region
                    detects(:,5) = (dets(thiscls,3) - dets(thiscls,1) + 1) ...
                        .* (dets(thiscls,4) - dets(thiscls,2) + 1);
                    % then just do the selection using NMS 
                    [~,lidx] = NmsIdx(detects,overlap);
                    llidx = unique(lidx);
                    cc = 1;
                    nlidx = lidx;
                    for ll=llidx'
                        nlidx(lidx==ll) = cc;
                        cc = cc + 1;
                    end
                    lidx = nlidx;
                    clear nlidx

                    cidx = unique(lidx);
                    c = length(cidx);

                    cdetects = zeros(c,5); % x1, y1, x2, y2, image
                    cdetects(:,5) = dets(1,5); % the index to the image

                    for jj=1:c
                        ccidx = lidx == cidx(jj);

                        if sum(ccidx) == 1
                            cdetects(jj,1:4) = detects(ccidx,1:4);
                        else
                            % get the largest region
                            cdetects(jj,1:4) = [min(detects(ccidx,1:2)),max(detects(ccidx,3:4))];
                        end
                    end

                    newdetections{j}{k} = cdetects;
                    newclusters{k}{j} = (C+1):(C+c);

                    C = C + c;
                end

                newdetections{j} = cell2mat(newdetections{j});
            end

            for k=1:lc
                newclusters{k} = cell2mat(newclusters{k})';
            end

            save(matfile,'newclusters','newdetections');

            rmdir(lockfile);
            if toc(start) > options.timelimit
                error('Time limit!');
            end
        end
    end 
end

end

