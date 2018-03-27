function options = mergeClusterScoreWrapper(iid, options)
% wrapper for getting the clusters on file, the ranking in a single image is based on the score,
% which is the similarity based on dot product in the whitening space

start = tic;
pause(mod(iid,5) + 1);

datasets = dir(options.datapath);
datasets = datasets(3:end);

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
        detpath = [datasetcache,classname,'/',options.canditmethod,'/'];

        seedfile = [tclasspath,'/seed.mat'];
        candfile = [detpath,'/seed.mat'];
        eldafile = [tclasspath,'elda.mat'];
        offsetfile = [tclasspath,'offset.mat'];
        detfile = [tclasspath,'eldadets-',options.canditmethod,'.mat'];
        matfile = [tclasspath,'merges-dets-',options.canditmethod,'.mat'];
        lockpath = [tclasspath,'merges-dets-',options.canditmethod,'.lock'];

        if ~fileExists(pclasspath) || ...
            ~fileExists(seedfile) || ~fileExists(candfile) || ...
            ~fileExists(detfile) || fileExists(matfile) || ...
            ~fileExists(eldafile) || ~fileExists(offsetfile) || ...
             ~makeDirOrFail(lockpath)
            continue;
        end

        % do the clustering here
        clear imagelist AScores ABindex AIindex
        S = [];
        C = [];
        load(pclasspath,'imagelist');
        S = load(seedfile,'bboxes');
        C = load(candfile,'bboxes');
        load(detfile,'AScores','ABindex','AIindex');
        if ~isempty(AScores)
            % modifiy the scores
            load(eldafile,'sumfeat');
            load(offsetfile,'offsets');
            offsets = cell2mat(offsets);
            ltop = size(AScores,2);
            for j=1:ltop
                AScores(:,j) = (AScores(:,j) + offsets) .* sumfeat;
            end
            [ idx, counts, clusters, detections, topNresults ] = ...
                mergeClusterScoreIU( S.bboxes, C.bboxes, imagelist, AScores, AIindex, ABindex, options );
        else
            idx = [];
            counts = [];
            clusters = [];
            detections = [];
            topNresults = [];
        end
        
        save(matfile,'idx','counts','clusters','detections','topNresults');
        rmdir(lockpath);
        if toc(start) > options.timelimit
            error('Time limit!');
        end
    end 
end

end

