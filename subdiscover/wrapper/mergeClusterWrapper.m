function options = mergeClusterWrapper(iid, options)
% wrapper for getting the clusters on file

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
        detfile = [tclasspath,'eldadets-',options.canditmethod,'.mat'];
        matfile = [tclasspath,'merge-dets-',options.canditmethod,'.mat'];
        lockpath = [tclasspath,'merge-dets-',options.canditmethod,'.lock'];

        if ~fileExists(pclasspath) || ...
            ~fileExists(seedfile) || ~fileExists(candfile) || ...
            ~fileExists(detfile) || fileExists(matfile) || ...
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
            [ idx, counts, clusters, detections, topNresults ] = ...
                mergeCluster( S.bboxes, C.bboxes, imagelist, AScores, AIindex, ABindex, options );
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

