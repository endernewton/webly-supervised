function options = eldaDetectReducer(iid, options)
% wrapper for getting the top detections among the categories

pause(mod(iid,5) + 1);

datasets = dir(options.datapath);
datasets = datasets(3:end);

left = options.eldatopall+1;

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
        makeDirOrFail(tclasspath);
        makeDirOrFail(detpath);

        prefile = [tclasspath,'eldascores-',options.canditmethod,'.mat'];
        detfile = [tclasspath,'eldadets-',options.canditmethod,'.mat'];
        detlock = [tclasspath,'eldadets-',options.canditmethod,'.lock'];
        
        if fileExists(prefile) && ~fileExists(detfile) && makeDirOrFail(detlock)
            disp(detfile);

            % the second version
            clear Scores Bindex
            load(prefile, 'Scores', 'Bindex');
            numim = length(Scores);
            if numim > 0
                % Indexes = cell2mat(cellfun2(@(x)(1:size(x,2)),Scores'))';
                Sizes = cellfun(@(x)size(x,2),Scores);
                Images = cell(numim,1);
                for j=1:numim
                    Images{j} = ones(Sizes(j),1) * j;
                end
                Images = cell2mat(Images);

                Scores = cell2mat(Scores')';
                Bindex = cell2mat(Bindex')';

                [~, SIndexes] = sort(Scores, 1, 'descend');
                SIndexes = SIndexes';
                numcls = size(Scores,2);
                AScores = -inf(numcls, left, 'single');
                ABindex = zeros(numcls, left, 'int32');
                AIindex = zeros(numcls, left, 'int32');

                for j=1:numim
                    AScores(j,1:left) = Scores(SIndexes(j,1:left),j)';
                    ABindex(j,1:left) = Bindex(SIndexes(j,1:left),j)';
                    AIindex(j,1:left) = Images(SIndexes(j,1:left));
                end
            else
                AScores = [];
                ABindex = [];
                AIindex = [];
            end

            save(detfile,'AScores','ABindex','AIindex');
            rmdir(detlock);
        end
    end 
end

end

