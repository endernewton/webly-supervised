function hsizes = reduceClusterSizer(iid, options)
% wrapper for getting the relative size of the final bounding boxes

start = tic;
pause(mod(iid,5) + 1);

datasets = dir(options.datapath);
datasets = datasets(3:end);

hsizes = zeros(1000,10,'single');

st = 1;
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

        % pclasspath = [datasetcache,classname,'/paths.mat'];
        tclasspath = [datasetcache,classname,'/',options.seedmethod,'/'];

        matfile = [tclasspath,'merges-reds-',options.canditmethod,'.mat'];
        seedfile = [tclasspath,'/seed.mat'];
        
        if fileExists(seedfile) && fileExists(matfile)
            clear bboxes newdetections
            load(seedfile,'bboxes');
            load(matfile,'newdetections');

            newdetections = cell2mat(newdetections);
            bboxes = cell2mat(bboxes);
            l = length(newdetections);
            rsizes = zeros(l,1);
            for j=1:l
                indi = newdetections(j,5);
                rsizes(j) = (newdetections(j,3) - newdetections(j,1) + 1) / bboxes(indi,3) * ...
                        (newdetections(j,4) - newdetections(j,2) + 1) / bboxes(indi,4);
            end
            hsizes(st,:) = hist(rsizes,0.05:0.1:0.95) / length(rsizes);

            st = st + 1;
        end
    end 
end

hsizes = hsizes(1:st-1,:);

end

