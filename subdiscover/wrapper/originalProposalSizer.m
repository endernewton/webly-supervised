function hsizes = originalProposalSizer(iid, options)
% wrapper for getting the relative size of the seed bounding boxes

start = tic;
pause(mod(iid,5) + 1);

datasets = dir(options.datapath);
datasets = datasets(3:end);

hsizes = zeros(1000,11,'single');

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
        detpath = [datasetcache,classname,'/',options.canditmethod,'/'];

        matfile = [detpath,'/seed.mat'];
        seedfile = [tclasspath,'/seed.mat'];
        
        if fileExists(seedfile) && fileExists(matfile)
            clear bboxes seeds
            load(seedfile,'bboxes');
            N = load(matfile,'bboxes');

            seeds = cell2mat(N.bboxes);
            bboxes = cell2mat(bboxes);
            l = length(seeds);
            rsizes = zeros(l,1);
            for j=1:l
                indi = seeds(j,7);
                rsizes(j) = (seeds(j,3) - seeds(j,1) + 1) / bboxes(indi,3) * ...
                        (seeds(j,4) - seeds(j,2) + 1) / bboxes(indi,4);
            end
            hsizes(st,:) = hist(rsizes,0.0:0.1:1.0) / length(rsizes);

            st = st + 1;
        end
    end 
end

hsizes = hsizes(1:st-1,:);

end

