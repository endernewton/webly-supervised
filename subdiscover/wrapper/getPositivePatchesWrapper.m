function options = getPositivePatchesWrapper(iid, options)
% wrapper for getting the negative patches

start = tic;
pause(mod(iid,5) + 1);

% later should have another wrapper changing the model definition and path during a single run
if options.initkey ~= caffe('get_init_key')
  options.initkey = caffeInit([options.modelpath, sprintf(options.modeldef,options.layernum)], ...
    [options.modelpath, sprintf(options.modelext,options.iter)], ...
    options.usegpu); 
end

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

        tclasspath = [datasetcache,classname,'/',options.seedmethod,'/'];

        classcachepath = [datasetcache,classname,'/'];
        pclasspath = [classcachepath,'paths.mat'];

        prepath = [tclasspath,'merges-reds-eboxt.mat'];
        % makeDirOrFail(classcachepath);
        matpath = [tclasspath,'reds-eboxt-feats.mat'];
        lockpath = [tclasspath,'reds-eboxt-feats.lock'];

        if ~fileExists(pclasspath) || ...
            ~fileExists(prepath) || fileExists(matpath) || ...
            ~makeDirOrFail(lockpath)
            continue;
        end

        % do the clustering here
        clear imagelist newdetections
        load(pclasspath,'imagelist');
        load(prepath,'newdetections');
        li = length(imagelist);
        newdetections = cell2mat(newdetections);

        detections = cell(li,1);
        if ~isempty(newdetections)
            for j=1:li
                detections{j} = newdetections(newdetections(:,end) == j, :);
            end
        end

        feats = featureSimple(detections, imagelist, options);
        
        save(matpath,'detections','feats');
        rmdir(lockpath);
        if toc(start) > options.timelimit
            caffe('reset');
            error('Time limit!');
        end
    end 
end

caffe('reset');

end
