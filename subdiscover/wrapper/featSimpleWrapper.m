function options = featSimpleWrapper(iid, options)
% wrapper for getting the simple features

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

        % disp(['Seed selection: ', options.seedmethod]);
        tclasspath = [datasetcache,classname,'/',options.seedmethod,'/'];

        makeDirOrFail(tclasspath);
        pclasspath = [datasetcache,classname,'/paths.mat'];
        
        detpath = tclasspath;
        detfile = [detpath,'feats.mat'];
        prefile = [detpath,'seed.mat'];
        lockpath = [detpath,'feats.lock'];
        
        if fileExists(pclasspath) && fileExists(prefile) && ~fileExists(detfile) && makeDirOrFail(lockpath)
            clear imagelist bboxes
            load(pclasspath,'imagelist');
            load(prefile,'bboxes');
        	switch options.extractmethod
        	case 'rcnn'
        		features = featureSimple(bboxes, imagelist, options);
        	otherwise
        		error('Extraction method not recognized!');
        	end
            save(detfile,'features');
            rmdir(lockpath);
            if toc(start) > options.timelimit
                error('Time limit!');
            end
        end
    end 
end

end

