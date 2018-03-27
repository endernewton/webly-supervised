function options = eldaOffsetWrapper(iid, options)
% wrapper for computing mu^T x Sigma^(-1) (x - mu) = mu^T x ~(w) x ||w||

start = tic;
pause(mod(iid,5) + 1);

metafolder = [options.cachepath,'META/'];
makeDirOrFail(metafolder);

metafile = [metafolder,options.seedmethod,'-featavg-',options.normfeat,'.mat'];
load(metafile,'AveFeat');

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
        
        detpath = tclasspath;
        detfile = [detpath,'offset.mat'];
        prefile = [detpath,'elda.mat'];
        lockpath = [detpath,'offset.lock'];
        
        if fileExists(prefile) && ~fileExists(detfile) && makeDirOrFail(lockpath)
            clear eldas
            load(prefile,'eldas');
            sizes = cellfun(@(x)size(x,1),eldas);
            numimg = sum(sizes);
            eldas = cell2mat(eldas);
            numfeat = size(eldas,2);
            % fire classifier on the mean
            offsets = - eldas * AveFeat';
            offsets = mat2cell(offsets,sizes,1);
            
            save(detfile,'offsets');
            rmdir(lockpath);
            if toc(start) > options.timelimit
                error('Time limit!');
            end
        end
    end
end

end

