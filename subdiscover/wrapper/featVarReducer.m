function options = featVarReducer(iid, options)
% wrapper for computing the average of the features

assert(iid == 1); % only the main one can do it
pause(mod(iid,5) + 1);

datasets = dir(options.datapath);
datasets = datasets(3:end);

Numimgsofar = 0;

metafolder = [options.cachepath,'META/'];
makeDirOrFail(metafolder);
metafile = [metafolder,options.seedmethod,'-featvar-',options.normfeat,'.mat'];

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
    
    tdatasetpath = [datasetcache,'/META-',options.seedmethod,'/'];
    makeDirOrFail(tdatasetpath);
    
    detfile = [tdatasetpath,'featvar-',options.normfeat,'.mat'];
    
    if fileExists(detfile)
        % load the features
        clear Varfeat numimgsofar
        load(detfile,'Varfeat','numimgsofar');

        if numimgsofar == 0
            continue;
        end

        if ~exist('VarFeat','var')
            % first time
            VarFeat = Varfeat;
            Numimgsofar = numimgsofar;
        else
            % then
            Numimgsofar = Numimgsofar + numimgsofar;
            ratio = numimgsofar / Numimgsofar; 
            VarFeat = VarFeat .* (1 - ratio) + Varfeat .* ratio;
        end
        fprintf('So far: %d, This: %d\n',Numimgsofar, numimgsofar);
    else
        disp([detfile,': Does not exist!']);
    end
end

if fileExists(metafile)
    disp('The target file already exists! Overwriting..');
end

PrecFeat = inv(VarFeat + diag(size(VarFeat,1)) * options.eldaregu); 
% so here the whitening matrix 

save(metafile,'Numimgsofar','VarFeat','PrecFeat');

end

