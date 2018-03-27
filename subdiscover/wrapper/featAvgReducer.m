function options = featAvgReducer(iid, options)
% wrapper for computing the average of the features

assert(iid == 1); % only the main one can do it
pause(mod(iid,5) + 1);

datasets = dir(options.datapath);
datasets = datasets(3:end);

numimgsofar = 0;

metafolder = [options.cachepath,'META/'];
makeDirOrFail(metafolder);
metafile = [metafolder,options.seedmethod,'-featavg-',options.normfeat,'.mat'];

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
        makeDirOrFail(tclasspath);
        
        detpath = tclasspath;
        detfile = [detpath,'featavg-',options.normfeat,'.mat'];
        
        if fileExists(detfile)
            % load the features
            clear avefeat numfeat
            load(detfile,'avefeat','numfeat');

            if numfeat == 0
                continue;
            end

            if ~exist('AveFeat','var')
                % first time
                AveFeat = avefeat;
                numimgsofar = numfeat;
            else
                % then
                numimgsofar = numimgsofar + numfeat;
                ratio = numfeat / numimgsofar; 
                AveFeat = AveFeat * (1 - ratio) + avefeat * ratio;
            end
            fprintf('So far: %d, This: %d\n',numimgsofar, numfeat);
        else
            disp([detfile,': Does not exist!']);
        end
    end 
end

if fileExists(metafile)
    disp('The target file already exists! Overwriting..');
end

save(metafile,'numimgsofar','AveFeat');

end

