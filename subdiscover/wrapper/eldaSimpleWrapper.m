function options = eldaSimpleWrapper(iid, options)
% wrapper for getting the simple features

start = tic;
pause(mod(iid,5) + 1);

metafolder = [options.cachepath,'META/'];
makeDirOrFail(metafolder);

metafile = [metafolder,options.seedmethod,'-featavg-',options.normfeat,'.mat'];
load(metafile,'AveFeat');

metafile = [metafolder,options.seedmethod,'-featvar-',options.normfeat,'.mat'];
load(metafile,'PrecFeat');

datasets = dir(options.datapath);
datasets = datasets(3:end);

dindex = 1:length(datasets);
if strcmp(options.machine,'warp')
    dindex = length(datasets):-1:1;
end

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
        detfile = [detpath,'elda.mat'];
        prefile = [detpath,'feats.mat'];
        lockpath = [detpath,'elda.lock'];
        
        if fileExists(prefile) && ~fileExists(detfile) && makeDirOrFail(lockpath)
            clear features
            load(prefile,'features');
            sizes = cellfun(@(x)size(x,1),features);
            numimg = sum(sizes);
            if numimg > 0
                features = cell2mat(features);
                numfeat = size(features,2);
                % train classifier
                disp('train..');
                bar = createProgressBar();
                for j=1:numimg
                    bar(j,numimg)
                    features(j,:) = features(j,:) - AveFeat;
                end
                eldas = features * PrecFeat';
                % since the value does not matter, normalize it
                sumfeat = max(sqrt(sum(eldas.^2,2)),eps(1));
                disp('normalize..');
                bar = createProgressBar();
                for j=1:numfeat
                    bar(j,numfeat);
                    eldas(:,j) = eldas(:,j) ./ sumfeat;
                end
                eldas = mat2cell(eldas,sizes,numfeat);
            else
                eldas = [];
                sumfeat = [];
            end
            
            save(detfile,'eldas','sumfeat');
            rmdir(lockpath);
            if toc(start) > options.timelimit
                error('Time limit!');
            end
        end
    end
end

end

