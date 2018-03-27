function options = featAvgWrapper(iid, options)
% wrapper for computing the average of the features

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

        % disp(['Seed selection: ', options.seedmethod]);
        tclasspath = [datasetcache,classname,'/',options.seedmethod,'/'];

        makeDirOrFail(tclasspath);
        
        detpath = tclasspath;
        detfile = [detpath,'featavg-',options.normfeat,'.mat'];
        prefile = [detpath,'feats.mat'];
        lockpath = [detpath,'featavg-',options.normfeat,'.lock'];
        
        if fileExists(prefile) && ~fileExists(detfile) && makeDirOrFail(lockpath)
            clear features
            load(prefile,'features');
            features = cell2mat(features);
            numfeat = size(features,1);
        	switch options.normfeat
        	case 'none'
        		avefeat = mean(features,1);
                save(detfile,'avefeat','numfeat');
        	case 'l1'
        		sumfeat = max(sum(abs(features),2),eps(1));
                dimfeat = size(features,2);
                disp('Normalizing L1..');
                bar = createProgressBar();
                for j=1:dimfeat
                    bar(j,dimfeat);
                    features(:,j) = features(:,j) ./ sumfeat;
                end
                avefeat = mean(features,1);
                save(detfile,'avefeat','numfeat','sumfeat');
            case 'l2'
                sumfeat = max(sqrt(sum(features.^2,2)),eps(1));
                dimfeat = size(features,2);
                disp('Normalizing L2..');
                bar = createProgressBar();
                for j=1:dimfeat
                    bar(j,dimfeat);
                    features(:,j) = features(:,j) ./ sumfeat;
                end
                avefeat = mean(features,1);
                save(detfile,'avefeat','numfeat','sumfeat');
        	end
            rmdir(lockpath);
            if toc(start) > options.timelimit
                error('Time limit!');
            end
        end
    end 
end

end

