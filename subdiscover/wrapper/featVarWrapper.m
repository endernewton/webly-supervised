function options = featVarWrapper(iid, options)
% wrapper for computing the variance of the features, aggregated

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

    tdatasetpath = [datasetcache,'/META-',options.seedmethod,'/'];
    makeDirOrFail(tdatasetpath);
    
    detfile = [tdatasetpath,'featvar-',options.normfeat,'.mat'];
    lockpath = [tdatasetpath,'featvar-',options.normfeat,'.lock'];

    if ~fileExists(detfile) && makeDirOrFail(lockpath)

        clear Varfeat Numfeat
        numimgsofar = 0;

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
            prefile = [tclasspath,'feats.mat'];
            
            clear features
            load(prefile,'features');
            features = cell2mat(features);
            numfeat = size(features,1);

            if numfeat == 0
                continue;
            end

        	switch options.normfeat
            case 'none'
                % nothing
        	case 'l1'
        		sumfeat = max(sum(abs(features),2),eps(1));
                dimfeat = size(features,2);
                disp('Normalizing L1..');
                bar = createProgressBar();
                for j=1:dimfeat
                    bar(j,dimfeat);
                    features(:,j) = features(:,j) ./ sumfeat;
                end
            case 'l2'
                sumfeat = max(sqrt(sum(features.^2,2)),eps(1));
                dimfeat = size(features,2);
                disp('Normalizing L2..');
                bar = createProgressBar();
                for j=1:dimfeat
                    bar(j,dimfeat);
                    features(:,j) = features(:,j) ./ sumfeat;
                end
        	end

            % minus the average
            for j=1:numfeat
                features(j,:) = features(j,:) - AveFeat;
            end

            % then compute the variance
            if ~exist('Varfeat','var')
                Varfeat = features' * features ./ numfeat;
                numimgsofar = numfeat;
            else
                numimgsofar = numimgsofar + numfeat;
                ratio = numfeat / numimgsofar; 
                Varfeat = Varfeat .* (1 - ratio) + features' * features ./ numimgsofar;
            end
        end

        save(detfile,'Varfeat','numimgsofar');

        rmdir(lockpath);
        if toc(start) > options.timelimit
            error('Time limit!');
        end

    end 
end

end

