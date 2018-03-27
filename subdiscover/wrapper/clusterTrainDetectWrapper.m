function options = clusterTrainDetectWrapper(iid, options)
% wrapper for train LDA cluster classifier and fire it on the representations to get the scores

start = tic;
pause(mod(iid,5) + 1);

metafolder = [options.cachepath,'META/'];
makeDirOrFail(metafolder);

metafile = [metafolder,options.seedmethod,'-featavg-',options.normfeat,'.mat'];
load(metafile,'AveFeat');

metafile = [metafolder,options.seedmethod,'-featvar-',options.normfeat,'.mat'];
load(metafile,'PrecFeat');

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
    
    datasetdisp = [options.disppath,datasets(d).name,'/'];
    makeDirOrFail(datasetdisp);
    system(['chmod +rx ',datasetdisp]);
    
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

        pclasspath = [datasetcache,classname,'/paths.mat'];
        tclasspath = [datasetcache,classname,'/',options.seedmethod,'/'];
        
        matfile = [tclasspath,'merges-reds-',options.canditmethod,'.mat'];
        tgtfile = [tclasspath,'merges-tgts-',options.canditmethod,'.mat'];
        tgtlock = [tclasspath,'merges-tgts-',options.canditmethod,'.lock'];

        if fileExists(pclasspath) && fileExists(matfile) && ~fileExists(tgtfile) && makeDirOrFail(tgtlock)
            clear newclusters newdetections imagelist
            load(pclasspath,'imagelist');
            load(matfile,'newclusters','newdetections');
            li = length(imagelist);
            newdetections = cell2mat(newdetections);
            detections = cell(li,1);
            for t=1:li
                detections{t} = newdetections(newdetections(:,end) == t,:);
            end

            switch options.extractmethod
            case 'rcnn'
                features = featureSimple(detections, imagelist, options);
            otherwise
                error('Extraction method not recognized!');
            end

            lc = length(newclusters);
            % sizes = cellfun(@(x)numel(x),newclusters);
            features = cell2mat(features);
            numfeat = size(features,2);

            eldas = zeros(lc,numfeat,'single');
            for c=1:lc
                thisc = newclusters{c};
                thisf = features(thisc,:);
                eldas(c,:) = mean(thisf,1) - AveFeat;
            end
            eldas = eldas * PrecFeat';

            sumfeat = max(sqrt(sum(eldas.^2,2)),eps(1));
            disp('normalize..');
            bar = createProgressBar();
            for j=1:numfeat
                bar(j,numfeat);
                eldas(:,j) = eldas(:,j) ./ sumfeat;
            end
            offsets = - eldas * AveFeat';

            % then file on the set of features to get the outputs
            tscores = cell(lc,1);
            for c=1:lc
                thisc = newclusters{c};
                thisf = features(thisc,:);
                tscores{c} = thisf * eldas(c,:)'; 
            end

            save(tgtfile, 'detections', 'eldas', 'sumfeat', 'offsets', 'tscores');
            system(['rm -rvf ',tgtlock]);
        end

        if toc(start) > options.timelimit
            error('Time limit!');
        end     
    end 
end

end

