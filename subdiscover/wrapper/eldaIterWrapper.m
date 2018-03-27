function options = eldaIterWrapper(iid, options)

start = tic;
pause(mod(iid,5) + 1);

% later should have another wrapper changing the model definition and path during a single run
if options.initkey ~= caffe('get_init_key')
  options.initkey = caffeInit([options.modelpath, sprintf(options.modeldef,options.layernum)], ...
    [options.modelpath, sprintf(options.modelext,options.iter)], ...
    options.usegpu); 
end

metafolder = [options.cachepath,'META/'];
makeDirOrFail(metafolder);

metafile = [metafolder,options.seedmethod,'-featavg-',options.normfeat,'.mat'];
load(metafile,'AveFeat');
AveFeat = AveFeat';

metafile = [metafolder,options.seedmethod,'-featvar-',options.normfeat,'.mat'];
load(metafile,'PrecFeat');

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

        disp(['Seed selection: ', options.seedmethod]);
        disp(['Candidate selection: ', options.canditmethod]);

        pclasspath = [datasetcache,classname,'/paths.mat'];
        tclasspath = [datasetcache,classname,'/',options.seedmethod,'/'];
        detpath = [datasetcache,classname,'/',options.canditmethod,'/'];        
        makeDirOrFail(tclasspath);
        makeDirOrFail(detpath);

        prefile = [tclasspath,'elda.mat'];
        prefile2 = [detpath,'seed.mat'];

        detfile = [tclasspath,'seediter-',options.canditmethod,'.mat'];
        detfile2 = [tclasspath,'eldaiter-',options.canditmethod,'.mat'];

        lockpath = [tclasspath,'seediter-',options.canditmethod,'.lck/'];
        cachepath = [tclasspath,'seediter-',options.canditmethod,'.lock/'];
        
        if fileExists(prefile) && fileExists(prefile2) && ~fileExists(detfile) && ~fileExists(detfile2)
            makeDirOrFail(cachepath);
            if iid == 1 && makeDirOrFail(lockpath)
                % do the detection
                clear eldas imagelist bboxes
                load(pclasspath,'imagelist');
                load(prefile,'eldas');
                load(prefile2,'bboxes');

                eldas = cell2mat(eldas);
                num_img = length(imagelist);
                matpaths = cell(num_img,1);
                for j=1:num_img
                    matpath = [cachepath,sprintf('%03d.mat',j)];
                    matpaths{j} = matpath;
                    llockpath = [cachepath,sprintf('%03d.lock',j)];
                    if ~fileExists(matpath) && makeDirOrFail(llockpath)
                        disp(matpath);

                        feats = featuresMultiple(bboxes{j}, imagelist{j}, options);
                        % then fire on the feats
                        elda = eldas(j,:);
                        ind = 0;
                        lind = inf;
                        indseq = [];
                        while ind ~= lind
                            scores = elda * feats;
                            lind = ind;
                            [~,ind] = max(scores);
                            indseq(end+1) = ind;
                            % get the new elda detector
                            feat = feats(:,ind) - AveFeat;
                            elda = PrecFeat * feat;
                            sfeat = max(sqrt(sum(elda.^2)),eps(1));
                            elda = elda ./ sfeat;
                            elda = elda';
                        end
                        disp(indseq);
                        
                        save(matpath,'indseq','ind','elda','sfeat');
                        rmdir(llockpath);
                        if toc(start) > options.timelimit
                            error('Time limit!');
                        end
                    end
                end

                while ~waitTillExists(matpaths)
                end

                % aggregate info
                Bboxes = bboxes;
                bboxes = cell(num_img,1);
                eldas = cell(num_img,1);
                sumfeat = zeros(num_img,1);
                indsequences = cell(num_img,1);

                bar = createProgressBar();
                for j=1:num_img
                    bar(j,num_img);
                    matpath = [cachepath,sprintf('%03d.mat',j)];
                    clear indseq ind elda sfeat
                    load(matpath,'indseq','ind','elda','sfeat');
                    indsequences{j} = indseq;
                    eldas{j} = elda;
                    sumfeat(j) = sfeat;
                    bboxes{j} = Bboxes{j}(ind,:);
                end

                save(detfile,'bboxes','indsequences');
                save(detfile2,'eldas','sumfeat');

                system(['rm -rvf ',cachepath]);
                system(['rm -rvf ',lockpath]);
                if toc(start) > options.timelimit
                    error('Time limit!');
                end
            elseif iid ~= 1
                % do the detection
                clear eldas imagelist bboxes
                load(pclasspath,'imagelist');
                load(prefile,'eldas');
                load(prefile2,'bboxes');

                eldas = cell2mat(eldas);
                num_img = length(imagelist);
                % matpaths = cell(num_img,1);
                for j=randperm(num_img)
                    matpath = [cachepath,sprintf('%03d.mat',j)];
                    % matpaths{j} = matpath;
                    llockpath = [cachepath,sprintf('%03d.lock',j)];
                    if ~fileExists(matpath) && makeDirOrFail(llockpath)
                        disp(matpath);

                        feats = featuresMultiple(bboxes{j}, imagelist{j}, options);
                        % then fire on the feats
                        elda = eldas(j,:);
                        ind = 0;
                        lind = inf;
                        indseq = [];
                        while ind ~= lind
                            scores = elda * feats;
                            lind = ind;
                            [~,ind] = max(scores);
                            indseq(end+1) = ind;
                            % get the new elda detector
                            feat = feats(:,ind) - AveFeat;
                            elda = PrecFeat * feat;
                            sfeat = max(sqrt(sum(elda.^2)),eps(1));
                            elda = elda ./ sfeat;
                            elda = elda';
                        end
                        disp(indseq);
                        
                        save(matpath,'indseq','ind','elda','sfeat');
                        rmdir(llockpath);
                        if toc(start) > options.timelimit
                            error('Time limit!');
                        end
                    end
                end
            end
        end
    end 
end

end

