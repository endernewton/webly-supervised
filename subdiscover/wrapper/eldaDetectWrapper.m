function options = eldaDetectWrapper(iid, options)
% wrapper for getting the detections on the candidates

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
if strcmp(options.machine,'workhorse')
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

        disp(['Seed selection: ', options.seedmethod]);
        disp(['Candidate selection: ', options.canditmethod]);

        pclasspath = [datasetcache,classname,'/paths.mat'];
        tclasspath = [datasetcache,classname,'/',options.seedmethod,'/'];
        detpath = [datasetcache,classname,'/',options.canditmethod,'/'];        
        makeDirOrFail(tclasspath);
        makeDirOrFail(detpath);

        prefile = [tclasspath,'elda.mat'];
        prefile2 = [detpath,'seed.mat'];

        detfile = [tclasspath,'eldascores-',options.canditmethod,'.mat'];
        lockpath = [tclasspath,'eldascores-',options.canditmethod,'.lck/'];
        cachepath = [tclasspath,'eldascores-',options.canditmethod,'.lock/'];
        
        if fileExists(prefile) && fileExists(prefile2) && ~fileExists(detfile)
            makeDirOrFail(cachepath);
            if iid == 1 && makeDirOrFail(lockpath)
                % do the detection
                clear eldas imagelist bboxes
                load(pclasspath,'imagelist');
                load(prefile,'eldas');
                if isempty(eldas)
                    Scores = [];
                    Bindex = [];
                    save(detfile,'Scores','Bindex');
                    system(['rm -rvf ',cachepath]);
                    system(['rm -rvf ',lockpath]);
                end
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
                        clear scores bindex
                        [scores, bindex] = scoreMultiple(eldas, bboxes{j}, imagelist{j}, options);

                        save(matpath,'scores','bindex');
                        rmdir(llockpath);
                        if toc(start) > options.timelimit
                            error('Time limit!');
                        end
                    end
                end

                while ~waitTillExists(matpaths)
                end

                % aggregate info
                Scores = cell(num_img,1);
                Bindex = cell(num_img,1);

                bar = createProgressBar();
                for j=1:num_img
                    bar(j,num_img);
                    matpath = [cachepath,sprintf('%03d.mat',j)];
                    clear scores bindex
                    load(matpath,'scores','bindex');
                    Scores{j} = scores;
                    Bindex{j} = bindex;
                end

                save(detfile,'Scores','Bindex');
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
                if isempty(eldas)
                    continue;
                end
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
                        clear scores bindex
                        [scores, bindex] = scoreMultiple(eldas, bboxes{j}, imagelist{j}, options);

                        save(matpath,'scores','bindex');
                        try
                            rmdir(llockpath);
                        catch ME
                            disp(ME.message);
                        end
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

