function options = getPositivePatchesMoreWrapper(iid, options)
% wrapper for getting the negative patches

start = tic;
pause(mod(iid,5) + 1);

% later should have another wrapper changing the model definition and path during a single run
if options.initkey ~= caffe('get_init_key')
  options.initkey = caffeInit([options.modelpath, sprintf(options.modeldef,options.layernum)], ...
    [options.modelpath, sprintf(options.modelext,options.iter)], ...
    options.usegpu); 
end

overlap = options.proposaloverlap;

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

        tclasspath = [datasetcache,classname,'/',options.seedmethod,'/'];

        classcachepath = [datasetcache,classname,'/'];
        pclasspath = [classcachepath,'paths.mat'];

        prepath = [tclasspath,'merges-reds-eboxt.mat'];
        % makeDirOrFail(classcachepath);
        matpath = [tclasspath,'reds-more-eboxt-feats.mat'];
        lockpath = [tclasspath,'reds-more-eboxt-feats.lock'];

        if ~fileExists(pclasspath) || ...
            ~fileExists(prepath) || fileExists(matpath) || ...
            ~makeDirOrFail(lockpath)
            continue;
        end

        % do the clustering here
        clear imagelist newdetections
        load(pclasspath,'imagelist');
        load(prepath,'newdetections');
        li = length(imagelist);
        newdetections = cell2mat(newdetections);
        detections = cell(li,1);

        if ~isempty(newdetections)
            for j=1:li
                gtbox = newdetections(newdetections(:,end) == j, :);
                if ~isempty(gtbox)
                    thisbox = seedEbox( imagelist{j}, j, options );
                    thisbox = thisbox(:,[1:4,7]);
                    lt = size(thisbox,1);
                    indicator = false(lt,1);
                    for k=1:size(gtbox,1)
                        coord = gtbox(k,1:4);
                        Left = max(thisbox(:,1),coord(1));
                        Up = max(thisbox(:,2),coord(2));
                        Right = min(thisbox(:,3),coord(3));
                        Down = min(thisbox(:,4),coord(4));
                        hI = Right-Left;
                        vI = Down-Up;
                        thisind =  (hI > 0) & (vI > 0);
                        areaI = (hI+1).*(vI+1);
                        areaA = (thisbox(:,3) - thisbox(:,1) + 1) .* (thisbox(:,4) - thisbox(:,2) + 1);
                        areaB = (coord(3) - coord(1) + 1) * (coord(4) - coord(2) + 1);
                        tov = areaI ./ (areaA + areaB - areaI);
                        thisind = thisind & (tov >= overlap) & tov < 1;
                        indicator = indicator | thisind;
                    end
                    % working as positive
                    tic_toc_print('%d: pos bboxes %d/%d\n', j, sum(indicator), length(indicator));
                    detections{j} = [gtbox;thisbox(indicator,:)];
                end
            end
        end

        save(matpath,'detections');
        feats = featureSimple(detections, imagelist, options);
        save(matpath,'feats','-append');

        rmdir(lockpath);
        if toc(start) > options.timelimit
            error('Time limit!');
        end

    end 
end

caffe('reset');

end
