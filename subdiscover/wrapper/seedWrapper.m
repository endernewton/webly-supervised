function options = seedWrapper(iid, options)
% wrapper for getting the seed images for further training

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

        disp(['Seed selection: ', options.seedmethod]);
        tclasspath = [datasetcache,classname,'/',options.seedmethod,'/'];

        makeDirOrFail(tclasspath);
        pclasspath = [datasetcache,classname,'/paths.mat'];
        pclasslock = [datasetcache,classname,'/paths.lock'];
        clear imagelist

        if iid == 1
            if fileExists(pclasspath)
                load(pclasspath,'imagelist');
            else
                if ~makeDirOrFail(pclasslock)
                    continue;
                else
                    images = dir([classpath,'images/*.*']);
                    images = images(3:end); % just get all the images
                    images = cat(1,{images(:).name})';
                    if ~isempty(images)
                        imagelist = strcat([classpath,'images/'],images);
                    else
                        imagelist = {};
                    end
                    save(pclasspath,'imagelist');
                    system(['rm -rvf ',pclasslock]);
                end
            end
        else
            waitTillExists({pclasspath});
            load(pclasspath,'imagelist');
        end
        
        detpath = tclasspath;
        detfile = [detpath,'seed.mat'];
        lockpath = [detpath,'seed.lock'];
        
        if ~fileExists(detfile) && makeDirOrFail(lockpath)
        	li = length(imagelist);
        	bboxes = cell(li,1);
        	bar = createProgressBar();
            for j=1:li
            	bar(j,li);
            	switch options.seedmethod
            	case 'full'
            		bboxes{j} = seedFull( imagelist{j}, j, options );
                case 'ebox'
                    bboxes{j} = seedEbox( imagelist{j}, j, options );
            	otherwise
            		error('Seed method not recognized!');
            	end
            end
            save(detfile,'bboxes');
            rmdir(lockpath);
            if toc(start) > options.timelimit
                error('Time limit!');
            end
        end
    end 
end

end

