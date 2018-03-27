function options = resizeImageWrapper( iid, options )
% mergeBillWrapper by Ender, xinleic@cs.cmu.edu
%  just resize the image to 256

st = tic;
pause(mod(iid,5) + 1);

crop_height = 256;
crop_width = 256;

datasets = dir(options.datapath);
datasets = datasets(3:end);

ld = length(datasets);

dindex = 1:ld;
for d=dindex
    if ~datasets(d).isdir
        continue;
    end
    
    datasetname = datasets(d).name;
    disp(datasetname);
    datasetpath = [options.datapath,datasets(d).name,'/'];
    % cachesetpath = [options.cachepath,datasets(d).name,'/'];
    
    classes = dir(datasetpath);
    classes = classes(3:end);

    lcc = length(classes);
    
    for i=1:lcc
        if ~classes(i).isdir
            continue;
        end
        
        classname = classes(i).name;
        classpath = [datasetpath,classes(i).name,'/'];
        % classcachepath = [cachesetpath,classes(i).name,'/'];
        
        imgfolder = [classpath,'images/'];
        % infofile = [classpath,'info.mat'];
        resfolder = [classpath,'resized/'];
        imgflagfile = [classpath,'resized.flag'];
        imgflaglock = [classpath,'resized.lock'];

        disp(classname);

        if ~fileExists(imgflagfile) && makeDirOrFail(imgflaglock) % && fileExists(infofile) 
            disp('Ready to be resized..');

            files = dir(imgfolder);
            files = files(3:end);
            lf = length(files);
            makeDirOrFail(resfolder);

            bar = createProgressBar();
            for f=1:lf
                bar(f,lf);
                im = imread([imgfolder,files(f).name]);
                im = imresize(im, [crop_height crop_width],'bilinear', 'antialiasing', false);
                imwrite(im,[resfolder,files(f).name]);
            end

            system(['touch ',imgflagfile]);
            system(['rm -rvf ',imgflaglock]);

            if toc(st) > options.timelimit
                error('Time limit!');
            end
        end
    end 
end

end
