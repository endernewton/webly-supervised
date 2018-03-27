function [mscores, msizes, hscores, hsizes] = eldaDetectSeedScorer(iid, options)
% wrapper for getting the scores of the lowest edgebox score

pause(mod(iid,5) + 1);

datasets = dir(options.datapath);
datasets = datasets(3:end);

mscores = zeros(1000,1);
msizes = zeros(1000,1);
hscores = zeros(1000,11,'single');
hsizes = zeros(1000,11,'single');

st = 1;
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

        detfile = [tclasspath,'eldadets-',options.canditmethod,'.mat'];
        prefile1 = [tclasspath,'seed.mat'];
        prefile2 = [detpath,'seed.mat'];
        
        if fileExists(detfile) && fileExists(prefile1) && fileExists(prefile2)
            disp(detfile);
            clear AScores ABindex AIindex bboxes
            load(detfile,'AScores','ABindex','AIindex');
            load(prefile1,'bboxes');
            bboxes = cell2mat(bboxes);
            sizes = (bboxes(:,3) - bboxes(:,1) + 1) .* (bboxes(:,4) - bboxes(:,2) + 1);
            clear bboxes
            load(prefile2,'bboxes');
            tsize = size(AScores);
            scores = inf(tsize);
            rsizes = inf(tsize);
            for i=1:tsize(1)
                for j=1:tsize(2)
                    scores(i,j) = bboxes{AIindex(i,j)}(ABindex(i,j),5);
                    bbox = bboxes{AIindex(i,j)}(ABindex(i,j),1:4);
                    rsizes(i,j) = (bbox(3) - bbox(1) + 1) * (bbox(4) - bbox(2) + 1) / sizes(i);
                end
            end
            % keyboard;
            scores = scores(:);
            rsizes = rsizes(:);

            mscores(st) = min(scores);
            msizes(st) = min(rsizes);

            hscores(st,:) = hist(scores,0.0:0.05:0.5);
            hsizes(st,:) = hist(rsizes,0.0:0.1:1.0);

            st = st + 1;
        end
    end 
end

mscores(st:end) = [];
msizes(st:end) = [];
hscores(st:end,:) = [];
hsizes(st:end,:) = [];

end

