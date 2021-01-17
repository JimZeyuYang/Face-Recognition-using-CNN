clear all;
close all;
clc;

opts = init_parameters();
imdb = load_data(opts);
net = cnn_network_init(opts.nclass);
net.meta.normalization.averageImage = imdb.images.data_mean;

trainfn = @cnn_train;

[net,info] = trainfn(net,imdb,getBatch(), ...
    'expDir',opts.expDir, ...
    net.meta.trainOpts, opts.train, ...
    'val',find(imdb.images.set == 3));


function net = cnn_network_init(nclass)
    rng('default');
    rng(0);
    
    f = 1/100;
    net.layers = {} ;
    
    net.layers{end+1} = struct(...
        'type', 'conv', ...
        'weights',{{f*randn(11,11,1,500,'single'), zeros(1,500,'single')}}, ...
        'stride',1, ...
        'pad',0);
    
    net.layers{end+1} = struct(...
        'type','pool', ...
        'pool',[2 2], ...
        'method','max', ...
        'stride',2, ...
        'pad',0);
    
     net.layers{end+1} = struct( ...
        'type','relu');
    
    net.layers{end+1} = struct(...
        'type', 'conv', ...
        'weights',{{f*randn(6,6,500,200,'single'), zeros(1,200,'single')}}, ...
        'stride',1, ...
        'pad',0);
    
    net.layers{end+1} = struct(...
        'type','pool', ...
        'pool',[2 2], ...
        'method','max', ...
        'stride',2, ...
        'pad',0);
    
     net.layers{end+1} = struct( ...
        'type','relu');
    
    net.layers{end+1} = struct(...
        'type', 'conv', ...
        'weights',{{f*randn(11,11,200,nclass,'single'), zeros(1,nclass,'single')}}, ...
        'stride',1, ...
        'pad',0);
    
    net.layers{end+1} = struct('type','softmaxloss');
    
    net.meta.inputSize = [64 64 1];
    net.meta.trainOpts.learningRate = 0.0005;
    net.meta.trainOpts.numEpochs = 150;
    net.meta.trainOpts.batchSize = 64;
    
    net = vl_simplenn_tidy(net);
    
    
    net = vl_simplenn_move(net, 'gpu');
    
    
    
end

function opts = init_parameters()
    run(fullfile('C:\Users\jimya\Documents\MATLAB\win_matconvnet-1.0-beta25\matconvnet-1.0-beta25\matconvnet-1.0-beta25\matlab\vl_setupnn.m'));
    opts.datadir = 'Data\Yale2';
    opts.expDir = 'model';
    opts.imdbPath = 'imdb.mat';
    
    opts.train = struct();
    opts.train.gpus = [1]; 
    opts.train.continue = true; 
    
    opts.nclass = 15;
end

function imdb = load_data(opts)
    if exist(opts.imdbPath,'file')
        imdb = load(opts.imdbPath);
        disp('load from imdb...');
    else
        imdb = cnn_setup_data(opts);
        mkdir(opts.expDir);
        save(opts.imdbPath,'-struct','imdb');
    end
end


function imdb = cnn_setup_data(opts)
    val_data = load(fullfile(opts.datadir,'val_data.mat'));
    train_data = load(fullfile(opts.datadir,'train_data.mat'));
    disp(val_data);
    disp(train_data);
    
    data = single(reshape(cat(3,val_data.val_images,train_data.train_images),64,64,1,[]));
    
    set = [3*ones(1,numel(val_data.val_labels)) 1*ones(1,numel(train_data.train_labels))];
    set = gpuArray(set);
    
    dataMean = mean(data(:,:,:,set==1),4);
    data = bsxfun(@minus,data,dataMean);
    data = gpuArray(data);
    
    labels = cat(2,val_data.val_labels,train_data.train_labels);
    labels = gpuArray(labels);
    
    imdb.images.data = data;
    imdb.images.data_mean = dataMean;
    imdb.images.labels = labels;
    imdb.images.set = set;
    imdb.meta.sets = {'train','val','test'};
end


function fn = getBatch()
    fn = @(x,y) getSimpleNNBatch(x,y);
end

function [images,labels] = getSimpleNNBatch(imdb,batch)
    images = imdb.images.data(:,:,:,batch);
    labels = imdb.images.labels(1,batch);
end