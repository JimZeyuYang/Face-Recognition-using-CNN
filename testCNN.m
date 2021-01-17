clear all;
close all;
clc;

opts = init_parameters();

imdb = load(fullfile('imdb.mat'));
im = load_test_data(opts,imdb);

net = load_network(opts);

accuracy = recognize(net,im);

fprintf('Accuracy %f',accuracy*100);
fprintf('%%\n');

function net = load_network(opts)
    net = load(fullfile(opts.modelDir,opts.modelName));
    net = net.net;
    net.layers{end}.type = 'softmax';
end


function opts = init_parameters()
    run(fullfile('C:\Users\jimya\Documents\MATLAB\win_matconvnet-1.0-beta25\matconvnet-1.0-beta25\matconvnet-1.0-beta25\matlab\vl_setupnn.m'));
    opts.datadir = 'Data\Yale2';
    opts.modelDir = 'model';
    models = dir(fullfile(opts.modelDir, '*.mat'));
    name = 'net-epoch-%d';
    opts.modelName = sprintf(name,length(models));
end

function im = load_test_data(opts,imdb)
    test_data = load(fullfile(opts.datadir,'test_data.mat'));
    data = single(reshape(test_data.test_images,64,64,1,[]));
    
    data = bsxfun(@minus,data,imdb.images.data_mean);
    
    set = 2*ones(1,numel(test_data.test_labels));
    
    labels = test_data.test_labels;
    
    im.images.data = data;
    im.images.labels = labels;
    im.images.set = set;
    im.meta.sets = {'train','val','test'};
end

function accuracy = recognize(net,im)
    res = vl_simplenn(net,im.images.data);
    disp(res(end));
    scores = squeeze(gather(res(end).x));
    [~, best] = max(scores);
    accuracy = length(find(best==im.images.labels))/length(im.images.labels);
end