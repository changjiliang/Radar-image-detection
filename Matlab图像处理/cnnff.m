function net = cnnff(net, x)
    n = numel(net.net.layers);
    net.net.layers{1}.a{1} = x;
    inputmaps = 1;

    for l = 2 : n   %  for each layer
        if strcmp(net.net.layers{l}.type, 'c')
            %  !!below can probably be handled by insane matrix operations
            for j = 1 : net.net.layers{l}.outputmaps   %  for each output map
                %  create temp output map
                z = zeros(size(net.net.layers{l - 1}.a{1}) - [net.net.layers{l}.kernelsize - 1 net.net.layers{l}.kernelsize - 1 0]);
                for i = 1 : inputmaps   %  for each input map
                    %  convolve with corresponding kernel and add to temp output map
                    z = z + convn(net.net.layers{l - 1}.a{i}, net.net.layers{l}.k{i}{j}, 'valid');
                end
                %  add bias, pass through nonlinearity
                net.net.layers{l}.a{j} = sigm(z + net.net.layers{l}.b{j});
            end
            %  set number of input maps to this layers number of outputmaps
            inputmaps = net.net.layers{l}.outputmaps;
        elseif strcmp(net.net.layers{l}.type, 's')
            %  downsample
            for j = 1 : inputmaps
                z = convn(net.net.layers{l - 1}.a{j}, ones(net.net.layers{l}.scale) / (net.net.layers{l}.scale ^ 2), 'valid');   %  !! replace with variable
                net.net.layers{l}.a{j} = z(1 : net.net.layers{l}.scale : end, 1 : net.net.layers{l}.scale : end, :);
            end
        end
    end

    %  concatenate all end layer feature maps into vector
    net.net.fv = [];
    for j = 1 : numel(net.net.layers{n}.a)
        sa = size(net.net.layers{n}.a{j});
        net.net.fv = [net.net.fv; reshape(net.net.layers{n}.a{j}, sa(1) * sa(2), sa(3))];
    end
    %  feedforward into output perceptrons
    net.net.o = sigm(net.net.ffW * net.net.fv + repmat(net.net.ffb, 1, size(net.net.fv, 2)));

end
