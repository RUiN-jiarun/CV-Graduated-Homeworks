function normal = myPMS(data, m, p, L_render, I_render)
%% Get lighting directions
l = (data.s)'; % illumination direction
[height, width] = size(data.mask);
pixel_num = length(m);

%% Get observation lighting intensities and images
img_num = size(data.filenames, 1);
I_obs = zeros(pixel_num, img_num);
for i = 1 : img_num
    E = imread_datadir_re(data, i);
    vec = normal_img2vec(E, m);
    I_obs(:, i) = 0.3*vec(:,1) + 0.6*vec(:,2) + 0.1*vec(:,3);   % add weight to get intensity
end

%% Sort intensities and remove a specific proportion of noise
[I_sort, idx_sort] = sort(I_obs, 2);     % Store the indices of pixels
% size(I_sort)
I = I_sort(:, floor(img_num*p):ceil(img_num*(1-p)));
idx = idx_sort(:, floor(img_num*p):ceil(img_num*(1-p)));
% size(idx_sort)

%% No noise removal
% I = I_obs;
% idx = zeros(pixel_num, size(l, 2));
% [row, col] = size(idx);
% for i = 1:row
%     for j = 1:col
%         idx(i,j) = j;
%     end
% end

%% Calculate b using least square
b = zeros(pixel_num, 3); 
for i = 1: pixel_num
   L = l(:,idx(i,:)); 
   % I = L * b
   b(i,:) = I(i,:) * pinv(L);
end

%% Calculate albedo and normal from b
albedo_vec = zeros(height * width, 1);
n_vec = zeros(height * width, 3);
for i = 1 : pixel_num
    albedo_vec(m(i),1) = norm(b(i,:));
    n_vec(m(i),:) = (b(i, :) / albedo_vec(m(i),:));
end

%% Reshape albedo map and normal map
albedo = reshape(albedo_vec, height, width, 1);
normal = reshape(n_vec, height, width, 3);

%% Render images
img_vec = albedo_vec .* n_vec * L_render' * I_render;
render = reshape(img_vec, height, width, 3);

%% Save results
dataName = data.filenames{1}(12:strlength(data.filenames{1})-8);
imwrite(albedo, strcat('../results/', dataName, '_Albedo.png'));
imwrite(render, strcat('../results/', dataName, '_Render.png'));
save(strcat('../results/', dataName, '_Albedo.mat'), 'albedo');