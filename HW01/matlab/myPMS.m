function normal = myPMS(data, m, p, L_render, Li_render)
%% Get lighting directions
l = (data.s)'; % illumination direction
[height, width, ~] = size(data.mask);
pixel_num = length(m);

%% Get observation lighting intensities and images
img_num = size(data.filenames, 1);
I_raw = zeros(pixel_num, img_num);
for i = 1 : img_num
    img = data.imgs{i};
    img = rgb2gray(img); % Convert RGB to intensity using rgb2gray
    img = img(m);       % Get target pixels by mask
    I_raw(:, i) = img;
end

%% Sort intensities and remove a specific proportion of noise
[I_sort, idx_sort] = sort(I_raw,2);
I = I_sort(:,floor(img_num*p):ceil(img_num*(1-p)));
idx = idx_sort(:,floor(img_num*p):ceil(img_num*(1-p)));
% size(idx_sort)

%% No noise removal
% I = I_raw;
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
   b(i,:) = I(i,:) * pinv(L); % least square
end

%% Calculate albedo and normal form b
albedo_vec = zeros(height * width, 1);
n_vec = zeros(height * width, 3);
for i = 1 : pixel_num
    albedo_vec(m(i),1) = norm(b(i,:));
    n_vec(m(i),1) = (b(i, 1) / albedo_vec(m(i),1));
    n_vec(m(i),2) = (b(i, 2) / albedo_vec(m(i),1));
    n_vec(m(i),3) = (b(i, 3) / albedo_vec(m(i),1));
end

%% Reshape albedo map
albedo= reshape(albedo_vec, height, width, 1);
normal = reshape(n_vec, height, width,3);

%% Render images
img_vec = albedo_vec .* n_vec * L_render' * Li_render;
render = reshape(img_vec, height, width, 3);
render = max(render, 0);

%% Save results
dataName = data.filenames{1}(12:strlength(data.filenames{1})-8);
imwrite(albedo, strcat('../results/', dataName, '_Albedo.png'));
imwrite(render, strcat('../results/', dataName, '_Render.png'));
save(strcat('../results/', dataName, '_Albedo.mat'), 'albedo');