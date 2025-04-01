%% 配置固定和移动图像所在目录
fixedDir = "/home/disk1/ZR/datasets/OurHSI/2023年12月20日栗峪口村高光谱数据/400-1000nm/3";
movingDir = "/home/disk1/ZR/datasets/OurHSI/2023年12月20日栗峪口村高光谱数据/900-2500nm/3";

%% 获取固定和移动图像文件列表（假设文件名称均为raw_*_rd_rf_or）
fixedFiles = dir(fullfile(fixedDir, 'raw_*_rd_rf_or'));
movingFiles = dir(fullfile(movingDir, 'raw_*_rd_rf_or'));

% 提取文件名并排序
fixedNames = sort({fixedFiles.name});
movingNames = sort({movingFiles.name});
disp(fixedNames)
disp(movingNames)

%% 遍历所有文件对，依次进行配准
for k = 1 : min(length(fixedNames), length(movingNames))

    fprintf('processing: %s, \n', movingNames{k});

    % 构造固定图像文件及其hdr文件路径
    fixedFile = fullfile(fixedDir, fixedNames{k});
    fixedHdrFile = fullfile(fixedDir, [fixedNames{k}, '.hdr']);

    % 读取固定图像数据
    hdr_fixed = read_envihdr(fixedHdrFile);
    img_hs_fixed = multibandread(fixedFile, hdr_fixed.size, [hdr_fixed.format '=>double'], ...
                                 hdr_fixed.header_offset, hdr_fixed.interleave, hdr_fixed.machine);
    fixed = img_hs_fixed(:,:,20);  % 选择第20个波段

    % 构造移动图像文件及其hdr文件路径
    movingFile = fullfile(movingDir, movingNames{k});
    movingHdrFile = fullfile(movingDir, [movingNames{k}, '.hdr']);

    % 读取移动图像数据
    hdr_moving = read_envihdr(movingHdrFile);
    img_hs_moving = multibandread(movingFile, hdr_moving.size, [hdr_moving.format '=>double'], ...
                                  hdr_moving.header_offset, hdr_moving.interleave, hdr_moving.machine);
    moving = img_hs_moving(:,:,20);  % 选择第20个波段

    % 配置优化器和度量准则（多模态）
    [optimizer, metric] = imregconfig('multimodal');
    optimizer.InitialRadius = optimizer.InitialRadius / 3.5;
    optimizer.MaximumIterations = 300;

    % 初始化存储配准后的移动图像数据
    [H, W, ~] = size(img_hs_fixed);
    [~, ~, C] = size(img_hs_moving);
    movingRegistered_all = zeros(H, W, C);

    % 对移动图像每个波段进行配准
    for band = 1:C
        fprintf('processing: %d th band of %s, \n', band, movingNames{k});
        moving_band = img_hs_moving(:,:,band);
        % 执行配准（仿射变换）
        movingRegistered = imregister(moving_band, fixed, 'affine', optimizer, metric);
        movingRegistered_all(:,:,band) = movingRegistered;
    end

    output_path = fullfile(movingDir, ['registered_', movingNames{k}, '.jpg']);
    save_rgb_from_grayscale(img_hs_fixed(:,:,2), movingRegistered_all(:,:,2), output_path);

    % 保存配准结果（例如保存为MAT文件）
    save_data = cat(3, img_hs_fixed, movingRegistered_all);
    outFile = fullfile(movingDir, ['registered_', movingNames{k}, '.mat']);
    fprintf('saving: %s, \n', movingNames{k});
    save(outFile, 'save_data', '-v7.3');

end
