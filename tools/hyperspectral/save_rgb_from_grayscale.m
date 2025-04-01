function save_rgb_from_grayscale(img1, img2, output_path)
    % 将两个灰度图像映射到 RGB 图像的 R 和 G 通道，并保存为 RGB 图像
    %
    % 输入参数：
    %   img1: 第一个灰度图像（映射到 R 通道）
    %   img2: 第二个灰度图像（映射到 G 通道）
    %   output_path: 输出文件路径（例如 'output.png'）
    %
    % 示例调用：
    %   img1 = rand(256, 256) * 2 - 0.5;  % 随机生成一个包含负数和大于 1 的值的图像
    %   img2 = rand(256, 256) * 2 - 0.5;  % 随机生成一个包含负数和大于 1 的值的图像
    %   save_rgb_from_grayscale(img1, img2, 'output.png');

    % 检查输入图像是否为灰度图像
    if size(img1, 3) ~= 1 || size(img2, 3) ~= 1
        error('输入图像必须是灰度图像');
    end

    % 处理负数：将负数替换为 0
    img1(img1 < 0) = 0;
    img2(img2 < 0) = 0;

    % 处理大于 1 的值：截断为 1
    img1(img1 > 1) = 1;
    img2(img2 > 1) = 1;

    % 标准化图像（归一化到 [0, 1]）
    img1 = mat2gray(img1);  % 归一化
    img2 = mat2gray(img2);  % 归一化

    % 将两个灰度图像映射到 RGB 图像的 R 和 G 通道
    [H, W] = size(img1);
    rgb_img = zeros(H, W, 3);  % 初始化 RGB 图像
    rgb_img(:, :, 1) = img1;   % R 通道
    rgb_img(:, :, 2) = img2;   % G 通道
    % B 通道留空（全为 0）

    % 保存 RGB 图像
    imwrite(rgb_img, output_path);
    fprintf('配准对比已保存到: %s\n', output_path);
end