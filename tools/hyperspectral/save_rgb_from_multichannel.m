function save_rgb_from_multichannel(img_hs, r_idx, g_idx, b_idx, output_path)
    % 从多通道图像中提取指定通道并保存为 RGB 图像
    %
    % 输入参数：
    %   img_hs: 多通道图像，形状为 H x W x C
    %   r_idx: 红色通道的索引
    %   g_idx: 绿色通道的索引
    %   b_idx: 蓝色通道的索引
    %   output_path: 输出文件路径（例如 'output.png'）
    %
    % 示例调用：
    %   save_rgb_from_multichannel(img_hs, 10, 20, 30, 'output.png');

    % 提取指定通道
    R = img_hs(:, :, r_idx);  % 红色通道
    G = img_hs(:, :, g_idx);  % 绿色通道
    B = img_hs(:, :, b_idx);  % 蓝色通道

    % 将三个通道拼接为 RGB 图像
    rgb_img = cat(3, R, G, B);

    rgb_img(rgb_img < 0) = 0;

    % 归一化数据到 [0, 1] 范围
    rgb_img = double(rgb_img);  % 转换为 double 类型
    rgb_img = rgb_img - min(rgb_img(:));  % 减去最小值
    rgb_img = rgb_img / max(rgb_img(:));  % 除以最大值

    % 保存为 RGB 图像
    imwrite(rgb_img, output_path);
    fprintf('RGB 图像已保存到: %s\n', output_path);
end