def scale_txt(path,dst_path):
    tmp = []
    with open(path,'r',encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            tmp.append(float(line.strip()))

    with open(dst_path,'w') as f:
        for data in tmp:
            f.write(f'{str(data / 10)}\n')



if __name__ == '__main__':
    path = 'data/spectral/val/labelSOM.txt'
    dst_path = 'data/spectral/val/labelSOM_scale.txt'
    scale_txt(path,dst_path)