#!/bin/bash
#JSUB -q gpu
#JSUB -gpgpu 4
#JSUB -m gpu25
#JSUB -e error.%J
#JSUB -o output.%J
#JSUB -n 10
source /apps/software/anaconda3/etc/profile.d/conda.sh
conda activate HDRformer


bash launch_ddp.sh --nproc 4 --config configs/hdr_former/hpcp/ablation/wo_absm.yaml