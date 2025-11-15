#!/bin/bash

# 默认参数
NPROC=4
SCRIPT="main_ddp.py"
CONFIG="configs/hdr_former/AVIRIS/ours/baseline_v4.yaml"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --script)
      SCRIPT="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done


# 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=$NPROC \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12345 \
    $SCRIPT \
    --config $CONFIG
