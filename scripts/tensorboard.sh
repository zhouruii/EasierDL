logdir=$1
port=${2:-6006}

tensorboard --logdir $logdir --port $port --bind_all