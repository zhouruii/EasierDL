logdir=$1
port=${2:-1234}

tensorboard --logdir $logdir --port $port --bind_all