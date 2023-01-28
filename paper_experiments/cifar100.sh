#!/usr/bin/env bash

pushd ../models

declare -a alphas=("1000" "0" "0.5")
wandb_api_key=$1

function run_fedavg() {
  echo "############################################## FedAVG CIFAR 100 ##############################################"
  alpha="$1"
  python main.py -api ${wandb_api_key} -dataset cifar100 --num-rounds 1000 --eval-every 50 --batch-size 100 --num-epochs 1 --clients-per-round 10 -model cnn -lr 0.1 --weight-decay 0 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha}
}

for alpha in "${alphas[@]}"; do
  echo "Alpha ${alpha}"
  run_fedavg "${alpha}"
  echo "Done"
done