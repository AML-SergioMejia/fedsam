#!/usr/bin/env bash

pushd ../models

declare -a alphas=("1000" "0" "0.5")
wandb_api_key=$1

function run_fedavg() {
  echo "############################################## FedAVG CIFAR 100 ##############################################"
  alpha="$1"
  python main.py -api ${wandb_api_key} -dataset cifar100 --num-rounds 1000 --eval-every 100 --batch-size 128 --num-epochs 1 --clients-per-round 5 -model cnn -lr 1 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha}
}

function run_adabest() {
  echo "############################################## Running FedAvg ##############################################"
  alpha="$1"
  python main.py -api ${wandb_api_key} -dataset cifar100 --num-rounds 1000 --eval-every 50 --batch-size 128 --num-epochs 1 --clients-per-round 5 -model cnn -lr 1 --weight-decay 0.0001 -device cuda:0 -algorithm adabest --client-algorithm adabest --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha} -beta 0.95 -mu 0.02
}

function run_fedavg_with_swa() {
  echo "############################################## Running FedAvg with SWA ##############################################"
  alpha="$1"
  python main.py -api ${wandb_api_key} -dataset cifar100 --num-rounds 1000 --eval-every 100 --batch-size 128 --num-epochs 1 --clients-per-round 5 -model cnn -lr 1 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-lr 1 --server-opt sgd --num-workers 0 --where-loading init -alpha ${alpha} -swa --swa-c 10 --swa-start 7500
}

function run_fedsam() {
  echo "############################################## Running FedSAM ##############################################"
  alpha="$1"
  python main.py -api ${wandb_api_key} -dataset cifar100 --num-rounds 1000 --eval-every 100 --batch-size 128 --num-epochs 1 --clients-per-round 5 -model cnn -lr 1 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-opt sgd --server-lr 1 --num-workers 0 --where-loading init -alpha ${alpha} --client-algorithm sam -rho 0.1 -eta 0
}

function run_fedsam_with_swa() {
  echo "############################################## Running FedSAM with SWA ##############################################"
  alpha="$1"
  python main.py -api ${wandb_api_key} -dataset cifar100 --num-rounds 1000 --eval-every 100 --batch-size 128 --num-epochs 1 --clients-per-round 5 -model cnn -lr 1 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-opt sgd --server-lr 1 --num-workers 0 --where-loading init -alpha ${alpha} --client-algorithm sam -rho 0.1 -eta 0 -swa --swa-c 10
}

function run_fedasam() {
  echo "############################################## Running FedASAM ##############################################"
  alpha="$1"
  python main.py -api ${wandb_api_key} -dataset cifar100 --num-rounds 1000 --eval-every 100 --batch-size 128 --num-epochs 1 --clients-per-round 5 -model cnn -lr 1 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-opt sgd --server-lr 1 --num-workers 0 --where-loading init -alpha ${alpha} --client-algorithm asam -rho 0.7 -eta 0.2
}

function run_fedasam_with_swa() {
  echo "############################################## Running FedASAM with SWA ##############################################"
  alpha="$1"
  python main.py -api ${wandb_api_key} -dataset cifar100 --num-rounds 1000 --eval-every 100 --batch-size 128 --num-epochs 1 --clients-per-round 5 -model cnn -lr 1 --weight-decay 0.0004 -device cuda:0 -algorithm fedopt --server-opt sgd --server-lr 1 --num-workers 0 --where-loading init -alpha ${alpha} --client-algorithm asam -rho 0.7 -eta 0.2 -swa --swa-c 10
}

echo "####################### EXPERIMENTS ON CIFAR100 #######################"
for alpha in "${alphas[@]}"; do
  echo "Alpha ${alpha}"
  run_adabest "${alpha}"
  run_fedavg "${alpha}"
  echo "Done"
done