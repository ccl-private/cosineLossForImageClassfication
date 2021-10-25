#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# bash scripts/run_distribute_train.sh /data/ccl/RP2K_rp2k_dataset/all/train_augmented/ /home/ccl/Documents/codes/python/minds/models/research/cv/arcface/r.json

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATA_PATH RANK_TABLE"
echo "For example: bash run.sh /path/dataset /path/rank_table"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
DATA_PATH=$(get_real_path $1)
RANK_TABLE=$(get_real_path $2)

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export RANK_TABLE_FILE=$RANK_TABLE
export RANK_SIZE=2

python_arr=()
trap ctrlc SIGINT
function ctrlc() {
  for pid in ${python_arr[@]}; do
    kill -9 $pid
  done
  echo "all killed"
}

for((i=0;i<RANK_SIZE;i++))
do
    rm -rf device$i
    mkdir device$i
    cp -r ./src/ ./device$i
    cp train.py  ./device$i
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    export CUDA_VISIBLE_DEVICES=$i
    echo "start training for device $i"
    env > env$i.log
    python train.py --train_url ./device$i --data_url $DATA_PATH --device_num $RANK_SIZE --device_id $DEVICE_ID > train.log$i 2>&1 &
    python_arr+=($!)
    cd ../
done

for pid in ${python_arr[@]}; do
  wait $pid
done

echo "finish"
cd ../
