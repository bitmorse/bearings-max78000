#!/bin/bash
# This script creates symlinks from this repo to the ai8x-training and ai8x-synthesis repos

export MODEL=bearingnet
export DATASET=ims_bearings
export REPONAME=bearings-max78000

echo "BASH: Linking qat_policy_$MODEL.yaml to ai8x-training/policies/qat_policy_$MODEL.yaml..."
ln -sf ../../$REPONAME/training/qat_policy_$MODEL.yaml ../ai8x-training/policies/qat_policy_$MODEL.yaml

echo "BASH: Linking schedule_$MODEL.yaml to ai8x-training/policies/schedule_$MODEL.yaml..."
ln -sf ../../$REPONAME/training/schedule_$MODEL.yaml ../ai8x-training/policies/schedule_$MODEL.yaml

echo "BASH: Linking $MODEL.py to ai8x-training/models/$MODEL.py..."
ln -sf ../../$REPONAME/training/$MODEL.py ../ai8x-training/models/$MODEL.py

echo "BASH: Linking $DATASET.py to ai8x-training/datasets/$DATASET.py..."
ln -sf ../../$REPONAME/training/$DATASET.py ../ai8x-training/datasets/$DATASET.py

echo "BASH: Linking $MODEL.yaml to ai8x-synthesis/networks/$MODEL.yaml..."
ln -sf ../../$REPONAME/synthesis/$MODEL.yaml ../ai8x-synthesis/networks/$MODEL.yaml

echo "BASH: Linking subfolders inside data folder to ai8x-synthesis/data..."
ln -sf ../../$REPONAME/data/$DATASET ../ai8x-training/data/$DATASET