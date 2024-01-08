#!/usr/bin/bash

#setup vars for hyperparams
export LR=0.001
export OPTIMIZER=adam
export EPOCHS=30
export BATCH_SIZE=64

export MODEL=memenet
export DATASET=memes

#make sure pip install pillow==9.0.1 is installed for Tensorboard

# Train the model
echo "BASH: Training the model (see tensorboard progress)..."
cd ../ai8x-training
conda run --live-stream -n ai8x-training \
python train.py --enable-tensorboard --lr $LR --optimizer $OPTIMIZER --epochs $EPOCHS --batch-size $BATCH_SIZE \
--deterministic --compress policies/schedule.yaml --qat-policy \
policies/qat_policy_$MODEL.yaml --model $MODEL --dataset $DATASET --confusion \
--param-hist --pr-curves --embedding --device MAX78000 "$@"

# Quantize the model
echo "BASH: Quantizing the model..."
echo "BASH: this will not work if torch layers were used during training!"
cd ../ai8x-synthesis
conda run --live-stream -n ai8x-synthesis \
python quantize.py ../ai8x-training/latest_log_dir/qat_best.pth.tar trained/qat_best-q.pth.tar --device MAX78000 -v "$@"


# Evaluating the model
#save-sample means that it will save the sample with ID = 10 from test set in header format so eval can run on micro
echo "BASH: Evaluating the model..."
cd ../ai8x-training
conda run --live-stream -n ai8x-training \
python train.py --model $MODEL --dataset $DATASET --confusion --evaluate \
--save-sample 10 \
--exp-load-weights-from ../ai8x-synthesis/trained/qat_best-q.pth.tar \
-8 --device MAX78000 "$@"

#moving saved sample
echo "BASH: Moving saved sample..."
mv sample_$DATASET.npy ../ai8x-synthesis/tests/sample_$DATASET.npy

# Convert the model to C code
echo "BASH: Converting the model to C code..."
cd ../ai8x-synthesis
rm -rf synthed_net
conda run --live-stream -n ai8x-synthesis \
python ai8xize.py --test-dir synthed_net --prefix $MODEL --checkpoint-file \
trained/qat_best-q.pth.tar --config-file networks/$MODEL.yaml \
--sample-input tests/sample_$DATASET.npy --softmax --device MAX78000 --compact-data \
--mexpress --timer 0 --display-checkpoint --verbose --overwrite "$@"

