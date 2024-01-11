#!/usr/bin/bash

#setup vars for hyperparams
export LR=0.001
export OPTIMIZER=adam
export EPOCHS=30
export BATCH_SIZE=4

export MODEL=memenet
export DATASET=memes
export LOSS="--regression" 
#export SOFTMAX="--softmax"
#export STREAMING="--fifo"
export MLATOR="--mlator" #needed when using 4 bit?

#make sure pip install pillow==9.0.1 is installed for Tensorboard

BEST_QCKPT=../ai8x-synthesis/trained/qat_best-q.pth.tar
#check if exists
if [ ! -f "$BEST_QCKPT" ]; then
    echo "BASH: Best QAT checkpoint not found, training new model..."

    # Train the model
    echo "BASH: Training the model (see tensorboard progress)..."
    cd ../ai8x-training

    conda run --live-stream -n ai8x-training \
    python train.py --enable-tensorboard $LOSS --lr $LR --optimizer $OPTIMIZER --epochs $EPOCHS --batch-size $BATCH_SIZE \
    --deterministic --compress policies/schedule.yaml --qat-policy \
    policies/qat_policy_$MODEL.yaml --model $MODEL --dataset $DATASET \
    --param-hist --device MAX78000 "$@"

    TRAINING_SUCCESS=$?

    if [ $TRAINING_SUCCESS -ne 0 ]; then
        echo "BASH: Training failed, exiting..."
        exit 1
    fi


    # Quantize the model
    BEST_CKPT=../ai8x-training/latest_log_dir/qat_best.pth.tar
    echo "BASH: Quantizing the model..."
    echo "BASH: this will not work if torch layers were used during training!"
    cd ../ai8x-synthesis
    conda run --live-stream -n ai8x-synthesis \
    python quantize.py $BEST_CKPT trained/qat_best-q.pth.tar --device MAX78000 -v "$@"

    QUANT_SUCCESS=$?
    if [ $QUANT_SUCCESS -ne 0 ]; then
        echo "BASH: Quantization failed, exiting..."
        exit 1
    fi
fi


# Evaluating the model
#save-sample means that it will save the sample with ID = 10 from test set in header format so eval can run on micro
echo "BASH: Evaluating the model..."
cd ../ai8x-training
conda run --live-stream -n ai8x-training \
python train.py --model $MODEL --dataset $DATASET $LOSS --evaluate \
--save-sample 10 \
--exp-load-weights-from $BEST_QCKPT \
-8 --device MAX78000 "$@"

EVAL_SUCCESS=$?
if [ $EVAL_SUCCESS -ne 0 ]; then
    echo "BASH: Eval failed, exiting..."
    exit 1
fi

#moving saved sample
echo "BASH: Moving saved sample..."
mv sample_$DATASET.npy ../ai8x-synthesis/tests/sample_$DATASET.npy


MOVE_SUCCESS=$?
if [ $MOVE_SUCCESS -ne 0 ]; then
    echo "BASH: Moving sample failed, exiting..."
    exit 1
fi

# Convert the model to C code
echo "BASH: Converting the model to C code..."
cd ../ai8x-synthesis
rm -rf synthed_net
conda run --live-stream -n ai8x-synthesis \
python ai8xize.py --test-dir synthed_net $STREAMING $MLATOR --prefix $MODEL --checkpoint-file \
trained/qat_best-q.pth.tar --config-file networks/$MODEL.yaml \
--sample-input tests/sample_$DATASET.npy $SOFTMAX --device MAX78000 --compact-data \
--mexpress --timer 0 --display-checkpoint --verbose --overwrite "$@"

GEN_SUCCESS=$?
if [ $GEN_SUCCESS -ne 0 ]; then
    echo "BASH: Gen code failed, exiting..."
    exit 1
fi