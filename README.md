# bearings-max78000
Condition monitoring of bearings on MAX78000

* setup.sh creates symlinks to the model and dataset files from this repo to the ai8x-training and ai8x-synthesis repos
* train.sh is a script to run the training, quantization and synthesis scripts in one go (hyperparameters are set in the script)
* bearings-max78000/synthed_net contains the C code for the MAX78000
* playground.ipynb is a Jupyter notebook for testing the quantized model (the final output) on the test set. It should give a simulation of the microcontrollers output.
* firmware/ contains the firmware for the MAX78000 that was generated by train.sh. It can be compiled and flashed using the MSDK. It is manually copied and modified, so you should not overwrite it with synth_net.

## Getting started
* You are using Python 3.8.11, Torch 1.8, Cuda 11 and Ubuntu 22.04.
* You have cloned [ai8x-training](https://github.com/MaximIntegratedAI/ai8x-training) and [ai8x-synthesis](https://github.com/MaximIntegratedAI/ai8x-synthesis) and have them in the same parent directory as this repository.
* ai8x-training hash: b8bed2be513607427c487a6247dcd2963975b524
* ai8x-synthesis hash: 35228561aeca21165456904a6ed274ae1e84ea64
* Follow the guide in ai8x-training/README to install the required Python packages for both ai8x-training and ai8x-synthesis repositories. This entails setting up two different conda environments each. You may need to downgrade Pillow to 9.0.1 in order to use tensorboard.
* After setting up the conda environments, run ´setup.sh´ to create symlinks of the model and dataset files in ai8x-training and ai8x-synthesis. This will allow the training and synthesis scripts to find the files.
* If setup was successful, you can run ´train.sh´ to start training. If you don't need to retrain, you can just directly compile and flash the C code.
* ai8x-training and synthesis can be cloned and run on an Ubuntu server. The synthed_net folder can be then copied to a Mac in order to debug locally.
* Follow this [guide](https://analog-devices-msdk.github.io/msdk/USERGUIDE/#visual-studio-code) on MSDK with Visual Studio Code, to find out how to debug on Mac.
* the loss function can only be chosen via an argument of training.py (--regression	for regression (MSE), otherwise default is classification)