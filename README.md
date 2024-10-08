## flameTimewarpML

Previous versions: [Readme v0.4.4](https://github.com/talosh/flameTimewarpML/blob/main/README_v044.md)

## Table of Contents
- [Status](#status)
- [Installation](#installation)
- [Training](#training)

### Installation

#### Installing from package - Linux
* Download parts and run
```bash
cat flameTimewarpML_v0.4.5-dev003.Linux.tar.gz.part-* > flameTimewarpML_v0.4.5-dev003.Linux.tar.gz
tar -xvf flameTimewarpML_v0.4.5-dev003.Linux.tar.gz
``` 
* Place "flameTimewarpML" folder into Flame python hooks path.

#### Installing from package - MacOS
* Unpack and run fix-xattr.command
* Place "flameTimewarpML" folder into Flame python hooks path.

### Installing and configuring python environment manually

* pre-configured miniconda environment should be placed into hidden "packages/.miniconda" folder
* the folder is hidden (starts with ".") in order to keep Flame from scanning it looking for python hooks
* pre-configured python environment usually packed with release tar file

* download Miniconda for Mac or Linux (I'm using python 3.11 for tests) from 
<https://docs.anaconda.com/free/miniconda/miniconda-other-installer-links/>

* install downloaded Miniconda python distribution, use "-p" to select install location. For example:

```bash
sh ~/Downloads/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -bfsm -p ~/miniconda3
```

* Activate anc clone default environment into another named "appenv" 

```bash
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda create --name appenv
conda activate appenv
```

* Install dependency libraries

```bash
conda install python=3.11 conda-forge::openimageio conda-forge::py-openimageio
conda install pyqt
conda install conda-pack
```

* Install pytorch. Please look up exact commands depending on OS and Cuda versions

* Linux example
```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

* MacOS example:

```bash
conda install pytorch::pytorch torchvision -c pytorch
```

* Install rest of the dependencies
```bash
pip install -r {flameTimewarpML folder}/requirements.txt
```

* Pack append environment into a portable tar file

```bash
conda pack --ignore-missing-files -n appenv
```

* Unpack environment to flameTimewarpML folder

```bash
mkdir  {flameTimewarpML folder}/packages/.miniconda/appenv/
tar xvf appenv.tar.gz -C {flameTimewarpML folder}/packages/.miniconda/appenv/
```

* Remove environment tarball

```bash
rm appenv.tar.gz
```

### Training



#### Finetune for specific shot or set of shots
Finetune option is avaliable as a menu item starting from 0.4.5 dev 003

#### Finetune using command line script

Export as Linear ACEScg (AP1) Uncompressed OpenEXR sequence.
Export your shots so each shot are in separate folder.
Copy pre-trained model file (large or lite) to the file to train with.
If motion is fast place the whole shot or its parts with fast motion to "fast" folder.

```bash
cd {flameTimewarpML folder}
./train.sh --state_file {Path to copied state file}/MyShot001.pth --generalize 1 --lr 4e-6 --acescc 0 --onecycle 1000 {Path to shot}/{fast}/
```

* Change number after "--onecycle" to set number of runs.
* Use "--acescc 0" to train in Linear or to retain input colourspace, "--acescc 100" to convert all samples to Log.
* Use "--frame_size" to modify training samples size
* Preview last 9 training patches in "{Path to shot}/preview" folder
* Use "--preview" to modify how frequently preview files are saved

#### Train your own model
```bash
cd {flameTimewarpML folder}
./train.sh --state_file {Path to MyModel}/MyModel001.pth --model flownet4_v004 --batch_size 4 {Path to Dataset}/
```

#### Batch size and learning rate
The batch size and learning rate are two crucial hyperparameters that significantly affect the training process and empirical tuning is necessary here.
When the batch size is increased, the learning rate can often be increased as well. A common heuristic is the linear scaling rule: when you multiply the batch size by a factor 
k, you can also multiply the learning rate by k. Another approach is the square root scaling rule: when you multiply the batch size by k multiply the learning rate by sqrt(k)


#### Dataset preparation
Training script will scan all the folders under a given path and will compose training samples out of folders where .exr files are found.
Only Uncompressed OpenEXR files are supported at the moment.

Export your shots so each shot are in separate folder.
There are two magic words in shot path: "fast" and "slow"
When "fast" is in path 3-frame window will be used for those shots.
When "slow" is in path 9-frame window will be used.
Default window size is 5 frames.
Put shots with fast motion in "fast" folder and shots where motion are slow and continious to "slow" folder to let the model learn more intermediae ratios.

- Scene001/
    - slow/
        - Shot001/
            - Shot001.001.exr
            - Shot001.002.exr
            - ...
    - fast/
        - Shot002
            - Shot002.001.exr
            - Shot002.002.exr
            - ...
    - normal/
        - Shot003
            - Shot003.001.exr
            - Shot003.002.exr
            - ...

#### Window size
Samples for training are 3 frames and ratio. The model is given the first and the last frame and tries to re-create middle frame at given ratio.

[TODO] - add documentation
