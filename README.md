## flameTimewarpML

Readme for version 0.4.4 and earlier: [Readme v0.4.4](https://github.com/talosh/flameTimewarpML/blob/main/README_v044.md)

## Table of Contents
- [Status](#status)
- [Training](#training)
- [Installation](#installation)

### Status

* Training script and Flownet4 tested in production on Linux with constant speed retime
* Flownet2 tested on MacOS Apple Silicon with Pytorch 2.3

    #### Todo for v0.4.5 dev 001

* (done) ~~Add generalization logic to training script~~
* (done) ~~Refine batch retime script~~
* Share pre-trained weights for Flownet4 to enable using it in prod with command line and limited interface.
* Optimize memory usage for Flownet4
* Add Fluidmorph and Interpolate logic to new Flame script.
* Test on Intel Mac with Pytorch 2.2.2 (Latest version avaliable for x86 Macs)

### Training

#### Finetune for specific shot or set of shots
Export as Linear ACEScg (AP1) Uncompressed OpenEXR sequence.
Export your shots so each shot are in separate folder.
Copy pre-trained model file (large or lite) to the file to train with.
If motion is fast place the whole shot or its parts with fast motion to "fast" folder.

```bash
cd {flameTimewarpML folder}
./train.sh --state_file {Path to copied state file}/MyShot001.pth --generalize 1 --lr 4e-6 --acescc 0 --onecycle 1000 {Path to shot}/{fast}/
```

Change number after "--onecycle" to set number of runs.
Use "--acescc 0" to train in Linear or to retain input colourspace, "--acescc 100" to convert all samples to Log.

#### Train your own model
```bash
cd {flameTimewarpML folder}
./train.sh --state_file {Path to MyModel}/MyModel001.pth {Path to Dataset}/
```

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


### Installation

#### Installing from package - Linux
* Download parts and run
```bash
cat miniconda.tar.part-* > miniconda.tar
tar -xvf miniconda.tar
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
sh ~/Downloads/Miniconda3-py311_24.1.2-0-Linux-x86_64.sh -bfsm -p ~/miniconda3
```

* Activate anc clone default environment into another named "appenv" 

```bash
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda create --name appenv --clone base
conda activate appenv
```

* Install dependency libraries

```bash
conda install pyqt
conda install numpy
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
