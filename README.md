# flameTimewarpML
Machine Learning frame interpolation tool for Autodesk Flame.  

Based on arXiv2020-RIFE, original implementation: https://github.com/hzwer/arXiv2020-RIFE
```
@article{huang2020rife,
  title={RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  journal={arXiv preprint arXiv:2011.06294},
  year={2020}
}
```
Flame's animation curve interpolation code from Julik Tarkhanov:
https://github.com/guerilla-di/flame_channel_parser

## Installation
### Single workstation / Automatic install

* Download latest release from [Releases](https://github.com/talosh/flameTimewarpML/releases) page
* All you need to do is put the .py file in /opt/Autodesk/shared/python and launch/relaunch flame. The first time it will upack/install what it needs. It will give you a prompt to let you know when it’s finished.

### Centralized / Manual Install

* Get the source from [Releases](https://github.com/talosh/flameTimewarpML/releases) page or directly from repo:
```
git clone https://github.com/talosh/flameTimewarpML.git
```
* It is possible to share Python3 code location between platforms, but Miniconda3 python environment should be placed separately for Mac and Linux.
This exmaple will use:

```
Python3 code location:
/Volumes/software/flameTimewarpML                       # Mac 
/mnt/software/flameTimewarpML                           # Linux

miniconda3 location:
/Volumes/software/miniconda3                            # Mac
/mnt/software/miniconda3                                # Linux
```

* Do not place those folders inside Flame's python hooks folder.
The only file that should be placed in Flame hooks folder is flameTimewarpML.py

* Create folders
```
mkdir -p /Volumes/software/flameTimewarpML                       # Mac
mkdir -p /Volumes/software/miniconda3                            # Mac

mkdir -p /mnt/software/flameTimewarpML                           # Linux
mkdir -p /mnt/software/miniconda3                                # Linux
```

* Copy the contents of 'bundle' folder to the code location.
```
cp -a bundle/* /Volumes/software/flameTimewarpML/       # Mac
cp -a bundle/* /mnt/software/flameTimewarpML/           # Linux
```

* Download Miniconda3 for Python 3.8 from https://docs.conda.io/en/latest/miniconda.html
<br>Shell installer is used for this example.

* Install Miniconda3 to centralized location:
```
sh Miniconda3-latest-MacOSX-x86_64.sh -bu -p /Volumes/software/miniconda3/  # Mac
sh Miniconda3-latest-Linux-x86_64.sh -bu -p /mnt/software/miniconda3/       # Linux
```
* In case you'd like to move Miniconda3 installation later you'll have to re-install it again.
Please refer to Anaconda docs: https://docs.anaconda.com/anaconda/user-guide/tasks/move-directory/

* Init installed Miniconda3 environment:
```
/Volumes/software/flameTimewarpML/init_env  /Volumes/software/miniconda3/   # Mac
/mnt/software/flameTimewarpML/init_env  /mnt/software/miniconda3/           # Linux
```
The script will open new konsole window.

* In case you do it over ssh remotely on Linux edit the line at the very end of init_env
```
 cmd_prefix = 'konsole -e /bin/bash --rcfile ' + tmp_bash_rc_file
```
```
cmd_prefix = '/bin/bash --rcfile ' + tmp_bash_rc_file
```

* To check if environment is properly initialized: there should be (base) before shell prompt. python --version should give Python 3.8.5 or greater

* From this shell prompt install dependency libraries:
```
pip3 install -r /Volumes/software/flameTimewarpML/requirements.txt          # Mac
pip3 install -r /mnt/software/flameTimewarpML/requirements.txt              # Linux
```

* It is possible to download the packages and install it without internet connection. Install Miniconda3 on the machine that is connected to internet and download dependency packages:
```
pip3 download -r bundle/requirements.txt -d packages_folder
```
then it is possible to install packages from folder:
```
pip3 install -r /Volumes/software/flameTimewarpML/requirements.txt --no-index --find-links /Volumes/software/flameTimewarpML/packages_folder
```
* Copy flameTimewarpML.py to your Flame python scripts folder

* set FLAMETWML_BUNDLE, FLAMETWML_MINICONDA environment variables to point code and miniconda locations.
```
export FLAMETWML_BUNDLE=/Volumes/software/flameTimewarpML/          # Mac
export FLAMETWML_MINICONDA=//Volumes/software/miniconda3/           # Mac

export FLAMETWML_BUNDLE=/mnt/software/flameTimewarpML/              # Linux
export FLAMETWML_MINICONDA=/mnt/software/miniconda3/                # Linux
```
<br> More granular settings per platform possible with
```
export FLAMETWML_BUNDLE_MAC=/Volumes/software/flameTimewarpML/
export FLAMETWML_BUNDLE_LINUX=/mnt/software/flameTimewarpML/
export FLAMETWML_MINICONDA_MAC=/Volumes/software/miniconda3/
export FLAMETWML_MINICONDA_LINUX=mnt/software/miniconda3/
```

* There are an option to set working folder over env variable
<br> Setting FLAMETWML_DEFAULT_WORK_FOLDER will set default folder and user will still be allowed to change it
```
export FLAMETWML_DEFAULT_WORK_FOLDER=/Volumes/projects/my_timewarps/
```

* Setting FLAMETWML_WORK_FOLDER will block user from changing it. This read every time just before the tool is launched so one can use it with other pipeline tools to change it dynamically depending on context

```
export FLAMETWML_WORK_FOLDER=/Volumes/projects/my_timewarps/
```
