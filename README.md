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
### Single workstation

* Download latest release from [Releases](https://github.com/talosh/flameTimewarpML/releases) page
* Unpack and copy included flameTimewarpML.py to /opt/Autodesk/shared/python.
* Start Flame or refresh your python hooks if already started with FLAME->Python->Rescan Python Hooks (Ctrl+Shift+H+P). flameTimewarpML installation dialog should appear. Click 'Continue' and give it about a minute to unpack its files in background. Check progress info in console. After the job is done you'll see another dialog confirming that you can start using app. If you right-click on a clip in Desktop reels or Libraries you should see new Timewarp ML menu.

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
<br>It is also possible to edit flameTimewarpML.py file to add paths directly:

```
# Configurable settings
FLAMETWML_BUNDLE_MAC = '/Volumes/software/flameTimewarpML/'
FLAMETWML_BUNDLE_LINUX = '/mnt/software/flameTimewarpML/'
FLAMETWML_MINICONDA_MAC = '/Volumes/software/miniconda3/'
FLAMETWML_MINICONDA_LINUX = '/mnt/software/miniconda3/'
menu_group_name = 'Timewarp ML'
DEBUG = False
```
