## flameTimewarpML

* pre-configured miniconda environment should be placed into hidden "packages/.miniconda" folder
* the folder is hidden (starts with ".") in order to keep Flame from scanning it looking for python hooks
* pre-configured python environment usually packed with release tar file

### Installing and configuring python environment manually

* download Miniconda for Mac or Linux (I'm using python 3.11 for tests) from 
<https://docs.anaconda.com/free/miniconda/miniconda-other-installer-links/>

* install downloaded Miniconda python distribution, use "-p" to select install location. For example:

```bash
sh Miniconda3-py311_24.1.2-0-Linux-x86_64.sh -bfsm -p ~/miniconda3
```

* Activate anc clone default environment into another named "appenv" 

```bash
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda create --name appenv --clone base
conda activate appenv
```

* Install dependency libraries

```bash
conda install numpy
pip install PySide6
conda install pytorch::pytorch -c pytorch
conda install conda-pack
```

```sh Miniconda3-py311_24.1.2-0-MacOSX-arm64.sh
eval "$(~/miniconda3/bin/conda shell.zsh hook)"
conda create --name twml --clone base
conda install numpy
pip install PySide6
conda install pytorch::pytorch -c pytorch
conda install conda-pack
conda pack --ignore-missing-files -n twml
tar xvf twml.tar.gz -C {flameTimewarpML folder}/packages/.miniconda/twml/
```
