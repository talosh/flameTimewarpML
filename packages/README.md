## flameTimewarpML

* pre-configured miniconda environment should be placed into hidden "packages/.miniconda" folder
* folder is hidden (starts with .) in order to keep Flame from scanning it looking for python hooks

### Installation

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
