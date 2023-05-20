# flameTimewarpML

Machine Learning frame interpolation tool for Autodesk Flame.  

Based on arXiv2020-RIFE, original implementation: <https://github.com/hzwer/arXiv2020-RIFE>

```data
@article{huang2020rife,
  title={RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  journal={arXiv preprint arXiv:2011.06294},
  year={2020}
}
```

Flame's animation curve interpolation code from Julik Tarkhanov:
<https://github.com/guerilla-di/flame_channel_parser>

## Installation

### Single workstation / Easy install

* Download latest release from [Releases](https://github.com/talosh/flameTimewarpML/releases) page
* All you need to do is put the .py file in /opt/Autodesk/shared/python and launch/relaunch flame. The first time it will unpack/install what it needs. It will give you a prompt to let you know when it’s finished.

### Preferences location

* In case you'd need to reset preferences

```bash
~/Library/Preferences/flameTimewarpML                   # Mac
~/.config/flameTimewarpML                               # Linux
```
