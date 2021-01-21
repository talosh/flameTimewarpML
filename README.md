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
## Installation

* Download latest release from [Releases](https://github.com/talosh/flameTimewarpML/releases) page
* Unpack and copy included flameTimewarpML.py to /opt/Autodesk/shared/python.
* Start Flame or refresh your python hooks if already started with FLAME->Python->Rescan Python Hooks (Ctrl+Shift+H+P). flameTimewarpML installation dialog should appear. Click 'Continue' and give it about a minute to unpack its files in background. Check progress info in console. After the job is done you'll see another dialog confirming that you can start using app. If you right-click on a clip in Desktop reels or Libraries you should see new Timewarp ML menu.
