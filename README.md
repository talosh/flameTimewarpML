# flameTimewarpML
Machine Learning frame interpolation tool for Autodesk Flame.  

Based on arXiv2020-RIFE, original implementation: https://github.com/hzwer/arXiv2020-RIFE
<br />
pytorch-liteflownet optical flow implementation: https://github.com/sniklaus/pytorch-liteflownet

## Installation

* Download latest release from [Releases](https://github.com/talosh/flameTimewarpML/releases) page
* Unpack and copy included flameTimewarpML.py to /opt/Autodesk/shared/python.
* Start Flame or click FLAME->Python->Rescan Python Hooks (Ctrl+Shift+H+P). You may experience a delay before the flameTimewarpML dialog appears for the first time. Click 'Continue' and give it about a minute to unpack its files in background and check progress info in Flame console. After the job is done you'll see another dialog confirming that you can start using app. If you right-click on a clip in Desktop reels or Libraries you should see new Timewarp ML menu.
