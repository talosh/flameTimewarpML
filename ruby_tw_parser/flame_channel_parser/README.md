# flame_channel_parser

[![Build Status](https://travis-ci.org/guerilla-di/flame_channel_parser.svg?branch=master)](https://travis-ci.org/guerilla-di/flame_channel_parser)

Flame is a compositing powerhouse, and it's animation tools are extremely good for quickly prototyping things in front of clients. However, once you want to pull your
animation into other packages it can get troublesome, since only Action nodes currently support any kind of export (and it's FBX, and it's only partial).
`flame_channel_parser` alleviates the problem - it can load **any** Flame setup file (an `.action` file, or a `.stabilizer` file, or any file created by one of your
Sparks - like Kronos timewarps for examople) and extract arbitrary animation curves as tables of frames and values.

This gem includes a small library for extracting, parsing and baking animation curves made on Discrodesk Floke/Inflinto, also known as flame.
Thanks to Marijn Eken, Philippe Soeiro and Andre Gagnon for their support and advice.

## Features:

* All extrapolation and interpolation methods are supported (yes two keyframes with tangents across 2000 frames will do!)
* Expressions on channels won't be evaluated (obviously!)

## Synopsis

To examing what channels your setup contains, use the flame_channel_inspect_binary
    
    $flame_channel_inspect /usr/discreet/projects/BZO/timewarp/s02_tw.timewarp

To just bake a specific channel, use the bake_flame_channel binary.

    $bake_flame_channel --channel Timing/Timing -e 123 /usr/discreet/projects/BZO/timewarp/s02_tw.timewarp > /mnt/3d/curves/shot2_tw.framecurve.txt

If you just need to extract a framecurve file (http://framecurve.org) from a timewarp, use the framecurve_from_flame which autodetects most of the settings by itself and gives you a nice .framecurve.txt file

    $framecurve_from_flame /usr/discreet/projects/BZO/timewarp/s02_tw.timewarp

This will create a framecurve file called s02_tw.timewarp.framecurve.txt next to the setup

The reverse is true for framecurve_to_flame

    $framecurve_from_flame /usr/discreet/projects/BZO/timewarp/s02_tw.framecurve.txt

will create setup files for use with both Timewarp modules and the Kronos spark.

The bonuses are of course in the details - **flame_channel_parser** is a meticulous
bitch and will faithfully replicate all the tricky things you can do to your animation curves to give them more _oomph_ - like broken tangents, two keyframes on
500 frames with wild oscillations, looping and linear extrapolations, constant interpolations for jumps and so on. In short, this will _interpolate_ your
animation channel and extract it frame by frame, so all the animation is intact. **No keyframe baking is necessary** for the Flame operator to do.
Of course it supports both varieties of Flame setups - the newer 2012 ones as good as the old 9.0 - 2011 ones.

To use the library:
    
    require "flame_channel_parser"
    
    # Parse the setup into channels
    channels = File.open("TW_Setup.timewarp") do | f |
      FlameChannelParser.parse(f)
    end
    
    # Find the channel that we are interested in, in this case
    # this is the "Timing" channel from any Timewarp setup
    frame_channel = channels.find{|c| c.name == "Timing/Timing" }
    
    # Grab the interpolator object for this channel.
    interpolator = frame_channel.to_interpolator
    
    # Now sample from frame 20 to frame 250.
    # You can also sample at fractional frames if you want to.
    (20..250).each do | frame_in_setup |
      p interpolator.value_at(frame_in_setup)
    end
    
## Requirements

* Ruby 1.8.7 and above

## Installation

    $gem install flame_channel_parser

## License

(The MIT License)

Copyright (c) 2011-2015 Julik Tarkhanov

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
