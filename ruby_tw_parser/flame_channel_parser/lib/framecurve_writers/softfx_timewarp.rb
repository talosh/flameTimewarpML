# Writes out a framecurve setup
class FlameChannelParser::FramecurveWriters::SoftfxTimewarp < FlameChannelParser::FramecurveWriters::Base
  DATETIME_FORMAT = '%a %b %d %H:%M:%S %Y'
  TIME = Time.local(2011,12,28,14,50,05)
  
  def self.extension
    '.timewarp'
  end
  
  def run_export(io)
    w = FlameChannelParser::Builder.new(io)
    
    w.timewarp_file_version "1.0"
    w.creation_date(TIME.strftime(DATETIME_FORMAT))
    w.linebreak(2)
    w.fields 0
    w.origin 1
    w.render_type false
    w.sampling_step 0
    w.interpolation 0
    w.flow_quality 0
    w.linebreak!(2)
    
    # Accumulate all the keys
    writer = KeyWriter.new
    yield(writer)
    speeds = generate_speed(writer.keys)
    
    # Compute intermediate frames and speeds
    w.animation do | anim |
      anim.channel("Speed") do | speed | 
        write_animation(speeds, speed, :constant)
      end
      anim.channel("Timing/Timing") do | c |
        write_animation(writer.keys, c, :linear)
      end
    end
    
  end
  
  private
  
  def write_animation(tuples, writer, interp = :linear)
    writer.value tuples[0][1]
    writer.extrapolation :constant
    writer.key_version 1
    writer.size tuples.length
    tuples.each_with_index do | tuple, i |
      at, value = tuple
      writer.key(i) do | k |
        k.frame at
        k.value value.to_f
        k.interpolation interp
        k.left_slope 2.4
        k.right_slope 2.4
      end
    end
  end
  
  def get_percentage(key, next_key)
    delta_y = next_key[1] - key[1]
    delta_t = next_key[0] - key[0]
    one_frame_differential = delta_y.to_f / delta_t
    percentage = one_frame_differential * 100
  end
  
  # Tricky bitch
  def generate_speed(keys)
    speeds = []
    keys.each_with_index do | key, idx |
      next_key = keys[idx + 1]
      percentage = if next_key.nil? # Last frame here!
        0.0
      else
        get_percentage(key, next_key)
      end
      speeds.push([key[0], percentage])
    end
    
    speeds
  end
end