require File.expand_path(File.dirname(__FILE__)) + "/segments"

# Used to sample Flame animation curves. Pass a Channel
# object to the interpolator and you can then sample values at arbitrary
# frames.
#
#   i = Interpolator.new(parsed_channel)
#   i.value_at(245.5) # => will interpolate and return the value at frame 245.5
class FlameChannelParser::Interpolator
  include FlameChannelParser::Segments
  
  NEG_INF = (-1.0/0.0)
  POS_INF = (1.0/0.0)
  
  # The constructor will accept a ChannelBlock object and convert it internally to a number of
  # segments from which samples can be made
  def initialize(channel)
    @segments = []
    @extrap = channel.extrapolation
    
    # Edge case - channel has no anim at all
    @segments = if channel.length.zero?
      [ConstantFunction.new(channel.base_value)]
    else
      create_segments_from_channel(channel)
    end
  end
  
  # Sample the value of the animation curve at this frame
  def sample_at(frame)
    if :cycle == @extrap
      return sample_from_segments(frame_number_in_cycle(frame))
    elsif :revcycle == @extrap
      return sample_from_segments(frame_number_in_revcycle(frame))
    else
      sample_from_segments(frame)
    end
  end
  
  # Returns the first frame number that is concretely defined as a keyframe
  # after the prepolation ends
  def first_defined_frame
    first_f = @segments[0].end_frame
    return 1 if first_f == NEG_INF
    return first_f
  end
  
  # Returns the last frame number that is concretely defined as a keyframe
  # before the extrapolation starts
  def last_defined_frame
    last_f = @segments[-1].start_frame
    return 100 if last_f == POS_INF
    return last_f
  end
  
  private
  
  def create_segments_from_channel(channel)
    # First the prepolating segment
    segments = [pick_prepolation(channel.extrapolation, channel[0], channel[1])]
    
    # Then all the intermediate segments, one segment between each pair of keys
    channel[0..-2].each_with_index do | key, index |
      segments << key_pair_to_segment(key, channel[index + 1])
    end
    
    # and the extrapolator
    segments << pick_extrapolation(channel.extrapolation, channel[-2], channel[-1])
  end
  
  def frame_number_in_revcycle(frame)
    animated_across = (last_defined_frame - first_defined_frame)
    # Absolute offset from the first keyframe of the animated segment
    offset = (frame - first_defined_frame).abs
    absolute_unit = offset % animated_across
    cycles = (offset / animated_across).floor
    if (cycles % 2).zero?
      first_defined_frame + absolute_unit
    else
      last_defined_frame - absolute_unit
    end
  end
  
  def frame_number_in_cycle(frame)
    animated_across = (last_defined_frame - first_defined_frame)
    offset = (frame - first_defined_frame)
    modulo = (offset % animated_across)
    first_defined_frame + modulo
  end
  
  def sample_from_segments(at_frame)
    segment = @segments.find{|s| s.defines?(at_frame) }
    raise "No segment on this curve that can interpolate the value at #{frame}" unless segment
    segment.value_at(at_frame)
  end
  
  def pick_prepolation(extrap_symbol, first_key, second_key)
    if extrap_symbol == :linear && second_key
      if first_key.interpolation != :linear
        LinearPrepolate.new(first_key.frame, first_key.value, first_key.left_slope)
      else
        # For linear keys the tangent actually does not do anything, so we need to look a frame
        # ahead and compute the increment
        increment = (second_key.value - first_key.value) / (second_key.frame - first_key.frame)
        LinearPrepolate.new(first_key.frame, first_key.value, increment)
      end
    else
      ConstantPrepolate.new(first_key.frame, first_key.value)
    end
  end
  
  def pick_extrapolation(extrap_symbol, previous_key, last_key)
    if extrap_symbol == :linear
      if previous_key && last_key.interpolation == :linear
        # For linear keys the tangent actually does not do anything, so we need to look a frame
        # ahead and compute the increment
        increment = (last_key.value - previous_key.value) / (last_key.frame - previous_key.frame)
        LinearExtrapolate.new(last_key.frame, last_key.value, increment)
      else
        LinearExtrapolate.new(last_key.frame, last_key.value, last_key.right_slope)
      end
    else
      ConstantExtrapolate.new(last_key.frame, last_key.value)
    end
  end
  
  
  # We need both the preceding and the next key
  def key_pair_to_segment(key, next_key)
    case key.interpolation
      when :bezier
        BezierSegment.new(key.frame, next_key.frame,
          key.value, next_key.value, 
          key.r_handle_x, 
          key.r_handle_y, 
          next_key.l_handle_x, next_key.l_handle_y)
      when :natural, :hermite
        print "We're in Natural:Hermite\n"
        print "key.frame: "
        puts (key.frame).inspect
        print "next_key.frame: "
        puts (next_key.frame).inspect
        print "key.value: "
        puts (key.value).inspect
        print "next_key.value: "
        puts (next_key.value).inspect
        print "key.right_slope: "
        puts (key.right_slope).inspect
        print "next_key.left_slope: "
        puts (next_key.left_slope).inspect

        HermiteSegment.new(key.frame, next_key.frame, key.value, next_key.value, key.right_slope, next_key.left_slope)
      when :constant
        ConstantSegment.new(key.frame, next_key.frame, key.value)
      else # Linear and safe
        LinearSegment.new(key.frame, next_key.frame, key.value, next_key.value)
    end
  end
  
end

