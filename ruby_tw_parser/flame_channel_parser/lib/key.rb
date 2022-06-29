# Represents a keyframe
class FlameChannelParser::Key
  
  # Frame on which the keyframe is set
  attr_accessor :frame
  
  # Value at this keyframe
  attr_accessor :value
  
  # Inteprolation type from this key onwards
  attr_accessor :interpolation
  
  # Curve order (relevant for 2012 only)
  attr_accessor :curve_order, :curve_mode
  
  # Left and right slope (will return raw slope values for pre-2012, and computed ones for 2012)
  attr_accessor :left_slope, :right_slope
  
  # Whether the tangents are broken at this keyframe
  attr_accessor :break_slope
  
  # Coordinates of the handles for 2012 setups
  attr_accessor :l_handle_x, :l_handle_y, :r_handle_x, :r_handle_y
  
  # Default value is 0.f
  def value
    @value.to_f
  end

  # Default frame is 1.0f
  def frame
    (@frame || 1).to_f
  end
  
  # Returns the RightSlope parameter of the keyframe which we use for interpolations
  def left_slope
    return right_slope unless break_slope
    
    if has_2012_tangents? # 2012 setups do not have slopes but have tangents
      dy = @value - @l_handle_y
      print "Left Slope DY: "
      puts dy.inspect
      dx = @l_handle_x.to_f - @frame
      (dy / dx  * -1)
    else
      @left_slope.to_f
    end
  end
  
  # Returns the LeftSlope parameter of the keyframe which we use for interpolations
  def right_slope
    if has_2012_tangents?
      print "@Frame: "
      puts @frame.inspect
      dy = @value - @r_handle_y
      print "Right Slope DY: "
      puts dy.inspect
      dx = @frame.to_f - @r_handle_x
      print "Right Slope DX: "
      puts dx.inspect
      print "Right Slope Value: "
      puts (dy / dx).inspect
      dy / dx
    else
      (@right_slope || nil).to_f 
    end
  end
  
  # Tells if this keyframe has 2012 tangents in it
  def has_2012_tangents?
    @has_tangents ||= !!(l_handle_x && l_handle_y)
  end
  
  # Adapter for old interpolation
  def interpolation
    # Just return the interpolation type for pre-2012 setups
    return (@interpolation || :constant) unless has_2012_tangents?
    
    return :constant if curve_order.to_s == "constant"
    return :hermite if curve_order.to_s == "cubic" && (curve_mode.to_s == "hermite" || curve_mode.to_s == "natural")
    return :bezier if curve_order.to_s == "cubic" && curve_mode.to_s == "bezier"
    return :linear if curve_order.to_s == "linear"
    
    raise "Cannot determine interpolation for #{inspect}"
  end
end