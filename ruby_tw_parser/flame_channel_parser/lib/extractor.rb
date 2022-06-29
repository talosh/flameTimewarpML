# Extracts and bakes a specific animation channel to a given buffer, one string per frame
class FlameChannelParser::Extractor
  
  DEFAULT_CHANNEL_TO_EXTRACT = "Timing/Timing"
  
  # Raised when a channel is not found in the setup file
  class ChannelNotFoundError < RuntimeError; end
  
  # Raised when you try to autodetect the length of a channel that has no keyframes
  class NoKeyframesError < RuntimeError; end
  
  # Pass the path to Flame setup here and you will get the animation curve on the object passed in
  # the :destionation option (defaults to STDOUT). The following options are accepted:
  #
  #  :destination - The object to write the output to, anything that responds to shovel (<<) will do
  #  :start_frame - From which frame the curve should be baked. Will default to the first keyframe of the curve
  #  :end_frame - Upto which frame to bake. Will default to the last keyframe of the curve
  #  :channel - Name of the channel to extract from the setup. Defaults to "Timing/Timing" (timewarp frame)
  #
  # Note that start_frame and end_frame will be converted to integers.
  # The output will look like this:
  #
  #   1  123.456
  #   2  124.567
  def self.extract(path, options = {})
    new.extract(path, options)
  end
  
  def extract(path, options)
    options = DEFAULTS.dup.merge(options)
    File.open(path) do |f|
      
      # Then parse
      channels = FlameChannelParser.parse(f)
      selected_channel = find_channel_in(channels, options[:channel])
      interpolator = FlameChannelParser::Interpolator.new(selected_channel)
      
      # Configure the range
      configure_start_and_end_frame(f, options, interpolator)
      
      # And finally...
      write_channel(interpolator, options[:destination], options[:start_frame], options[:end_frame])
    end
  end
  
  private
  
  DEFAULT_START_FRAME = 1
  DEFAULTS = {
    :destination => $stdout,
    :start_frame => nil,
    :end_frame => nil,
    :channel => DEFAULT_CHANNEL_TO_EXTRACT,
    :on_curve_limits => false
  }
  
  SETUP_END_FRAME_PATTERN = /(MaxFrames|Frames)(\s+)(\d+)/
  SETUP_START_FRAME_PATTERN = /(MinFrame)(\s+)(\d+)/
  
  def start_and_end_frame_from_curve_length(interp)
    s, e = interp.first_defined_frame.to_i, interp.last_defined_frame.to_i
    if (!s || !e)
      raise NoKeyframesError, "This channel probably has no animation so there " + 
        "is no way to automatically tell how many keyframes it has. " +
        "Please set the start and end frame explicitly."
    elsif s == e
      raise NoKeyframesError, "This channel has only one keyframe " + 
        "at frame #{s}and baking it makes no sense."
    end
    [s, e]
  end
  
  def configure_start_and_end_frame(f, options, interpolator)
    # If the settings specify last and first frame...
    if options[:on_curve_limits]
      options[:start_frame], options[:end_frame] = start_and_end_frame_from_curve_length(interpolator)
    else # Detect from the setup itself (the default)
      # First try to detect start and end frames from the known flags
      f.rewind
      detected_start, detected_end = detect_start_and_end_frame_in_io(f)
      
      options[:start_frame] = options[:start_frame] || detected_start || DEFAULT_START_FRAME
      options[:end_frame] ||= detected_end
      
      # If the setup does not contain that information retry with curve limits
      if !options[:start_frame] || !options[:end_frame]
        options[:on_curve_limits] = true
        configure_start_and_end_frame(f, options, interpolator)
      end
    end
  end
  
  
  def detect_start_and_end_frame_in_io(io)
    cur_offset, s, e = io.pos, nil, nil
    io.rewind
    while line = io.gets
      if (elements = line.scan(SETUP_START_FRAME_PATTERN)).any? 
        s = elements.flatten[-1].to_i
      elsif (elements = line.scan(SETUP_END_FRAME_PATTERN)).any? 
        e = elements.flatten[-1].to_i
        return [s, e]
      end
    end
  end
  
  def compose_channel_not_found_message(for_channel, other_channels)
    message = "Channel #{for_channel.inspect} not found in this setup (set the channel with the :channel option). Found other channels though:" 
    message << "\n"
    message += other_channels.map{|c| "\t%s\n" % c.path }.join
  end
  
  def find_channel_in(channels, channel_path)
    selected_channel = channels.find{|c| channel_path == c.path }
    unless selected_channel
      raise ChannelNotFoundError, compose_channel_not_found_message(channel_path, channels)
    end
    selected_channel
  end
  
  def write_channel(interpolator, to_io, from_frame_i, to_frame_i)
    
    if (to_frame_i - from_frame_i) == 1
      $stderr.puts "WARNING: You are extracting one animation frame. Check the length of your setup, or set the range manually"
    end
    
    (from_frame_i..to_frame_i).each do | frame |
      write_frame(to_io, frame, interpolator.sample_at(frame))
    end
  end
  
  def write_frame(to_io, frame, value)
    line = "%d\t%.5f\n" % [frame, value]
    to_io << line
  end
end