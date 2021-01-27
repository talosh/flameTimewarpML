module FlameChannelParser
  # Comnplete parser class for all Flame/Smoke versions
  class Parser
    
    # Here you can assign a logger proc or a lambda that will be call'ed with progress reports
    attr_accessor :logger_proc
    
    # Parses the setup passed in the IO. If a block is given to the method it will yield Channel
    # objects one by one instead of accumulating them into an array (useful for big setups)
    def parse(io)
      @do_logs = (@logger_proc.respond_to?(:call))
      
      channels = []
      node_name, node_type = nil, nil
      
      until io.eof?
        line = io.gets
        if line =~ NODE_TYPE_MATCHER
          node_type = $1
        elsif line =~ NODE_NAME_MATCHER
          node_name = $1
        elsif line =~ CHANNEL_MATCHER && channel_is_useful?($1)
          log("Parsing channel #{$1.inspect}")
          channel = parse_channel(io, $1, node_type, node_name)
          if block_given?
            yield(channel)
          else
            channels << channel
          end
        end
      end
      
      return channels unless block_given?
    end
    
    # This method will be called internally with information on items being processed.
    # The implementation just calls the logger_proc instance variable
    def log(message)
      @logger_proc.call(message) if @do_logs
    end
    
    # Override this method to skip some channels, this will speedup
    # your code alot
    def channel_is_useful?(channel_name)
      true
    end
    
    # Defines a number of regular expression matchers applied to the file as it is being parsed
    def key_matchers #:nodoc:
      [
        # Previously:
        
        [:frame, :to_f,  /Frame ([\-\d\.]+)/],
        [:value, :to_f,  /Value ([\-\d\.]+)/],
        [:left_slope, :to_f, /LeftSlope ([\-\d\.]+)/],
        [:right_slope, :to_f, /RightSlope ([\-\d\.]+)/],
        [:interpolation, :to_s, /Interpolation (\w+)/],
        [:break_slope, :to_s, /BreakSlope (\w+)/],
        
        # 2012 intoroduces:
        
        [:r_handle_x, :to_f, /RHandleX ([\-\d\.]+)/],
        [:l_handle_x, :to_f, /LHandleX ([\-\d\.]+)/],
        [:r_handle_y, :to_f, /RHandleY ([\-\d\.]+)/],
        [:l_handle_y, :to_f, /LHandleY ([\-\d\.]+)/],
        [:curve_mode, :to_s,  /CurveMode (\w+)/],
        [:curve_order, :to_s,  /CurveOrder (\w+)/],
      ]
    end
    base_value_matcher = /Value ([\-\d\.]+)/
    keyframe_count_matcher = /Size (\d+)/
    
    BASE_VALUE_MATCHER = /Value ([\-\d\.]+)/
    KF_COUNT_MATCHER = /Size (\d+)/
    EXTRAP_MATCHER = /Extrapolation (\w+)/
    CHANNEL_MATCHER = /Channel (.+)\n/
    NODE_TYPE_MATCHER = /Node (\w+)/
    NODE_NAME_MATCHER = /Name (\w+)/
    LITERALS = %w( linear constant natural hermite cubic bezier cycle revcycle )
    
    def parse_channel(io, channel_name, node_type, node_name)
      c = Channel.new(channel_name, node_type, node_name)
      
      indent, end_mark = nil, "ENDMARK"
      
      while line = io.gets
      
        unless indent 
          indent = line.scan(/^(\s+)/)[1]
          end_mark = "#{indent}End"
        end
      
        if line =~ KF_COUNT_MATCHER
          num_keyframes = $1.to_i
          num_keyframes.times do | idx |
            log("Extracting keyframe %d of %d" % [idx + 1, num_keyframes])
            c.push(extract_key_from(io))
          end
        elsif line =~ BASE_VALUE_MATCHER# && empty?
          c.base_value = $1.to_f
        elsif line =~ EXTRAP_MATCHER
          c.extrapolation = symbolize_literal($1)
        elsif line.strip == end_mark
          break
        end
      end
      
      return c
    end
    
    def extract_key_from(io)
      frame = nil
      end_matcher = /End/
      key = Key.new
      
      until io.eof?
        line = io.gets
        if line =~ end_matcher
          return key
        else
          key_matchers.each do | property, cast_method, pattern  |
              if line =~ pattern
                v = symbolize_literal($1.send(cast_method))
                key.send("#{property}=", v) 
              end
          end
        end
      end
      raise "Did not detect any keyframes!"
    end
    
    def symbolize_literal(v)
      LITERALS.include?(v) ? v.to_sym : v
    end
  end
end