# Prints out a viewable tree of channel metadata. Useful when you need to inspect comparable setups
# for small differentces in channel ordering and animation.
class FlameChannelParser::Inspector
  
  def initialize(channels_arr)
    @branches = OH.new
    channels_arr.each {|c| cluster(c) }
  end
  
  def pretty_print(output = $stdout)
    @out = output
    print_branch(@branches, initial_indent = 0)
  end
  
  private
  
  class OH < Hash # It sucks to be Ruby 1.8-compatible sometimes.
    def initialize
      super
      @keys_in_order = []
    end
    
    def []=(k, v)
      @keys_in_order.delete(k)
      @keys_in_order << k
      super(k, v)
    end
    
    def each_pair
      @keys_in_order.each {|k| yield(k, self[k]) }
    end
  end
  
  def puts(string)
    @out.puts(string)
  end
  
  def print_branch(branch, indent)
    branch.each_pair do | k, v|
      if v.is_a?(Hash)
        puts((" " * indent) + k)
        print_branch(v, indent + 1)
      else
        puts((" " * indent) + k + " - " + channel_metadata(v))
      end
    end
  end
  
  def channel_metadata(channel)
    if channel.length.zero?
      "no animations, value %s" % [channel.base_value]
    elsif channel.length > 1
      first_key = channel[0].frame
      last_key = channel[-1].frame
      "animated, %d keys, first at %d last at %d" % [channel.length, first_key, last_key]
    else
      first_key = channel[0].frame
      "animated, 1 key at %d, value %s" % [first_key, channel[0].value]
    end
  end
  
  def cluster(channel)
    path_parts = channel.name.split('/')
    leaf_name = path_parts.pop
  
    current = @branches
    path_parts.each do | path_part |
      current[path_part] ||= OH.new
      current = current[path_part]
    end
    current[leaf_name] = channel
  end
  
end
