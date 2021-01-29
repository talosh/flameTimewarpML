require File.dirname(__FILE__) + '/flame_channel_parser/version'

module FlameChannelParser
  
  module FramecurveWriters; end
  
  # Parse a Flame setup into an array of Channel objects.
  # If a block is given to the method it will yield Channel
  # objects one by one instead of accumulating them into an array (useful for big setups)
  def self.parse(io)
    c = get_parser_class(io)
    if block_given?
      c.new.parse(io, &Proc.new)
    else
      c.new.parse(io)
    end
  end
  
  # Parse a Flame setup at passed path. Will return the channels instead of yielding them
  def self.parse_file_at(path)
    File.open(path, &method(:parse))
  end
  
  private
  
  # Returns the XML parser class for XML setups
  def self.get_parser_class(for_io)
    tokens = %w( <Setup> <?xml )
    current = for_io.pos
    tokens.each do | token |
      for_io.rewind
      return XMLParser if for_io.read(token.size) == token
    end
    return Parser
  ensure
    for_io.seek(current)
  end
end

%w(
  key channel parser segments interpolator extractor timewarp_extractor builder xml_parser inspector
).each {|f| require File.expand_path(File.dirname(__FILE__) + "/" + f ) }

%w(
  base softfx_timewarp batch_timewarp kronos 
).each {|f| require File.expand_path(File.dirname(__FILE__) + "/framecurve_writers/" + f ) }
