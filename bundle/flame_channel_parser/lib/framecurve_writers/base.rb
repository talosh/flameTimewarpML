# Writes out a framecurve setup
class FlameChannelParser::FramecurveWriters::Base
  class KeyWriter
    attr_reader :keys
    def initialize
      @keys = []
    end
    
    def key(at, value)
      @keys.push([at.to_i,value.to_f])
    end
    
  end
  
  def self.inherited(by)
    @@writers ||= []
    @@writers.push(by)
  end
  
  # Yields each defined writer class to the block
  def self.with_each_writer
    @@writers.each(&Proc.new)
  end
  
  # Should return the desired extension for the exported file
  def self.extension
    '.timewarp'
  end
  
  # Run the exporter writing the result to the passed IO. Will yield a KeyWriter
  # to the caller for writing frames (call key(at, value) on it)
  def run_export(io)
    w = KeyWriter.new
    yield(w)
    w.keys.each do | at, value |
      io.puts("%d %.5f" % [at, value])
    end
  end
  
  # Run the exporter writing the result to it, and pulling framecurve frames from the passed Curve
  # object
  def run_export_from_framecurve(io, curve)
    run_export(io) do | writer |
      curve.to_materialized_curve.each_tuple do | t |
        writer.key(t.at, t.value)
      end
    end
  end
end