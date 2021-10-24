# Writes out a framecurve setup
class FlameChannelParser::FramecurveWriters::Kronos < FlameChannelParser::FramecurveWriters::SoftfxTimewarp
  TOKEN = Regexp.new('__INSERT_FRAME_ANIM__')
  TEMPLATE = File.dirname(__FILE__) + "/templates/SampleKronos.F_Kronos"
  
  def self.extension
    '.F_Kronos'
  end
  
  def run_export(io)
    buf = StringIO.new
    w = FlameChannelParser::Builder.new(buf)
    w.channel("Frame") do | c |
      writer = KeyWriter.new
      yield(writer)
      write_animation(writer.keys, c, :linear)
    end
    
    # Entab everything
    template = File.read(TEMPLATE)
    io.write(template.gsub(TOKEN, buf.string))
  end
end