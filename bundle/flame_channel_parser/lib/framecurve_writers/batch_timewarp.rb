# Writes out a Batch timewarp node setup
class FlameChannelParser::FramecurveWriters::BatchTimewarp < FlameChannelParser::FramecurveWriters::SoftfxTimewarp
  
  TEMPLATE = File.dirname(__FILE__) + "/templates/BatchTW.xml"
  TEMPLATE_KEY = File.dirname(__FILE__) + "/templates/key.xml"
  
  def self.extension
    '.timewarp_node'
  end
  
  def run_export(io)
    w = KeyWriter.new
    yield(w)
    keys = w.keys
    
    keys_data = ''
    keys.each_with_index do | k, idx |
      keys_data << templatize(TEMPLATE_KEY, :frame => k[0].to_i, :value => k[1].to_f, :idx => idx)
    end
    
    # Whole range BOTH in the source and destination
    used_frames = (keys.map{|k| k[1]} + keys.map{|k| k[0]}).sort
    first_frame, last_frame = used_frames[0], used_frames[-1]
    
    info = {:start_frame => first_frame, :last_frame => last_frame, :first_value => keys[0][1], :key_size => keys.size, :keys => keys_data }
    io.write(templatize(TEMPLATE, info))
  end
  
  private
  def templatize(file, hash)
    t = File.read(file)
    hash.each_pair do | pattern, value |
      p = Regexp.escape('$%s' % pattern)
      handle = Regexp.new(p, [Regexp::MULTILINE, Regexp::EXTENDED])
      t.gsub!(handle, value.to_s)
    end
    raise "Not all substitutions done" if t.include?('$')
    
    return t
  end
end