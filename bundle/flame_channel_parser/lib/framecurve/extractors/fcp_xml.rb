require File.dirname(__FILE__) + "/xml_bridge"

module Framecurve
module Extractors

# Pulls all varispeed timeremaps from an FCP XML V4 file
class FCP_XML
  
  def initialize()
    @xml = Framecurve::XMLBridge.new
  end
  
  def extract(io)
    doc = @xml.document_from_io(io)
    # Find all of the parameterid elements with graphdict
    @xml.xpath_each(doc, "//parameterid") do | e |
      if @xml.element_text(e) == "graphdict"
        parameter = @xml.parent(e)
        effect = @xml.parent(e, 2)
        if @xml.xpath_first_text(effect, "name") == "Time Remap"
          $stderr.puts "Found a timewarp at %s" % parameter.xpath
          
          curve = pull_timewarp(parameter, File.basename(io.path))
          destination = compose_filename(io.path, parameter)
          
          File.open(destination, "wb") do | f |
            $stderr.puts "Writing out a framecurve to %s" % destination
            Framecurve::Serializer.new.validate_and_serialize(f, curve)
          end
        end
      end
    end
  end
  
  private
  
  def compose_filename(source_path, parameter)
    path = @xml.xpath_of(parameter).split("/")
    # Path to the parameter node is something like this
    # /xmeml/project/children/sequence/media/video/track/clipitem[1]/filter[7]/effect/parameter[6] 
    # From this we want to preserve:
    # sequence and it's name
    # track number
    # clipitem number (offset)
    relevant_nodenames = path.grep(/^(sequence|track|clipitem)/)
    # Add default indices
    relevant_nodenames.map! do | nodename_and_offset |
      if nodename_and_offset =~ /\]$/
        elems = nodename_and_offset.scan(/(\w+)(\[(\d+)\])/).flatten
        [elems.shift, elems.pop]
      else
        [nodename_and_offset, "1"]
      end
    end
    mappings = {"sequence" => "SEQ", "track"=>"V", "clipitem" => "CLIP"}
    naming = relevant_nodenames.map do | nodename, offset |
      [mappings[nodename], offset].join("")
    end.join('-')
    
    [source_path, naming, "framecurve.txt"].join('.')
  end
  
  def pull_timewarp(param, source_filename)
    clipitem = @xml.parent(param, 3)
    c = Framecurve::Curve.new
    $stderr.puts clipitem.xpath
    
    c.comment!("From FCP XML %s" % source_filename)
    c.comment!("Information from %s" % clipitem.xpath)
  #  c.comment!("Sequence start TC is %s", timecode_from(param))
    clip_item_name = @xml.xpath_first_text(clipitem, "name")
    
    # The clip in point in the edit timeline, also first frame of the TW
    in_point = @xml.xpath_first_text(clipitem, "in").to_i
    out_point = @xml.xpath_first_text(clipitem, "out").to_i
    
    c.filename = [clip_item_name, "framecurve.txt"].join('.')
    
    # Accumulate keyframes
    @xml.xpath_each(param, "keyframe") do | kf |
      write_keyframe(c, kf, in_point, out_point)
    end
    $stderr.puts("Generated a curve of #{c.length} keys")
    c
  end
  
  def timecode_from(param)
    sequence = @xml.parent_by_name(param, "sequence")
    tc = @xml.xpath_first_text(sequence, "timecode/string")
  end
  
  def write_keyframe(c, kf, in_point, out_point)
    kf_when, kf_value = @xml.xpath_first_text(kf, "when").to_i, @xml.xpath_first_text(kf, "value").to_f
    # TODO: should be a flag
    at = kf_when - in_point + 1
    value = kf_value + 1 # FCP starts clips at 0
    c.tuple!(at, value) unless (at < 1 || value < 1 || at > out_point)
  end
  
end
end
end