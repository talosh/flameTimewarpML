# Writes out a Curve object to the passed IO
class Framecurve::Serializer
  
  # Serialize the passed curve into io. Will use the materialized curve version.
  # Will write the file with CRLF linebreaks instead of LF.
  # Also, if the passed Curve object does not contain a preamble (URL and column headers)
  # they will be added automatically
  def serialize(io, curve)
    write_preamble(io) unless curve_has_preamble?(curve)
    curve.each do | record |
      io.write("%s\r\n" % record)
    end
  end
  
  # Serialize the passed curve into io and raise an exception
  def validate_and_serialize(io, curve)
    v = Framecurve::Validator.new
    v.validate(curve)
    raise Framecurve::Malformed, "Will not serialize a malformed curve: #{v.errors.join(', ')}" if v.any_errors?
    serialize(io, curve)
  end
  
  private
  
  def write_preamble(io)
    io.write("# http://framecurve.org/specification-v1\n")
    io.write("# at_frame\tuse_frame_of_source\n")
  end
  
  def curve_has_preamble?(curve)
    first_comment, second_comment = curve[0], curve[-1]
    return false unless first_comment && second_comment
    return false unless (first_comment.comment? && second_comment.comment?)
    return false unless first_comment.text.include?("http://framecurve.org")
    return false unless second_comment.text.include?("at_frame\tuse_frame_of_source")
    
    true
  end 
end