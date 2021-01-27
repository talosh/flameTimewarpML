# Validates a Curve object for well-formedness and completeness.
#   v = Validator.new
#   v.parse(io_handle)
#   v.errors => []
#   v.warnings => ["Do not put cusswords in your framecurves"]
class Framecurve::Validator
  attr_reader :warnings, :errors
  
  def initialize
    @warnings, @errors, @performed = [], [], false
  end
  
  # Tells whether this validator instance has any errors
  def any_errors?
    @errors.any?
  end
  
  # Tells whether this validator instance has any warnings
  def any_warnings?
    @warnings.any?
  end
  
  # Parse and validate a file (API similar to Parser#parse)
  def parse_and_validate(path_or_io)
    begin
      validate(Framecurve::Parser.new.parse(path_or_io))
    rescue Framecurve::Malformed => e
      @errors.push(e.message)
    end
  end
  
  # Validate a passed Curve object
  def validate(curve)
    initialize # reset
    methods_matching(/^(verify|recommend)/).each do | method_name |
      method(method_name).call(curve)
    end
    @performed = true
  end
  
  # Returns true if validation has been performed and there are no warnings and no errors
  def ok?
    @performed && !any_errors? && !any_warnings?
  end
  
  private
  
  def methods_matching(pattern)
    private_methods.select { |m|  m.to_s =~ pattern }
  end
  
  def verify_at_least_one_line(curve)
    @errors.push("The framecurve did not contain any lines at all") if curve.empty?
  end
  
  def verify_at_least_one_tuple(curve)
    first_tuple = curve.find{|e| e.tuple? }
    @errors.push("The framecurve did not contain any frame correlation records") unless first_tuple
  end
  
  def verify_proper_sequencing(curve)
    tuples = curve.select{|e| e.tuple? }
    frame_numbers = tuples.map{|t| t.at }
    proper_sequence = frame_numbers.sort
    
    unless frame_numbers == proper_sequence
      @errors.push("The frame sequencing is out of order " + 
      "(expected #{proper_sequence.inspect} but got #{frame_numbers.inspect})." +
      " The framecurve spec mandates that frames are recorded sequentially") 
    end
  end
  
  def verify_no_linebreaks_in_comments(curve)
    curve.each_with_index do | r, i |
      if r.comment? && (r.text.include?("\r") || r.text.include?("\n"))
         @errors.push("The comment at line %d contains a line break" % (i + 1))
      end
    end
  end
  
  def verify_non_negative_source_and_destination_frames(curve)
    curve.each_with_index do | t, i |
      next unless t.tuple?
      
      line_no = i + 1
      if t.at < 1
        @errors.push("The line %d had it's at_frame value (%d) below 1. The spec mandates at_frame >= 1." % [line_no, t.at])
      elsif t.value < 0
        @errors.push("The line %d had a use_frame_of_source value (%.5f) below 0. The spec mandates use_frame_of_source >= 0." % [line_no, t.value])
      end
    end
  end
  
  def verify_file_naming(curve)
    return unless curve.respond_to?(:filename) && curve.filename
    unless curve.filename =~ /\.framecurve\.txt$/
      @errors.push("The framecurve file has to have the .framecurve.txt double extension, but had %s" % File.extname(curve.filename).inspect)
    end
  end
  
  def verify_no_duplicate_records(curve)
    detected_dupes = []
    curve.each do | t |
      next unless t.tuple?
      next if detected_dupes.include?(t.at)
      elements = curve.select{|e| e.tuple? && e.at == t.at }
      if elements.length > 1
        detected_dupes.push(t.at)
        @errors.push("The framecurve contains the same frame (%d) twice or more (%d times)" % [t.at, elements.length])
      end
    end
  end
  
  def recommend_proper_preamble(curve)
    unless curve[0] && curve[0].comment? && curve[0].text =~ /framecurve\.org\/specification/
      @warnings.push("It is recommended that a framecurve starts with a comment with the specification URL")
    end
  end
  
  def recommend_proper_column_headers(curve)
    line_two = curve[1]
    unless line_two && line_two.comment? && line_two.text =~ /at_frame\tuse_frame_of_source/
      @warnings.push("It is recommended for the second comment to provide a column header")
    end
  end
end