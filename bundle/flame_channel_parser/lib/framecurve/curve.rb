# Represents a curve file with comments and frame correlation records
class Framecurve::Curve
  include Enumerable
  
  # If this curve has been generated or parsed from a file, the parser 
  # will preserve the filename here
  attr_accessor :filename
  
  def initialize(*elements)
    @elements = []
    elements.flatten.each do | e |
      @elements.push(e)
    end
  end
  
  # Iterates over all the tuples in the curve
  def each_tuple
    @elements.each do | e |
      yield(e) if e.tuple?
    end
  end
  
  # Return the tuples in this curve
  def only_tuples
    @elements.select{|e| e.tuple? }
  end
  
  # Iterates over all the elements in the curve
  def each
    @elements.each(&Proc.new)
  end
  
  # Iterates over all the comments in the curve
  def each_comment
    @elements.each do | e |
      yield(e) if e.comment?
    end
  end
  
  # Adds a comment line
  def comment!(text)
    @elements.push(Framecurve::Comment.new(text.strip))
  end
  
  # Adds a tuple
  def tuple!(at, value)
    t = Framecurve::Tuple.new(at.to_i, value.to_f)
    # Validate for sequencing
    if any_tuples?
      last_frame = only_tuples[-1].at
      if t.at <= last_frame
        raise Framecurve::Malformed, "Cannot add a frame that comes before or at the same frame as the previous one (%d after %d)" % [t.at, last_frame]
      end
    end
    
    @elements.push(t)
  end
  
  # Returns the number of lines in this curve file
  def length
    @elements.size
  end
  
  # Tells whether the curve contains any elements
  def empty?
    @elements.empty?
  end
  
  # Get a record by offset (line number 0-based)
  def [](at)
    @elements[at]
  end
  
  # Tells whether the curve has any tuples at all
  def any_tuples?
    @elements.any? {|e| e.tuple? }
  end
  
  # Returns a new curve with the same data with all the intermediate frames interpolated properly
  # and all the comments except for the preamble removed
  def to_materialized_curve
    c = self.class.new
    c.comment! "http://framecurve.org/specification-v1"
    c.comment! "at_frame\tuse_frame_of_source"
    each_defined_tuple {|t|  c.tuple!(t.at, t.value) }
    return c
  end
  
  # Yields each tuple that is defined by this framecurve in succession.
  # For example, if the curve contains tuples at (1, 123.45) and (10, 167.89)
  # this method will yield 10 times for each defined integer frame value
  def each_defined_tuple
    tuples = select{|e| e.tuple? }
    tuples.each_with_index do | tuple, idx |
      next_tuple = tuples[idx + 1]
      if next_tuple.nil?
        yield(tuple)
      else # Apply linear interpolation
        dt = next_tuple.at - tuple.at
        if dt == 1
          yield(tuple)
        else
          dy = next_tuple.value - tuple.value
          delta = dy / dt
          dt.times do | increment |
            value_inc = delta * increment
            yield(Framecurve::Tuple.new(tuple.at + increment, tuple.value + value_inc))
          end
        end
      end
    end
  end
end