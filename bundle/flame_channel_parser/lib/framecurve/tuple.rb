# Represents one Framecurve frame correlation record
class Framecurve::Tuple
  include Comparable
  attr_reader :at, :value
  
  def initialize(at, value)
    @at, @value = at, value
  end

  def tuple?
    true
  end
  
  def comment?
    false
  end
  
  def to_s
    "%d\t%.5f" % [at, value]
  end
  
  def <=>(another)
    to_s <=> another.to_s
  end
end
