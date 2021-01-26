# Represents a framecurve comment
class Framecurve::Comment
  include Comparable
  attr_reader :text
  
  def initialize(text)
    @text = text
  end
   
  def tuple?
    false
  end
  
  def comment?
    true
  end
  
  def to_s
    ['#', text.to_s.gsub(/\r\n?/, '')].join(' ')
  end
  
  def <=>(another)
    to_s <=> another.to_s
  end
end