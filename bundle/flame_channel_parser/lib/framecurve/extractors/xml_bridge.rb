require "rexml/document"

#:nodoc:

# This is a thin wrapper around the REXML library. We want FC to use nokogiri
# or libxml in the future, without modifying it's core functionality. Therefore we will use
# a thin layer on top of REXML and then migrate it to offer various backends
class Framecurve::XMLBridge
  
  include REXML
  
  # TODO: detect for subclasses here
  def self.new
    super
  end
  
  # Return a wrapped document
  def document_from_io(io)
    REXML::Document.new(io)
  end
  
  # Get the first node matching the XPath expression
  def xpath_first(from_root_node, path)
    REXML::XPath.first(from_root_node, path)
  end
  
  # Get the text of the first node matching the XPath expression
  def xpath_first_text(from_root_node, path)
    element_text(xpath_first(from_root_node, path))
  end
  
  # Yields each XPath-satisfying element to the passed block
  def xpath_each(from_root_node, path)
    REXML::XPath.each(from_root_node, path, &Proc.new)
  end
  
  # Returns the xpath to that specific node in the document
  def xpath_of(node)
    node.xpath.to_s
  end
  
  # Get a Nth parent of the passed node
  def parent(of_node, level = 1)
    ancestor = of_node
    until level.zero?
      ancestor = ancestor.parent
      level -= 1
    end
    return ancestor
  end
  
  def parent_by_name(of_node, parent_name)
    ancestor = of_node
    until ancestor == ancestor.parent
      return ancestor if ancestor.node_name == parent_name
      ancestor == ancestor.parent
    end
    nil
  end
  
  def element_text(elem)
    elem.text.to_s
  end
  
end