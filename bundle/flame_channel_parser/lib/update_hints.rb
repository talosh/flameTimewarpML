require "net/http"
require "open-uri"
require "timeout"
require "rexml/document"
require "rubygems" unless defined?(Gem)

module UpdateHints
  VERSION = '1.0.3'
  
  # Checks whether rubygems.org has a new version of this specific gem
  # and prints how an update can be obtained if need be.
  # Note: it swallows ANY exception to prevent network-related errors when for some
  # reason the method is run while the app is offline
  def self.version_check(gem_name, present_version_str, destination = $stderr)
    begin
      version_check_without_exception_capture(gem_name, present_version_str, destination)
    rescue Exception
    end
  end
  
  private
  
  def self.version_check_without_exception_capture(gem_name, present_version_str, destination)
    # Gem::Version was known to throw when a frozen string is passed to the constructor, see
    # https://github.com/rubygems/rubygems/commit/48f1d869510dcd325d6566df7d0147a086905380
    int_present = Gem::Version.new(present_version_str.dup)
    int_avail = Checker.new(gem_name).get_latest
    
    if int_avail > int_present
      destination << "Your version of #{gem_name} is probably out of date\n"
      destination << "(the current version is #{int_avail}, but you have #{present_version_str}).\n"
      destination << "Please consider updating (run `gem update #{gem_name}`)\n"
    end
  end
  
  class Checker #:nodoc: :all
    GEMCUTTER_URI = "http://rubygems.org/api/v1/versions/%s.xml"
    
    def initialize(gem_name)
      @gem = gem_name
    end
    
    def get_latest
      gem_versions_url = GEMCUTTER_URI % @gem
      extract_version_from_xml(open(gem_versions_url))
    end
    
    private
    
    # Stubbable (arrives via open-uri)
    def open(*any)
      super
    end
    
    # Citing http://stackoverflow.com/questions/5616933/how-do-you-create-pre-release-gems
    # Prerelease versions will have text tacked onto the version string
    #   gem.version = "1.0.0.pre"      # convention used by rubygems itself
    #    gem.version = "1.0.0.beta"   
    #    gem.version = "1.0.0.rc1"
    #    gem.version = "1.0.0.bacon"
    # Gemcutter actually provides us with a convenience element for this ("prerelease")
    def extract_version_from_xml(io)
      # doc.elements.each("*/section/item")
      detected_versions = []
      
      doc = REXML::Document.new(io)
      
      # Limit our search to prerelease=false elements versions
      doc.elements.each('//prerelease[text() = "false"]') do | prerelease_element |
        # Gem::Version was known to throw when a frozen string is passed to the constructor, see
        #  # https://github.com/rubygems/rubygems/commit/48f1d869510dcd325d6566df7d0147a086905380
        number_str = prerelease_element.parent.elements['number'].text.dup
        detected_versions << Gem::Version.new(number_str)
      end
      
      detected_versions.sort.pop
    end
  end
  
  
end