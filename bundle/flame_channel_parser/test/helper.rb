require "test/unit"
require "stringio"
require "tempfile"
require "cli_test"

require File.dirname(__FILE__) + "/../lib/flame_channel_parser"

class Test::Unit::TestCase
  def assert_same_buffer(ref_buffer, actual_buffer, message = "The line should be identical")
    [ref_buffer, actual_buffer].each{|io| io.rewind }
    at_line = 0
    until ref_buffer.eof? && actual_buffer.eof?
      at_line += 1
      reference_line, output_line = ref_buffer.readline, actual_buffer.readline
      assert_equal reference_line, output_line, "Line #{at_line} - #{message}"
    end
  end
end