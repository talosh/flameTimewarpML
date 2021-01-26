# -*- encoding: utf-8 -*-
require File.dirname(__FILE__) + '/lib/flame_channel_parser/version'
Gem::Specification.new do |s|
  s.name = "flame_channel_parser"
  s.version = FlameChannelParser::VERSION

  s.required_rubygems_version = Gem::Requirement.new(">= 0") if s.respond_to? :required_rubygems_version=
  s.authors = ["Julik Tarkhanov"]
  s.date = Time.now.utc.strftime("%Y-%m-%d")
  s.description = "Reads and interpolates animation channels in IFFS setups"
  s.email = "me@julik.nl"
  s.executables = ["bake_flame_channel", "framecurve_from_flame", "framecurve_to_flame"]
  s.extra_rdoc_files = [
    "README.md"
  ]
  s.files = `git ls-files -z`.split("\x0")
  s.homepage = "http://guerilla-di.org/flame-channel-parser/"
  s.licenses = ["MIT"]
  s.require_paths = ["lib"]
  s.rubygems_version = "1.8.11"
  s.summary = "A parser/interpolator for Flame/Smoke animation curves"

  s.specification_version = 3
  s.add_runtime_dependency("update_hints", ["~> 1.0"])
  s.add_runtime_dependency("framecurve", ["~> 2"])
  s.add_development_dependency("rake", [">= 0"])
  s.add_development_dependency("cli_test", ["~> 1.0"])
  s.add_development_dependency("test-unit")
end
