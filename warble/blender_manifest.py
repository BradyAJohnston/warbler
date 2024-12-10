schema_version = "1.0.0"

id = "warble"
version = "0.0.1"
name = "warble"
tagline = "Interactive GPU simulations"
maintainer = "Brady Johnston <brady.johnston@me.com>"
type = "add-on"
website = "https://bradyajohnston.github.io/warble"
tags = ["3D View", "Animation", "Bake", "Phsyics"]

blender_version_min = "4.2.0"
license = [
    "SPDX:GPL-3.0-or-later",
]
platforms = ["windows-x64", "linux-x64"]

# # Optional: bundle 3rd party Python modules.
# # https://docs.blender.org/manual/en/dev/advanced/extensions/python_wheels.html
# wheels = [
#   "./wheels/hexdump-3.3-py3-none-any.whl",
#   "./wheels/jsmin-3.0.1-py3-none-any.whl",
# ]

[permissions]
files = "Saving cached simulations to disk"

# # Optional: advanced build settings.
# # https://docs.blender.org/manual/en/dev/advanced/extensions/command_line_arguments.html#command-line-args-extension-build
# [build]
# # These are the default build excluded patterns.
# # You only need to edit them if you want different options.
# paths_exclude_pattern = [
#   "__pycache__/",
#   "/.git/",
#   "/*.zip",
# ]
