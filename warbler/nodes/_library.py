"""Register the built node-group library as a Blender asset library, so its
groups show in the Add menu and Asset Browser under the Warbler catalogs."""

from pathlib import Path

import bpy

ASSET_DIR = (Path(__file__).parent.parent / "assets").resolve()
NODES_FILE = ASSET_DIR / "nodes.blend"
LIBRARY_NAME = "Warbler"


def _libraries() -> bpy.types.AssetLibraryCollection:
    preferences = bpy.context.preferences
    assert preferences is not None
    return preferences.filepaths.asset_libraries


def register() -> None:
    libraries = _libraries()
    library = libraries.get(LIBRARY_NAME)
    if library is None:
        library = libraries.new(name=LIBRARY_NAME, directory=str(ASSET_DIR))
    # An extension update or a source checkout elsewhere moves the directory
    # `path` exists at runtime but is missing from the bpy stubs
    library.path = str(ASSET_DIR)  # ty: ignore[invalid-assignment]


def unregister() -> None:
    libraries = _libraries()
    library = libraries.get(LIBRARY_NAME)
    if library is not None:
        libraries.remove(library)
