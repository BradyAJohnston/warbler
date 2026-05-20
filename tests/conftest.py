from pathlib import Path
import sys

import bpy
import pytest


import warbler
from warbler import manager as mgr
from warbler.simulation import SimulatorXPBD
from warbler.manager import get_manager

CURRENT = Path(__file__).parent
PROJECT = CURRENT.parent
BLEND_DIR = CURRENT / "blend_files"

sys.path.insert(0, str(PROJECT))

warbler.register()


@pytest.fixture(autouse=True)
def clean_and_save(request):
    """Load a clean homefile before each test; save a .blend for inspection after."""
    bpy.ops.wm.read_homefile(app_template="")

    # Fresh SimulationManager so each test starts with no simulations
    bpy.types.Scene.SimulationManager = mgr.SimulationManager()

    yield

    BLEND_DIR.mkdir(exist_ok=True)
    safe_name = request.node.name.replace("/", "-").replace("[", "_").replace("]", "")
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_DIR / f"{safe_name}.blend"))


@pytest.fixture
def rigid_simulation():
    """Compiled CPU simulation: one active cube rigid body above the ground plane."""
    cube = bpy.data.objects["Cube"]
    cube.location = (0, 0, 5)
    cube.wb.is_active = True
    cube.wb.sim_shape = "CUBE"

    coll = bpy.data.collections.new("Rigid")
    bpy.context.scene.collection.children.link(coll)
    coll.objects.link(cube)

    man = get_manager(bpy.context)
    man.add(SimulatorXPBD())
    item = man.active_item
    item.sim_rigid_collection = coll
    item.device = "cpu"
    item.use_ground_plane = True
    item.substeps = 3

    man.active_simulation.compile()
    return man.active_simulation
