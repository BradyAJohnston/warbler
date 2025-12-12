from .simulation import Simulation
import bpy
from bpy.types import Context, Scene
from bpy.app.handlers import persistent
from . import props


class SimulationManager:
    def __init__(self):
        self.simulations: dict[str, Simulation] = {}

    @property
    def scene(self) -> Scene:
        return bpy.context.scene

    @property
    def wb_props(self) -> props.WarblerSceneProperties:
        return props.scene_properties(bpy.context)

    @property
    def sim_items(self) -> bpy.types.bpy_prop_collection_idprop:  ## type: ignore
        return self.scene.wb_sim_list  # type: ignore

    def add(self, simulation: Simulation) -> None:
        self.simulations[simulation.uuid] = simulation
        item = self.sim_items.add()
        item.name = simulation.uuid
        simulation._manager = self

    def step_simulations(self):
        for id in list(self.simulations.keys()):
            if id not in self.sim_items:
                del self.simulations[id]

        for id, sim in self.simulations.items():
            if sim.is_active:
                sim.step()

    def remove(self, index: int) -> None:
        self.sim_items.remove(index)


def get_manager(context: Context | None) -> SimulationManager:
    if context is None:
        context = bpy.context
    if not hasattr(context.scene, "SimulationManager"):
        raise RuntimeError
    return context.scene.SimulationManager  # type: ignore


def update_simulations(scene: bpy.types.Scene) -> None:
    if hasattr(scene, "SimulationManager"):
        manager: SimulationManager = scene.SimulationManager  # type: ignore
        manager.step_simulations()


@persistent
def _step_simulations(self, context: bpy.types.Context) -> None:
    update_simulations(context.scene)
