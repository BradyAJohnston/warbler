from .warbler import Warbler
import bpy
from bpy.app.handlers import persistent


class SimulationManager:
    def __init__(self):
        self.simulation: Warbler | None = None

    def step_simulations(self):
        if self.simulation is not None:
            self.simulation.step()


@persistent
def update_simulations(scene: bpy.types.Scene) -> None:
    manager: SimulationManager = scene.SimulationManager
    manager.step_simulations()


def _step_simulations(self, context: bpy.types.Context) -> None:
    update_simulations(context.scene)
