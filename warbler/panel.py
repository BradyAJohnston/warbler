import bpy
from bpy.types import Panel, UILayout

from .geometryset import GeometrySet
from .manager import get_manager
from .ops import (
    WB_OT_AddDemo,
    WB_OT_AddProteinNetwork,
    WB_OT_AddSimulation,
    WB_OT_CompileSimulation,
    WB_OT_FreeSimulation,
    WB_OT_RemoveSimulation,
)
from .props import WarblerObjectProperties, WarblerSceneProperties

# Attributes warbler knows how to map into a simulation
_KNOWN_PARTICLE_ATTRS = {
    "position",
    "velocity",
    "mass",
    "radius",
    "pinned",
    "on_surface",
    "surface_offset",
    "affinity",
    "binding",
    "attach",
}

DOMAIN_SHORT = {
    "POINT": "pt",
    "EDGE": "edge",
    "FACE": "face",
    "CORNER": "corner",
    "CURVE": "curve",
    "INSTANCE": "inst",
}
TYPE_SHORT = {
    "FLOAT": "float",
    "INT": "int",
    "FLOAT_VECTOR": "vec3",
    "FLOAT_COLOR": "color",
    "BYTE_COLOR": "color",
    "BOOLEAN": "bool",
    "FLOAT2": "vec2",
    "INT32_2D": "int2",
    "QUATERNION": "quat",
    "FLOAT4X4": "mat4",
    "INT8": "int8",
}


def draw_geometry_info(
    layout: UILayout,
    obj: bpy.types.Object,
    context: bpy.types.Context,
    known_attrs: set[str] | None = None,
) -> None:
    """Draw a read-only summary of an object's GN-evaluated geometry.

    Uses GeometrySet so the displayed data matches exactly what warbler reads
    at compile time, one row per geometry component.
    """
    if known_attrs is None:
        known_attrs = _KNOWN_PARTICLE_ATTRS

    try:
        geo = GeometrySet(obj, context)
    except Exception:  # noqa: BLE001 — GN evaluation can fail arbitrarily; show it in the UI
        layout.label(text="Could not evaluate geometry", icon="ERROR")
        return

    box = layout.box()
    col = box.column(align=True)
    found = False
    for label, icon, attrs in (
        ("Points", "POINTCLOUD_DATA", geo.points),
        ("Curves", "CURVES_DATA", geo.curves),
        ("Mesh", "MESH_DATA", geo.mesh),
    ):
        data = attrs._data
        if data is None or len(attrs) == 0:
            continue
        found = True
        text = f"{label}: {len(attrs):,} points"
        if isinstance(data, bpy.types.Curves):
            text += f", {len(data.curves):,} curves"
        elif isinstance(data, bpy.types.Mesh):
            text += f", {len(data.polygons):,} faces"
        col.label(text=text, icon=icon)
        for attr in data.attributes:
            if attr.name.startswith(".") or attr.name not in known_attrs:
                continue
            domain = DOMAIN_SHORT.get(attr.domain, attr.domain)
            dtype = TYPE_SHORT.get(attr.data_type, attr.data_type)
            col.label(text=f"  {attr.name}  ({dtype}, {domain})", icon="CHECKMARK")
    instances = geo.instances
    if len(instances):
        found = True
        col.label(
            text=f"Instances: {len(instances):,} rigid bodies",
            icon="OUTLINER_OB_GROUP_INSTANCE",
        )
        for name in instances.to_props():
            col.label(text=f"  {name}  (Instance)", icon="CHECKMARK")
    if not found:
        col.label(text="No geometry data", icon="ERROR")


def create_panel(
    layout: UILayout, idname: str | None = None, default_closed: bool = False
) -> tuple[UILayout, UILayout | None]:
    if idname is None:
        idname = "NewPanelName"
    header, panel = layout.panel(idname, default_closed=default_closed)
    return header, panel


class WB_UL_RigidBodyCollection(bpy.types.UIList):
    def draw_item(  # type: ignore
        self,
        context: bpy.types.Context,
        layout: bpy.types.UILayout,
        data,
        item: bpy.types.Object,
        icon,
        active_data,
        active_property,
        *,
        index=0,
        flt_flag=0,
    ):
        props: WarblerObjectProperties = item.wb  # type: ignore
        row = layout.row()
        row.label(text=item.name)
        row.prop(props, "is_active", text="", icon_only=True, icon="ADD")


class WB_PT_WarblerPanel(Panel):
    bl_idname = "WB_PT_WarblerPanel"
    bl_label = "Warbler"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "physics"

    def draw(self, context):
        layout = self.layout
        assert layout is not None and context is not None
        man = get_manager(context)
        sprops: WarblerSceneProperties = context.scene.wb
        layout.label(text="Simulation Settings")

        layout.prop(context.scene.render, "fps")

        layout.separator()
        layout.label(text="Active Object Settings")

        row = layout.row()

        row.template_list(
            "WB_UL_SimulationList",
            "warbler_simulations",
            sprops,
            "sim_list",
            sprops,
            "manager_active_index",
            rows=3,
        )

        col = row.column()
        col.operator(WB_OT_AddSimulation.bl_idname, text="", icon="ADD")
        col.operator(WB_OT_RemoveSimulation.bl_idname, text="", icon="REMOVE")

        try:
            item = man.active_item
        except IndexError:
            item = None

        if item is not None:
            row = layout.row()
            row.scale_y = 2
            row.operator(
                WB_OT_CompileSimulation.bl_idname,
                text="Compile" if not item.is_compiled else "Re-Compile",
            )
            if item.is_compiled:
                row.operator(WB_OT_FreeSimulation.bl_idname, icon="CANCEL")

        row = layout.row(align=True)
        row.operator_menu_enum(WB_OT_AddDemo.bl_idname, "demo", icon="NODETREE")
        row.operator(WB_OT_AddProteinNetwork.bl_idname, icon="RNA")

        if item is None:
            return

        col = layout.column()
        col.enabled = not item.is_compiled

        col.template_list(
            "WB_UL_RigidBodyCollection",
            f"{item.name}_rigid_objects",
            item.sim_rigid_collection,
            "objects",
            sprops,
            "manager_active_index",
            rows=3,
        )

        if item.is_compiled:
            full_time = item.time_sync + item.time_compute
            col.label(text=f"Simulation time: {full_time * 1e3:,.2f} ms")
            col.label(text=f"Compute:  {item.time_compute * 1e3:,.2f} ms")
            col.label(text=f"Sync:  {item.time_sync * 1e3:,.2f} ms")

        col.prop(item, "scale")

        col.prop(item, "solver")
        col.prop(item, "device")
        sub = col.row()
        sub.active = item.device == "cuda"
        sub.prop(item, "use_cuda_graph")
        col.prop(item, "substeps")
        col.prop(item, "iterations")

        header, panel = create_panel(layout, idname="source")
        header.label(text="Source")
        if panel:
            col = panel.column()
            col.prop(item, "source", text="Object")
            col.prop(item, "surface")
            if item.source is not None:
                draw_geometry_info(
                    col,
                    item.source,
                    context,
                    known_attrs=_KNOWN_PARTICLE_ATTRS,
                )

            header, panel = create_panel(panel, "spring", default_closed=True)
            if header:
                header.label(text="Springs")
            if panel:
                col = panel.column()
                col.prop(item, "spring_ke")
                col.prop(item, "spring_kd")
                col.prop(item, "bend_ke")
                col.prop(item, "bend_rest_straight")
                col.prop(item, "network_cutoff")
                col.prop(item, "domain_coupling")
                col.prop(item, "break_strain")
                col.prop(item, "reform_strain")

        header, panel = create_panel(layout, "forces", default_closed=True)
        header.label(text="Forces")
        if panel:
            col = panel.column()
            col.prop(item, "gravity")
            col.prop(item, "temperature")
            col.prop(item, "drag")
            col.separator()
            col.label(text="Binding (affinity ↔ binding sites)")
            col.prop(item, "binding_depth")
            col.prop(item, "binding_width")

        header, panel = create_panel(layout, "cloth", default_closed=True)
        header.label(text="Cloth")
        if panel:
            col = panel.column()
            col.prop(item, "cloth_tri_ke")
            col.prop(item, "cloth_tri_ka")
            col.prop(item, "cloth_tri_kd")
            col.prop(item, "cloth_edge_ke")
            col.prop(item, "cloth_edge_kd")
            sub = col.column()
            sub.active = item.solver == "VBD"
            sub.prop(item, "cloth_self_contact")
            row = sub.column()
            row.active = item.solver == "VBD" and item.cloth_self_contact
            row.prop(item, "cloth_self_contact_margin")
            row.prop(item, "cloth_self_contact_gap")

        header, panel = create_panel(layout, "contacts", default_closed=True)
        header.label(text="Contacts")
        if panel:
            col = panel.column()
            col.prop(item, "use_ground_plane")
            col.prop(item, "soft_contact_ke")
            col.prop(item, "soft_contact_kd")
            col.prop(item, "soft_contact_mu")
            sub = col.column()
            sub.active = item.solver == "VBD"
            sub.prop(item, "rigid_soft_log_barrier")
            sub.prop(item, "rigid_soft_dat")

        header, panel = create_panel(layout, idname="rigid_bodies")
        header.label(text="Rigid Bodies")
        if panel:
            col = panel.column()
            col.prop(item, "sim_rigid_collection")
            sub = col.column()
            sub.active = item.solver == "VBD"
            sub.prop(item, "attach_stiffness")
            sub.prop(item, "attach_damping")
            col.separator()
            col.label(text="Source Instances")
            col.prop(item, "instance_ke")
            col.prop(item, "instance_kd")
            col.prop(item, "instance_mu")


def move_above_physics_panels() -> None:
    """Draw the Warbler panel above Blender's own physics panels.

    ``bl_order`` can't go below 0 and panels of equal order draw in the order
    they were registered, so Blender's physics panels (registered at startup)
    are re-registered after Warbler's, parents first as listed.
    """
    panels = [
        cls
        for cls in bpy.types.Panel.__subclasses__()
        if cls.__module__.startswith("bl_ui.")
        and getattr(cls, "bl_space_type", None) == "PROPERTIES"
        and getattr(cls, "bl_context", None) == "physics"
        and cls.is_registered
    ]
    for cls in reversed(panels):
        bpy.utils.unregister_class(cls)
    for cls in panels:
        bpy.utils.register_class(cls)


CLASSES = [WB_PT_WarblerPanel, WB_UL_RigidBodyCollection]
