import bpy
from bpy.types import Panel, UILayout

from .geometryset import GeometrySet
from .manager import get_manager
from .ops import WB_OT_AddSimulation, WB_OT_CompileSimulation, WB_OT_RemoveSimulation
from .props import WarblerObjectProperties, WarblerSceneProperties

# Attributes warbler knows how to map into a simulation
_KNOWN_PARTICLE_ATTRS = {"position", "velocity", "mass", "radius", "pinned"}

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
    at compile time (GN pointcloud output takes priority over base mesh).
    """
    if known_attrs is None:
        known_attrs = _KNOWN_PARTICLE_ATTRS

    try:
        geo = GeometrySet(obj, context)
    except Exception:  # noqa: BLE001 — GN evaluation can fail arbitrarily; show it in the UI
        layout.label(text="Could not evaluate geometry", icon="ERROR")
        return

    data = geo.data
    if data is None:
        layout.label(text="No geometry data", icon="ERROR")
        return

    box = layout.box()
    col = box.column(align=True)

    if isinstance(data, bpy.types.PointCloud):
        pt_count = geo._get_point_count()
        col.label(text=f"Points: {pt_count:,}", icon="POINTCLOUD_DATA")
    elif isinstance(data, bpy.types.Mesh):
        pt_count = geo._get_point_count()
        col.label(text=f"Vertices: {pt_count:,}", icon="MESH_DATA")
        col.label(text=f"Faces: {len(data.polygons):,}")
    elif isinstance(data, bpy.types.Curves):
        col.label(
            text=f"Curves: {len(data.curves):,}  Points: {geo._get_point_count():,}",
            icon="CURVES_DATA",
        )
    else:
        col.label(text=f"Unsupported type: {geo.obj_type}", icon="QUESTION")
        return

    attrs = data.attributes
    if not attrs:
        return

    col.separator()
    col.label(text="Attributes:")
    for attr in attrs:
        if attr.name.startswith("."):
            continue
        domain = DOMAIN_SHORT.get(attr.domain, attr.domain)
        dtype = TYPE_SHORT.get(attr.data_type, attr.data_type)
        known = attr.name in known_attrs
        icon = "CHECKMARK" if known else "DOT"
        col.label(text=f"  {attr.name}  ({dtype}, {domain})", icon=icon)


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

        header, panel = create_panel(layout, idname="particles")
        header.label(text="Particles")
        if panel:
            col = panel.column()
            col.prop(item, "particle_source", text="Source")
            if item.particle_source is not None:
                draw_geometry_info(
                    col,
                    item.particle_source,
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
                col.prop(item, "spring_kf")

        header, panel = create_panel(layout, idname="rigid_bodies")
        header.label(text="Rigid Bodies")
        if panel:
            col = panel.column()
            col.prop(item, "sim_rigid_collection")
            col.prop(item, "rigid_decay_frames")
            col.prop(item, "substeps")
            col.prop(item, "iterations")
            col.prop(item, "is_active")

        row = layout.row()
        row.scale_y = 2
        row.operator(
            WB_OT_CompileSimulation.bl_idname,
            text="Compile" if not item.is_compiled else "Re-Compile",
        )


CLASSES = [WB_PT_WarblerPanel, WB_UL_RigidBodyCollection]
