# warbler

An add-on to bring interactive GPU simuilations via NVIDIA's [warp](https://github.com/NVIDIA/warp) to Blender's viewport.

## Usage

Build components for a simulation through Geometry Nodes, then compile a simulation to run on the GPU through NVIDIA's [`newton`](https://github.com/newton-physics/newton) and [`warp`](https://github.com/NVIDIA/warp) frameworks.

![](docs/img/interactive_curves.mp4)

## Installation

Use the `uv` package manager, ensure it is installed on your system.

Inside the cloner repo, run these commands:

Download required packages and setup the `blender_manifest.toml` with this command:

```bash
uvx extbpy sync
```

Build the standalone `.zip` that is required for regular installation:

```bash
uvx extbpy build
```
