# warbler

An add-on to bring interactive GPU simuilations via NVIDIA's [warp](https://github.com/NVIDIA/warp) to Blender's viewport.

> [!CAUTION]
> `warbler` is currently only compatible with NVIDIA GPUs. If you machine doesn't have one it will NOT work.

## Usage

Currently only adds the operator `Start Simulation` (accessible through the `F3` menu) to create a cube of particles that then simulate and interact with the default cube being moved around the scene as below.

https://github.com/user-attachments/assets/271fc06d-f851-4935-807a-313d8bdfa687

## Installation

Clone the add-on, then run the build script through Blender or a compatible python that is `3.11`. This will create a `warbler_X.X.X.zip` that can be installed through Blender's regular add-on installation.

```bash
git clone git@github.com:BradyAJohnston/warbler.git
cd warbler
blender -b -P build.py
```
