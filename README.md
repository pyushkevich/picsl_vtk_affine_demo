# Affine Transformations Teaching Tool

Interactive VTK app (via Trame) to explore affine transformations on a mesh. Used as a teaching aid when teaching about different ways to represent rotations in 3D and how to decompose affine transformations via polar decomposition.

## What it does
- Visualizes a VTK mesh with cube axes and camera controls
- Applies affine transforms:
  - Rotation: Euler angles, Quaternions, Axis–Angle
  - Translation, Scaling, Shear (via Euler-like angles)
- Shows live 3x3 rotation and 4x4 affine matrices
- Mesh controls: representation, color-by array, LUT presets, opacity
- Reset buttons for each transform and global reset

## Requirements
- Python 3.9+
- numpy, scipy, vtk
- trame, trame-vuetify, trame-vtk, trame-components

Install:
```bash
python3 -m pip install picsl_vtk_affine_demo
```

## Run
```bash
python3 -m picsl_vtk_affine_demo
```
Open the URL printed in the terminal (e.g., http://localhost:8080).

## Usage
- Use the drawer to pick a transform card (Rotation, Quaternions, Axis–Angle, Translation, Scaling, Shear).
- Sliders and numeric inputs are synchronized; edits update the view and matrices.
- Use “Reset” buttons to restore defaults.