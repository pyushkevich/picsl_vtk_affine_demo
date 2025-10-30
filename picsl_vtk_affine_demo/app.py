"""
Original Code: https://github.com/Kitware/trame/blob/master/examples/06_vtk/Applications/MultiFilter/app.py

Installation requirements:
    pip install trame trame-vuetify trame-vtk trame-components
"""

import os
import numpy as np
from scipy.linalg import logm

from trame.app import get_server
from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.widgets import vuetify, trame, vtk as vtk_widgets

from vtkmodules.vtkCommonDataModel import vtkDataObject
from vtk import vtkPolyData, vtkPoints, vtkCellArray
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.vtkRenderingAnnotation import vtkCubeAxesActor
from vtkmodules.vtkFiltersSources import vtkArrowSource

from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersGeneral import vtkTransformFilter

from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkDataSetMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkPolyDataMapper
)

from importlib import resources

# Required for interactor initialization
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa

# Required for rendering initialization, not necessary for
# local rendering, but doesn't hurt to include it
import vtkmodules.vtkRenderingOpenGL2  # noqa

from trame_vtk.modules.vtk.serializers import configure_serializer

# Configure scene encoder
configure_serializer(encode_lut=True, skip_light=True)

CURRENT_DIRECTORY = os.path.abspath(os.path.dirname(__file__))

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


class Representation:
    Points = 0
    Wireframe = 1
    Surface = 2
    SurfaceWithEdges = 3


class LookupTable:
    Rainbow = 0
    Inverted_Rainbow = 1
    Greyscale = 2
    Inverted_Greyscale = 3


# -----------------------------------------------------------------------------
# VTK pipeline
# -----------------------------------------------------------------------------

renderer = vtkRenderer()
renderWindow = vtkRenderWindow()
renderWindow.AddRenderer(renderer)

renderWindowInteractor = vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()



# Read Data
mesh_filename = resources.files("picsl_vtk_affine_demo.meshes").joinpath("disk_out_ref.vtu")

print(f'Reading mesh from: {mesh_filename}')
reader = vtkXMLUnstructuredGridReader()
reader.SetFileName(mesh_filename)
reader.Update()
print('Mesh read successfully.')
print(f'Number of points: {reader.GetOutput().GetNumberOfPoints()}')

# Extract Array/Field information
dataset_arrays = []
fields = [
    (reader.GetOutput().GetPointData(), vtkDataObject.FIELD_ASSOCIATION_POINTS),
    (reader.GetOutput().GetCellData(), vtkDataObject.FIELD_ASSOCIATION_CELLS),
]
for field in fields:
    field_arrays, association = field
    for i in range(field_arrays.GetNumberOfArrays()):
        array = field_arrays.GetArray(i)
        array_range = array.GetRange()
        dataset_arrays.append(
            {
                "text": array.GetName(),
                "value": i,
                "range": list(array_range),
                "type": association,
            }
        )
default_array = dataset_arrays[0]
default_min, default_max = default_array.get("range")

# Transform
transform = vtkTransform()
transform_filter = vtkTransformFilter()
transform_filter.SetTransform(transform)
transform_filter.SetInputConnection(reader.GetOutputPort())

# Mesh
mesh_mapper = vtkDataSetMapper()
mesh_mapper.SetInputConnection(transform_filter.GetOutputPort())
mesh_actor = vtkActor()
mesh_actor.SetMapper(mesh_mapper)
renderer.AddActor(mesh_actor)

# Mesh: Setup default representation to surface
mesh_actor.GetProperty().SetRepresentationToSurface()
mesh_actor.GetProperty().SetPointSize(1)
mesh_actor.GetProperty().EdgeVisibilityOff()

# Mesh: Apply rainbow color map
mesh_lut = mesh_mapper.GetLookupTable()
mesh_lut.SetHueRange(0.666, 0.0)
mesh_lut.SetSaturationRange(1.0, 1.0)
mesh_lut.SetValueRange(1.0, 1.0)
mesh_lut.Build()

# Mesh: Color by default array
mesh_mapper.SelectColorArray(default_array.get("text"))
mesh_mapper.GetLookupTable().SetRange(default_min, default_max)
if default_array.get("type") == vtkDataObject.FIELD_ASSOCIATION_POINTS:
    mesh_mapper.SetScalarModeToUsePointFieldData()
else:
    mesh_mapper.SetScalarModeToUseCellFieldData()
mesh_mapper.SetScalarVisibility(True)
mesh_mapper.SetUseLookupTableScalarRange(True)

# Cube Axes
cube_axes = vtkCubeAxesActor()
renderer.AddActor(cube_axes)

# Cube Axes: Boundaries, camera, and styling
cube_axes.SetBounds(mesh_actor.GetBounds())
cube_axes.SetCamera(renderer.GetActiveCamera())
cube_axes.SetXLabelFormat("%6.1f")
cube_axes.SetYLabelFormat("%6.1f")
cube_axes.SetZLabelFormat("%6.1f")
cube_axes.SetFlyModeToOuterEdges()

# Create a VTK polydata that has a single line
rotation_axis_points = vtkPoints()
rotation_axis_indices = vtkCellArray()

rotation_axis_points.InsertNextPoint( 0, 0, -20)
rotation_axis_points.InsertNextPoint( 0, 0, 20)
rotation_axis_indices.InsertNextCell(2)
rotation_axis_indices.InsertCellPoint(0)
rotation_axis_indices.InsertCellPoint(1)
rotation_axis = vtkPolyData()
rotation_axis.SetPoints(rotation_axis_points)
rotation_axis.SetLines(rotation_axis_indices)


rotation_axis_mapper = vtkPolyDataMapper()
rotation_axis_mapper.SetInputData(rotation_axis)

rotation_axis_actor = vtkActor()
rotation_axis_actor.SetMapper(rotation_axis_mapper)
renderer.AddActor(rotation_axis_actor)

renderer.ResetCamera()

# I want to set the camera so it is located somewhere along the (1,1,2) direction
# and pointing at the center of the scene, and so that the scene is properly framed.
camera = renderer.GetActiveCamera()
camera.SetPosition(0.1, -1, -0.5)
camera.SetFocalPoint(0, 0, 0)
renderer.ResetCamera()

# -----------------------------------------------------------------------------
# Affine Transform Setup
# -----------------------------------------------------------------------------
class AffineTransformExample:
    def __init__(self):
        self.euler_angles = np.zeros(3)
        self.rotation_matrix = np.eye(3)
        self.affine_matrix = np.eye(4)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])  # x,y,z,w
        self.axis_angle = np.array([0.0, 0.0, 1.0, 0.0])  # x,y,z,theta
        self.translation = np.zeros(3)
        self.scaling = np.ones(3)
        self.shear_euler_angles = np.zeros(3)
        
    def skew_symmetric(self, u):
        """
        Map a 3-vector u into a 3x3 skew-symmetric matrix S such that
        S @ v = np.cross(u, v) for any 3-vector v.
        """
        u = np.asarray(u, dtype=float).reshape(3,)
        ux, uy, uz = u
        return np.array([
            [0.0, -uz,  uy],
            [uz,  0.0, -ux],
            [-uy, ux,  0.0],
        ], dtype=float)
        
    def set_euler_from_rotation_matrix(self, R):
        self.euler_angles[0] = np.arctan2(R[2,1],R[2,2]) * 180 / np.pi
        self.euler_angles[1] = -np.arcsin(R[2,0]) * 180 / np.pi
        self.psi = np.arctan2(R[1,0],R[0,0]) * 180 / np.pi
        
    def set_quat_from_rotation_matrix(self, A):
        eps=0.001
        if A.shape != (3, 3):
            raise ValueError("Input must be a 3x3 matrix")

        if 1 + A[0, 0] + A[1, 1] + A[2, 2] > eps:
            q4 = 0.5 * np.sqrt(1 + A[0, 0] + A[1, 1] + A[2, 2])
            q1 = 0.25 * (A[2, 1] - A[1, 2]) / q4
            q2 = 0.25 * (A[0, 2] - A[2, 0]) / q4
            q3 = 0.25 * (A[1, 0] - A[0, 1]) / q4

        elif 1 + A[0, 0] - A[1, 1] - A[2, 2] > eps:
            q1 = 0.5 * np.sqrt(1 + A[0, 0] - A[1, 1] - A[2, 2])
            q2 = 0.25 * (A[0, 1] + A[1, 0]) / q1
            q3 = 0.25 * (A[0, 2] + A[2, 0]) / q1
            q4 = 0.25 * (A[2, 1] - A[1, 2]) / q1

        elif 1 + A[1, 1] - A[0, 0] - A[2, 2] > eps:
            q2 = 0.5 * np.sqrt(1 + A[1, 1] - A[0, 0] - A[2, 2])
            q3 = 0.25 * (A[1, 2] + A[2, 1]) / q2
            q1 = 0.25 * (A[0, 1] + A[1, 0]) / q2
            q4 = 0.25 * (A[0, 2] - A[2, 0]) / q2

        elif 1 + A[2, 2] - A[0, 0] - A[1, 1] > eps:
            q3 = 0.5 * np.sqrt(1 + A[2, 2] - A[0, 0] - A[1, 1])
            q2 = 0.25 * (A[1, 2] + A[2, 1]) / q3
            q4 = 0.25 * (A[1, 0] - A[0, 1]) / q3
            q1 = 0.25 * (A[0, 2] + A[2, 0]) / q3

        else:
            raise ValueError("rot2quat: bad rotation matrix!")

        self.quat = np.array([q1, q2, q3, q4], dtype=np.float32)
        
    def set_axis_angle_from_rotation_matrix(self, R):
        S = logm(R)
        v = np.array([S[2,1], S[0,2], S[1,0]], dtype=np.float32)
        norm_v = np.linalg.norm(v)
        if np.abs(norm_v) > 1e-6:
            v = v / norm_v
        else:
            v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        theta = norm_v * 180 / np.pi
        self.axis_angle = np.array([v[0], v[1], v[2], theta], dtype=np.float32)
            
    def compute_rotation_matrix_from_euler_angles(self, phi, theta, psi):
        Ax = self.rotate_around_axis(1,0,0, np.pi * phi / 180)
        Ay = self.rotate_around_axis(0,1,0, np.pi * theta / 180)
        Az = self.rotate_around_axis(0,0,1, np.pi * psi / 180)
        return Az @ Ay @ Ax
        
    def update_rotation_matrix(self, R, update_euler=True, update_quat=True, update_aa=True):
        # Set the rotation matrix
        self.rotation_matrix = R
        
        # Compute the affine matrix
        
        # Scaling diagonal matrix
        K = np.diag(self.scaling)
        
        # Shear pre-rotation matrix
        U = self.compute_rotation_matrix_from_euler_angles(
            self.shear_euler_angles[0], self.shear_euler_angles[1], self.shear_euler_angles[2])
        
        # Shearing and scaling matrix
        S = U @ K @ U.T
        
        # Affine matrix
        self.affine_matrix[:3,:3] = self.rotation_matrix @ S
        self.affine_matrix[:3,3] = self.translation
        
        # Update rotation representations
        if update_euler:
            self.set_euler_from_rotation_matrix(R)
        if update_quat:
            self.set_quat_from_rotation_matrix(R)           
        if update_aa:
            self.set_axis_angle_from_rotation_matrix(R)
            
        # Update the main transform
        transform.SetMatrix(self.affine_matrix.flatten().tolist())
        
        # Update the transform for the rotation arrow
        a = self.axis_angle[0:3] * 20
        rotation_axis_points.SetPoint(0, -a[0], -a[1], -a[2])
        rotation_axis_points.SetPoint(1, a[0], a[1], a[2])
        rotation_axis_points.Modified()
        
    def rotate_around_axis(self, ux, uy, uz, theta):
        """ 
        Rotation matrix corresponding to rotation theta around axis u
        """
        u = np.array([ux,uy,uz])
        I = np.eye(3)
        Q = np.outer(u, u)
        X = self.skew_symmetric(u)
        M = np.cos(theta) * I + np.sin(theta) * X + (1-np.cos(theta)) * Q
        return M


    def set_from_euler_angles(self, phi, theta, psi):
        self.euler_angles = np.array([phi, theta, psi], dtype=np.float32)
        M = self.compute_rotation_matrix_from_euler_angles(phi, theta, psi)
        self.update_rotation_matrix(M, update_euler=False)
        
    def set_from_quaternions(self, q):
        self.quat = q 
        qhat = self.quat[0:-1]
        w = self.quat[-1]
        R = (w**2 - np.dot(qhat,qhat)) * np.eye(3) + 2 * np.outer(qhat,qhat) + 2 * w * self.skew_symmetric(qhat)
        self.update_rotation_matrix(R, update_quat=False)
        
    def set_from_axis_angle(self, w):
        self.axis_angle = w
        theta = w[3] * np.pi / 180
        I = np.eye(3)
        Q = np.outer(w[0:3], w[0:3])
        X = self.skew_symmetric(w[0:3])
        M = np.cos(theta) * I + np.sin(theta) * X + (1-np.cos(theta)) * Q
        self.update_rotation_matrix(M, update_aa=False)
        
    def set_translation(self, t):
        self.translation = t
        self.update_rotation_matrix(self.rotation_matrix, 
                                    update_euler=False, 
                                    update_quat=False, 
                                    update_aa=False)
        
    def set_scaling(self, s):
        self.scaling = s
        self.update_rotation_matrix(self.rotation_matrix, 
                                    update_euler=False, 
                                    update_quat=False, 
                                    update_aa=False)
        
    def set_shear_euler_angles(self, phi, theta, psi):
        self.shear_euler_angles = np.array([phi, theta, psi], dtype=np.float32)
        self.update_rotation_matrix(self.rotation_matrix, 
                                    update_euler=False, 
                                    update_quat=False, 
                                    update_aa=False)
        
    def reset_rotation(self):
        self.set_from_euler_angles(0.0, 0.0, 0.0)
        
    def reset_all(self):
        self.translation = np.zeros(3)
        self.scaling = np.ones(3)
        self.euler_angles = np.zeros(3)
        self.set_from_euler_angles(0.0, 0.0, 0.0)

ate = AffineTransformExample()
ate.reset_rotation()


# -----------------------------------------------------------------------------
# Trame setup
# -----------------------------------------------------------------------------

server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

state.setdefault("active_ui", None)

for i,m in enumerate('xyz'):
    state.setdefault(f'rotation_{m}', float(ate.euler_angles[i]))
    state.setdefault(f'aa_{m}', float(ate.axis_angle[i]))
    for j,n in enumerate('xyz'):
        state.setdefault(f'rotation_matrix_{i}_{j}', f'{ate.rotation_matrix[i,j]:0.4f}')
    
for i,m in enumerate('xyzw'):
    state.setdefault(f'quat_{m}', float(ate.quat[i]))
    
for i in range(4):
    for j in range(4):
        state.setdefault(f'affine_matrix_{i}_{j}', f'{ate.affine_matrix[i,j]:0.3f}')

state.setdefault("aa_theta", float(ate.axis_angle[3]))

# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


@state.change("cube_axes_visibility")
def update_cube_axes_visibility(cube_axes_visibility, **kwargs):
    cube_axes.SetVisibility(cube_axes_visibility)
    ctrl.view_update()


# Selection Change
def actives_change(ids):
    _id = ids[0]
    if _id == "1":  # Mesh
        state.active_ui = "mesh"
    elif _id == "2":  # Euler Angles
        state.active_ui = "euler_angles"
    elif _id == "3":  # Quaternions
        state.active_ui = "quaternions"
    elif _id == "4":  # Axis-Angle
        state.active_ui = "axis_angle"
    elif _id == "5":  # Translation 
        state.active_ui = "translation"
    elif _id == "6":  # Scaling
        state.active_ui = "scaling"
    elif _id == "7":  # Shear
        state.active_ui = "shear"
    else:
        state.active_ui = "nothing"


# Visibility Change
def visibility_change(event):
    _id = event["id"]
    _visibility = event["visible"]

    if _id == "1":  # Mesh
        mesh_actor.SetVisibility(_visibility)
    ctrl.view_update()


# Representation Callbacks
def update_representation(actor, mode):
    property = actor.GetProperty()
    if mode == Representation.Points:
        property.SetRepresentationToPoints()
        property.SetPointSize(5)
        property.EdgeVisibilityOff()
    elif mode == Representation.Wireframe:
        property.SetRepresentationToWireframe()
        property.SetPointSize(1)
        property.EdgeVisibilityOff()
    elif mode == Representation.Surface:
        property.SetRepresentationToSurface()
        property.SetPointSize(1)
        property.EdgeVisibilityOff()
    elif mode == Representation.SurfaceWithEdges:
        property.SetRepresentationToSurface()
        property.SetPointSize(1)
        property.EdgeVisibilityOn()


@state.change("mesh_representation")
def update_mesh_representation(mesh_representation, **kwargs):
    update_representation(mesh_actor, mesh_representation)
    ctrl.view_update()


# Color By Callbacks
def color_by_array(actor, array):
    _min, _max = array.get("range")
    mapper = actor.GetMapper()
    mapper.SelectColorArray(array.get("text"))
    mapper.GetLookupTable().SetRange(_min, _max)
    if array.get("type") == vtkDataObject.FIELD_ASSOCIATION_POINTS:
        mesh_mapper.SetScalarModeToUsePointFieldData()
    else:
        mesh_mapper.SetScalarModeToUseCellFieldData()
    mapper.SetScalarVisibility(True)
    mapper.SetUseLookupTableScalarRange(True)


@state.change("mesh_color_array_idx")
def update_mesh_color_by_name(mesh_color_array_idx, **kwargs):
    array = dataset_arrays[mesh_color_array_idx]
    color_by_array(mesh_actor, array)
    ctrl.view_update()


# Color Map Callbacks
def use_preset(actor, preset):
    lut = actor.GetMapper().GetLookupTable()
    if preset == LookupTable.Rainbow:
        lut.SetHueRange(0.666, 0.0)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    elif preset == LookupTable.Inverted_Rainbow:
        lut.SetHueRange(0.0, 0.666)
        lut.SetSaturationRange(1.0, 1.0)
        lut.SetValueRange(1.0, 1.0)
    elif preset == LookupTable.Greyscale:
        lut.SetHueRange(0.0, 0.0)
        lut.SetSaturationRange(0.0, 0.0)
        lut.SetValueRange(0.0, 1.0)
    elif preset == LookupTable.Inverted_Greyscale:
        lut.SetHueRange(0.0, 0.666)
        lut.SetSaturationRange(0.0, 0.0)
        lut.SetValueRange(1.0, 0.0)
    lut.Build()


@state.change("mesh_color_preset")
def update_mesh_color_preset(mesh_color_preset, **kwargs):
    use_preset(mesh_actor, mesh_color_preset)
    ctrl.view_update()


# Opacity Callbacks
@state.change("mesh_opacity")
def update_mesh_opacity(mesh_opacity, **kwargs):
    mesh_actor.GetProperty().SetOpacity(mesh_opacity)
    ctrl.view_update()

# Update the rotation matrix state from the AffineTransformExample
def update_rotation_matrix_state():
    with state:
        for i in range(3):
            for j in range(3):
                state[f"rotation_matrix_{i}_{j}"] = f'{ate.rotation_matrix[i,j]:0.4f}'
        for i in range(4):
            for j in range(4):
                state[f"affine_matrix_{i}_{j}"] = f'{ate.affine_matrix[i,j]:0.3f}'
        for i, m in enumerate('xyz'):
            state[f'rotation_{m}'] = float(ate.euler_angles[i])
            state[f'aa_{m}'] = float(ate.axis_angle[i])
            state[f'tran_{m}'] = float(ate.translation[i])
            state[f'scale_{m}'] = float(ate.scaling[i])
            state[f'shear_rot_{m}'] = float(ate.shear_euler_angles[i])
        for i, m in enumerate('xyzw'):
            state[f'quat_{m}'] = float(ate.quat[i])
        state['aa_theta'] = float(ate.axis_angle[3])

# Euler Angles Callbacks
def update_euler_angle(pos, value):
    value = float(value)
    e_curr = ate.euler_angles
    if np.abs(e_curr[pos] - value) < 1e-6:
        return

    e_curr[pos] = value
    ate.set_from_euler_angles(e_curr[0], e_curr[1], e_curr[2])
    update_rotation_matrix_state()
    ctrl.view_update()

@state.change("rotation_x")
def update_rotation_x(rotation_x, **kwargs):
    update_euler_angle(0, rotation_x)

@state.change("rotation_y")
def update_rotation_y(rotation_y, **kwargs):
    update_euler_angle(1, rotation_y)

@state.change("rotation_z")
def update_rotation_z(rotation_z, **kwargs):
    update_euler_angle(2, rotation_z)
    
# Quaternion Callbacks
def update_quaternions(pos, value):
    value = float(value)
    
    # Read the current quaternion values in ate
    q_curr = ate.quat
    
    # Don't do anything if the value is unchanged
    if np.abs(q_curr[pos] - value) < 1e-6:
        return
    
    # Update the quaternion component at position pos to value, then normalize the others
    norm = np.sqrt((q_curr.dot(q_curr) - q_curr[pos]**2) / (1 - value**2))
    if norm == 0:
        dummy_val = np.sqrt((1.0 - value**2) / 3)
        q = q_curr * 0 + dummy_val
        q[pos] = value           
    else:
        q = q_curr / norm
        q[pos] = value
    
    # Update the affine transform
    ate.set_from_quaternions(q)
    
    # Update the other quaternion components (avoiding recursion)
    with state:
        for i,f in enumerate('xyzw'):
            if i != pos:
                state[f'quat_{f}'] = float(q[i])
        
    # Update rotation matrix state
    update_rotation_matrix_state()
    ctrl.view_update()

@state.change("quat_x")
def update_quat_x(quat_x, **kwargs):
    update_quaternions(0, quat_x)

@state.change("quat_y")
def update_quat_y(quat_y, **kwargs):
    update_quaternions(1, quat_y)

@state.change("quat_z")
def update_quat_z(quat_z, **kwargs):
    update_quaternions(2, quat_z)

@state.change("quat_w")
def update_quat_w(quat_w, **kwargs):
    update_quaternions(3, quat_w)
    
    
# Quaternion Callbacks
def update_axis_angle(pos, value):
    value = float(value)
    
    # Read the current quaternion values in ate
    w_curr = ate.axis_angle
    
    # Don't do anything if the value is unchanged
    if np.abs(w_curr[pos] - value) < 1e-6:
        return
    
    # The vector component must be normalized to length one
    w = w_curr.copy()
    if pos < 3:
        # Update the quaternion component at position pos to value, then normalize the others
        norm = np.sqrt((w_curr[:-1].dot(w_curr[:-1]) - w_curr[:-1][pos]**2) / (1 - value**2))
        if norm == 0:
            dummy_val = np.sqrt((1.0 - value**2) / 3)
            w[:-1] = w_curr[:-1] * 0 + dummy_val
        else:
            w[:-1] = w_curr[:-1] / norm
    w[pos] = value
    
    # Update the affine transform
    ate.set_from_axis_angle(w)
    
    # Update the other quaternion components (avoiding recursion)
    with state:
        for i,f in enumerate('xyz'):
            if i != pos:
                state[f'aa_{f}'] = float(w[i])
        if pos != 3:
            state['aa_theta'] = float(w[3])
        
    # Update rotation matrix state
    update_rotation_matrix_state()
    ctrl.view_update()


@state.change("aa_x")
def update_aa_x(aa_x, **kwargs):
    update_axis_angle(0, aa_x)

@state.change("aa_y")
def update_aa_y(aa_y, **kwargs):
    update_axis_angle(1, aa_y)
    
@state.change("aa_z")
def update_aa_z(aa_z, **kwargs):
    update_axis_angle(2, aa_z)
    
@state.change("aa_theta")
def update_aa_theta(aa_theta, **kwargs):
    update_axis_angle(3, aa_theta)
    
def update_translation(pos, value):
    value = float(value)
    t_curr = ate.translation
    if np.abs(t_curr[pos] - value) < 1e-6:
        return
    t_curr[pos] = value
    ate.set_translation(t_curr)
    update_rotation_matrix_state()
    ctrl.view_update()
    
@state.change("tran_x")
def update_translation_x(tran_x, **kwargs):
    update_translation(0, tran_x)

@state.change("tran_y")
def update_translation_y(tran_y, **kwargs):
    update_translation(1, tran_y)
    
@state.change("tran_z")
def update_translation_z(tran_z, **kwargs):
    update_translation(2, tran_z)
    
def update_scaling(pos, value):
    value = float(value)
    s_curr = ate.scaling
    if np.abs(s_curr[pos] - value) < 1e-6:
        return
    s_curr[pos] = value
    ate.set_scaling(s_curr)
    update_rotation_matrix_state()
    ctrl.view_update()
    
@state.change("scale_x")
def update_scaling_x(scale_x, **kwargs):
    update_scaling(0, scale_x)
    
@state.change("scale_y")
def update_scaling_y(scale_y, **kwargs):
    update_scaling(1, scale_y)
    
@state.change("scale_z")
def update_scaling_z(scale_z, **kwargs):
    update_scaling(2, scale_z)
    
def update_shear_rotation(pos, value):
    value = float(value)
    sh_curr = ate.shear_euler_angles
    if np.abs(sh_curr[pos] - value) < 1e-6:
        return
    sh_curr[pos] = value
    ate.set_shear_euler_angles(sh_curr[0], sh_curr[1], sh_curr[2])
    update_rotation_matrix_state()
    ctrl.view_update()
    
@state.change("shear_rot_x")
def update_shear_rotation_x(shear_rot_x, **kwargs):
    update_shear_rotation(0, shear_rot_x)
    
@state.change("shear_rot_y")
def update_shear_rotation_y(shear_rot_y, **kwargs):
    update_shear_rotation(1, shear_rot_y)
    
@state.change("shear_rot_z")
def update_shear_rotation_z(shear_rot_z, **kwargs):
    update_shear_rotation(2, shear_rot_z)
    

# -----------------------------------------------------------------------------
# GUI elements
# -----------------------------------------------------------------------------


def standard_buttons():
    vuetify.VCheckbox(
        v_model=("cube_axes_visibility", True),
        on_icon="mdi-cube-outline",
        off_icon="mdi-cube-off-outline",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    vuetify.VCheckbox(
        v_model="$vuetify.theme.dark",
        on_icon="mdi-lightbulb-off-outline",
        off_icon="mdi-lightbulb-outline",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    vuetify.VCheckbox(
        v_model=("viewMode", "local"),
        on_icon="mdi-lan-disconnect",
        off_icon="mdi-lan-connect",
        true_value="local",
        false_value="remote",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    with vuetify.VBtn(icon=True, click="$refs.view.resetCamera()"):
        vuetify.VIcon("mdi-crop-free")


def pipeline_widget():
    trame.GitTree(
        sources=(
            "pipeline",
            [
                {"id": "1", "parent": "0", "visible": 1, "name": "Mesh"},
                {"id": "2", "parent": "1", "visible": 1, "name": "Euler Angles"},
                {"id": "3", "parent": "2", "visible": 1, "name": "Quaternions"},
                {"id": "4", "parent": "3", "visible": 1, "name": "Axis-Angle"},
                {"id": "5", "parent": "4", "visible": 1, "name": "Translation"},
                {"id": "6", "parent": "5", "visible": 1, "name": "Scaling"},
                {"id": "7", "parent": "6", "visible": 1, "name": "Shear"}
            ],
        ),
        actives_change=(actives_change, "[$event]"),
        visibility_change=(visibility_change, "[$event]"),
    )


def ui_card(title, ui_name):
    with vuetify.VCard(v_show=f"active_ui == '{ui_name}'"):
        vuetify.VCardTitle(
            title,
            classes="grey lighten-1 py-1 grey--text text--darken-3",
            style="user-select: none; cursor: pointer",
            hide_details=True,
            dense=True,
        )
        content = vuetify.VCardText(classes="py-2")
    return content


def ui_card_always(title):
    with vuetify.VCard():
        vuetify.VCardTitle(
            title,
            classes="grey lighten-1 py-1 grey--text text--darken-3",
            style="user-select: none; cursor: pointer",
            hide_details=True,
            dense=True,
        )
        content = vuetify.VCardText(classes="py-2")
    return content


def mesh_card():
    with ui_card(title="Mesh", ui_name="mesh"):
        vuetify.VSelect(
            # Representation
            v_model=("mesh_representation", Representation.Surface),
            items=(
                "representations",
                [
                    {"text": "Points", "value": 0},
                    {"text": "Wireframe", "value": 1},
                    {"text": "Surface", "value": 2},
                    {"text": "SurfaceWithEdges", "value": 3},
                ],
            ),
            label="Representation",
            hide_details=True,
            dense=True,
            outlined=True,
            classes="pt-1",
        )
        with vuetify.VRow(classes="pt-2", dense=True):
            with vuetify.VCol(cols="6"):
                vuetify.VSelect(
                    # Color By
                    label="Color by",
                    v_model=("mesh_color_array_idx", 0),
                    items=("array_list", dataset_arrays),
                    hide_details=True,
                    dense=True,
                    outlined=True,
                    classes="pt-1",
                )
            with vuetify.VCol(cols="6"):
                vuetify.VSelect(
                    # Color Map
                    label="Colormap",
                    v_model=("mesh_color_preset", LookupTable.Rainbow),
                    items=(
                        "colormaps",
                        [
                            {"text": "Rainbow", "value": 0},
                            {"text": "Inv Rainbow", "value": 1},
                            {"text": "Greyscale", "value": 2},
                            {"text": "Inv Greyscale", "value": 3},
                        ],
                    ),
                    hide_details=True,
                    dense=True,
                    outlined=True,
                    classes="pt-1",
                )
        vuetify.VSlider(
            # Opacity
            v_model=("mesh_opacity", 1.0),
            min=0,
            max=1,
            step=0.1,
            label="Opacity",
            classes="mt-1",
            hide_details=True,
            dense=True,
        )

# Add a text/slider pair
def add_text_slider_pair(label, label_short, v_model, min, max, step):
    with vuetify.VRow(dense=True, classes="align-center"):
        with vuetify.VCol(cols="4"):
            vuetify.VTextField(
                v_model=v_model,
                type="number",
                label=label,
                hide_details=True,
                dense=True,
                outlined=True,
            )
        with vuetify.VCol(cols="8"):
            vuetify.VSlider(
                v_model=v_model,
                min=min,
                max=max,
                step=step,
                label=label_short,
                classes="my-1",
                hide_details=True,
                dense=True,
            )

# Add a reset button
def add_reset_button(what, trigger_name):
    with vuetify.VRow(dense=True, classes="align-center"):
        with vuetify.VCol(cols="7"):
            with vuetify.VBtn(
                f"Reset",
                color="primary",
                classes="ma-2",
                click=f"trigger('{trigger_name}')",
                # add tooltips
                v_tooltip=f"'Reset {what} parameters to default values'"
            ):
                pass
        with vuetify.VCol(cols="5"):
            with vuetify.VBtn(
                f"Reset All",
                color="primary",
                classes="ma-2",
                click=f"trigger('reset_all')"
            ):
                pass

# Add a text output for each slider showing the current value
def rotation_card():
    with ui_card(title="Euler Angles", ui_name="euler_angles"):
        add_text_slider_pair("X (deg)", "X", ("rotation_x", 0), -180, 180, 0)
        add_text_slider_pair("Y (deg)", "Y", ("rotation_y", 0), -90, 90, 0)
        add_text_slider_pair("Z (deg)", "Z", ("rotation_z", 0), -180, 180, 0)
        add_reset_button('Rotation', 'reset_rotation')

# Add a quaternion card
def quaternion_card():
    with ui_card(title="Quaternions", ui_name="quaternions"):
        add_text_slider_pair("X", "X", ("quat_x", 0), -1, 1, 0.0)
        add_text_slider_pair("Y", "Y", ("quat_y", 0), -1, 1, 0.0)
        add_text_slider_pair("Z", "Z", ("quat_z", 0), -1, 1, 0.0)
        add_text_slider_pair("W", "W", ("quat_w", 0), -1, 1, 0.0)
        add_reset_button('Rotation', 'reset_rotation')
        
# Add an axis-angle card
def axis_angle_card():
    with ui_card(title="Axis-Angle", ui_name="axis_angle"):
        add_text_slider_pair("Axis X", "X", ("aa_x", 0), -1, 1, 0.0)
        add_text_slider_pair("Axis Y", "Y", ("aa_y", 0), -1, 1, 0.0)
        add_text_slider_pair("Axis Z", "Z", ("aa_z", 0), -1, 1, 0.0)
        add_text_slider_pair("Theta (deg)", "Theta", ("aa_theta", 0), -180, 180, 0.0)
        add_reset_button('Rotation', 'reset_rotation')

# Add translation card
def translation_card():
    with ui_card(title="Translation", ui_name="translation"):
        add_text_slider_pair("delta X", "X", ("tran_x", 0), -20, 20, 0.0)
        add_text_slider_pair("delta Y", "Y", ("tran_y", 0), -20, 20, 0.0)
        add_text_slider_pair("delta Z", "Z", ("tran_z", 0), -20, 20, 0.0)
        add_reset_button('Translation', 'reset_tran')

# Add scaling card
def scaling_card():
    with ui_card(title="Scaling", ui_name="scaling"):
        add_text_slider_pair("Scale X", "X", ("scale_x", 1.0), 0.1, 5.0, 0.1)
        add_text_slider_pair("Scale Y", "Y", ("scale_y", 1.0), 0.1, 5.0, 0.1)
        add_text_slider_pair("Scale Z", "Z", ("scale_z", 1.0), 0.1, 5.0, 0.1)
        add_reset_button('Scaling', 'reset_scale')

# Add shearing euler angles card
def shear_card():
    with ui_card(title="Shearing", ui_name="shear"):
        add_text_slider_pair("X (deg)", "X", ("shear_rot_x", 0), -180, 180, 0)
        add_text_slider_pair("Y (deg)", "Y", ("shear_rot_y", 0), -90, 90, 0)
        add_text_slider_pair("Z (deg)", "Z", ("shear_rot_z", 0), -180, 180, 0)
        add_reset_button('Shearing', 'reset_shear')

def rotation_matrix_display():
    with ui_card_always(title="Rotation Matrix"):
        for i in range(3):
            with vuetify.VRow(dense=True, classes="align-center"):
                for j in range(3):
                    with vuetify.VCol(cols="4"):
                        vuetify.VTextField(
                            v_model=(f"rotation_matrix_{i}_{j}", ate.rotation_matrix[i,j]),
                            type="text",
                            precision=3,
                            label=f"R[{i+1},{j+1}]",
                            hide_details=True,
                            dense=True,
                            outlined=True,
                            readonly=True,
                        )
                        
def affine_matrix_display():
    with ui_card_always(title="Affine Matrix (Homogeneous)"):
        for i in range(4):
            with vuetify.VRow(dense=True, classes="align-center"):
                for j in range(4):
                    with vuetify.VCol(cols="3"):
                        vuetify.VTextField(
                            v_model=(f"affine_matrix_{i}_{j}", ate.affine_matrix[i,j]),
                            type="text",
                            precision=3,
                            label=f"A[{i+1},{j+1}]",
                            hide_details=True,
                            dense=True,
                            outlined=True,
                            readonly=True,
                        )
                        


@ctrl.trigger("reset_rotation")
def reset_rotation(event=None):
    ate.reset_rotation()
    update_rotation_matrix_state()
    ctrl.view_update()
    
@ctrl.trigger("reset_tran")
def reset_tran(event=None):
    ate.set_translation(np.array([0.0, 0.0, 0.0], dtype=np.float32))
    update_rotation_matrix_state()
    ctrl.view_update()
    
@ctrl.trigger("reset_scale")
def reset_scale(event=None):
    ate.set_scaling(np.array([1.0, 1.0, 1.0], dtype=np.float32))
    update_rotation_matrix_state()
    ctrl.view_update()
    
@ctrl.trigger("reset_shear")
def reset_shear(event=None):
    ate.set_shear_euler_angles(0.0, 0.0, 0.0)
    update_rotation_matrix_state()
    ctrl.view_update()

@ctrl.trigger("reset_all")
def reset_all(event=None):
    ate.reset_all()
    update_rotation_matrix_state()
    ctrl.view_update()

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("UPenn BE5370 - Affine Transformations Teaching Tool")

    with layout.toolbar:
        # toolbar components
        vuetify.VSpacer()
        vuetify.VDivider(vertical=True, classes="mx-2")
        standard_buttons()

    with layout.drawer as drawer:
        # drawer components
        drawer.width = 325
        pipeline_widget()
        vuetify.VDivider(classes="mb-2")
        mesh_card()
        rotation_card()
        quaternion_card()
        axis_angle_card()
        translation_card()
        scaling_card()
        shear_card()
        
        # add a button to reset the transform            
        vuetify.VDivider(classes="mb-2")
        rotation_matrix_display()
        affine_matrix_display()
        

    with layout.content:
        # content components
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            # view = vtk_widgets.VtkRemoteView(renderWindow, interactive_ratio=1)
            # view = vtk_widgets.VtkLocalView(renderWindow)
            view = vtk_widgets.VtkRemoteLocalView(
                renderWindow, namespace="view", mode="local", interactive_ratio=1
            )
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera
            ctrl.on_server_ready.add(view.update)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    server.start()
