"""
This script is meant to run from within blender to 
automatically generate the dataset

20 unique meshes, contained in a 'meshes' collection,
assigned the spacial normal-visualizing material(so that a normal map can be rendered)
the rendering engine is EeVee(since no realistic lighting is required, only normals)

for mesh in meshes:
    generate 10 randomly sampled point clouds
    for cloud in point_clouds:
        generate 400 randomly sampled 3D rotations
        for rotation in random_rotations:
            render image EeVee, using applied rotation 


scene structure:
collection "meshes": contains the 20 training meshes
collection "main": contains camera and currently rendered object

directory structure:
./dataset/ parent directory
./dataset/object_name, a sub directory for every one of the meshes
./dataset/object_name/clouds.npy , a file containing 10 point clouds, 5k points each
./dataset/object_name/point_cloud_1, 10 subdirectories, one for each for each point cloud, 
    containing numbered, rendered normal maps in PNG format
./dataset/object_name/point_cloud_1/rotations.txt , a text file containig the euler rotation for each normal map
"""
import bpy
import time
import random
import numpy as np
import pathlib
import math
import bmesh

POINTS = 5000
BASE_FOLDER = '/hdd/Datasets/normals/'
CLOUD_RESEEDS = 10
ROTATIONS_PER_MESH = 100

def render_and_save_image(path):    
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    #bpy.context.scene.camera = bpy.context.active_object
    bpy.context.scene.render.resolution_x = 960
    bpy.context.scene.render.resolution_y = 540
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)

# Function to generate random rotations (Euler angles in degrees)
def generate_random_rotation():
    # Define the range for each rotation axis in degrees
    min_x, max_x = -180, 180  # Rotation around the X-axis
    min_y, max_y = -180, 180  # Rotation around the Y-axis
    min_z, max_z = -180, 180  # Rotation around the Z-axis
    
    # Generate random angles for each axis
    rotation_x = math.radians(random.uniform(min_x, max_x))
    rotation_y = math.radians(random.uniform(min_y, max_y))
    rotation_z = math.radians(random.uniform(min_z, max_z))
    
    return (rotation_x, rotation_y, rotation_z)

def sample_triangular_face(face):
    u = random.random()
    v = random.random()

    # Ensure that the sum of u and v doesn't exceed 1
    if u + v > 1:
        u = 1 - u
        v = 1 - v

    w = 1 - u - v

    # Calculate the point's coordinates using barycentric interpolation
    sampled_point = (face.verts[0].co * u +
                    face.verts[1].co * v +
                    face.verts[2].co * w)
    return sampled_point

def sample_mesh(obj,bm,probs):
    """
    samples some points from a mesh
    """
    bpy.context.view_layer.update()
    mat = obj.matrix_world # matrix that moves the particle position from object coordinates to world coordinates
    clouds = []
    camera_location = np.array(list(bpy.data.objects["Camera"].location))
    selected_face_indices = np.random.choice(len(probs),size=POINTS,replace=True,p=probs)

    for i in range(CLOUD_RESEEDS): 
        points = np.array([list(mat @ sample_triangular_face(bm.faces[j])) for j in selected_face_indices])
        points = points - camera_location
        clouds.append(points)
    return clouds

def main(collection_name):
    main_collection = bpy.data.collections[collection_name]
    mesh_collection = bpy.data.collections['meshes']
    base_path = pathlib.Path(BASE_FOLDER).resolve()

    objects = [obj for obj in mesh_collection.objects.values() if obj.type == 'MESH']
    for obj in objects:
        mesh_path = base_path.joinpath(obj.name)
        mesh_path.mkdir()

        prev_position = obj.location.copy()
        prev_rotation = obj.rotation_euler.copy()

        # create bmesh instance
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        areas = np.array([face.calc_area() for face in bm.faces])
        probs = areas / np.sum(areas)

        mesh_collection.objects.unlink(obj)
        main_collection.objects.link(obj)
        
        if bpy.context.active_object != obj: # if not selected
            bpy.context.view_layer.objects.active = obj
        rotations = [generate_random_rotation() for _ in range(ROTATIONS_PER_MESH)]
        rot_file_path = mesh_path.joinpath("rotations.txt")
        with rot_file_path.open('w') as f:
            f.write('\n'.join([f'{rotation[0]},{rotation[1]},{rotation[2]}' for rotation in rotations]))

        for rot_index,rotation in enumerate(rotations):
            obj.rotation_euler = rotation
            rot_path = mesh_path.joinpath(f"rotation_{rot_index}")
            rot_path.mkdir()
            arrays = sample_mesh(obj,bm,probs) # create 10 point clouds as np arrays of shape (5000,3)
            img_path = rot_path.joinpath("normals.png")
            render_and_save_image(str(img_path))
            for arr_index,arr in enumerate(arrays):
                pth = rot_path.joinpath(f"cloud_{arr_index}")
                np.save(str(pth),arr)
        
        bm.free()

        # return to original collection and transform
        main_collection.objects.unlink(obj)
        mesh_collection.objects.link(obj)

        obj.location = prev_position
        obj.rotation_euler = prev_rotation


def clear_temp(): # clear the temp collection
    temp_col = bpy.data.collections['temp']
    for obj in temp_col.objects.values():
        temp_col.objects.remove(obj)

def cloud_debug():
    obj = bpy.context.active_object
    global CLOUD_RESEEDS
    global POINTS
    CLOUD_RESEEDS = 1
    POINTS = 500
    
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    areas = np.array([face.calc_area() for face in bm.faces])
    probs = areas / np.sum(areas)

    cloud = sample_mesh(obj,bm,probs)[0]
    bm.free()

    visualize_cloud(cloud,0.01)

def visualize_cloud(cloud,scale):
    temp_col = bpy.data.collections['temp']
    for point in cloud:
        bpy.ops.mesh.primitive_cube_add(location=point,scale=(scale,scale,scale))



#main('main')
