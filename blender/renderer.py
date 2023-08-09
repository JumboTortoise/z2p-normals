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

PARTICLE_COUNT = 5000
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

def make_particle_settings():
    
    particle_settings = bpy.data.particles.new("MAIN_PSETTINGS")
    particle_settings.type = 'HAIR'
    particle_settings.count = PARTICLE_COUNT  # Number of particles you want
    particle_settings.emit_from = 'FACE'  # Emit particles from the volume of the object
    particle_settings.use_emit_random = True
    particle_settings.render_type = "PATH" # set to None to not see particles
    #particle_settings.instance_object = bpy.data.objects['PARTICLE']
    return particle_settings

def sample_mesh(obj,settings):
    """
    adds a particle system to the selected object and gets their positions as a point cloud
    """
    particle_system = obj.modifiers.new(name="point_cloud_particle_system", type='PARTICLE_SYSTEM')
    particle_system.particle_system.settings = settings

    
    clouds = []
    for i in range(CLOUD_RESEEDS):
        particle_system.particle_system.seed = random.randint(0,1_000_000)
        locations = [list(particle.location) for particle in obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).particle_systems[0].particles]
        if len(locations) != PARTICLE_COUNT:
            raise AssertionError(f"only {len(locations)} particle locations were found, instead of the expected {PARTICLE_COUNT}")
        clouds.append(np.array(locations))
        #time.sleep(0.2)
    bpy.ops.object.particle_system_remove()
    return clouds

def main(collection_name):
    main_collection = bpy.data.collections[collection_name]
    mesh_collection = bpy.data.collections['meshes']
    psettings = make_particle_settings()
    base_path = pathlib.Path(BASE_FOLDER).resolve()

    objects = [obj for obj in mesh_collection.objects.values() if obj.type == 'MESH']
    for obj in objects:
        mesh_path = base_path.joinpath(obj.name)
        mesh_path.mkdir()

        prev_position = obj.location.copy()
        prev_rotation = obj.rotation_euler.copy()

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
            arrays = sample_mesh(obj,psettings) # create 10 point clouds as np arrays of shape (5000,3)
            img_path = rot_path.joinpath("normals.png")
            render_and_save_image(str(img_path))
            for arr_index,arr in enumerate(arrays):
                pth = rot_path.joinpath(f"cloud_{arr_index}")
                np.save(str(pth),arr)
        
        # return to original collection and transform
        main_collection.objects.unlink(obj)
        mesh_collection.objects.link(obj)

        obj.location = prev_position
        obj.rotation_euler = prev_rotation

def clear_particle_settings():
    objects = [obj for obj in bpy.data.objects.values() if obj.type == 'MESH']
    for obj in objects:
        if bpy.context.active_object != obj: # if not selected
            bpy.context.view_layer.objects.active = obj
        l = len(obj.particle_systems)
        for _ in range(l):
            bpy.ops.object.particle_system_remove()
    psettings = bpy.data.particles.values()
    for v in psettings:
        bpy.data.particles.remove(v)

#main('main')
clear_particle_settings()