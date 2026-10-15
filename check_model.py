import trimesh
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "model.glb"
mesh = trimesh.load(path, force="mesh")

if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

bounds = mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
center = mesh.centroid

print("Границы модели (min):", bounds[0])
print("Границы модели (max):", bounds[1])
print("Центр модели:        ", center)
print("Размеры (X, Y, Z):   ", bounds[1] - bounds[0])