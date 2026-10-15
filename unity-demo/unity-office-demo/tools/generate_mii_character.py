"""Generate a smooth Mii-inspired body and articulated rectangular legs for Unity."""

import math
from pathlib import Path


OUTPUT = Path(__file__).resolve().parents[1] / "Assets" / "Resources" / "MiiCharacter.obj"
LEG_OUTPUT = OUTPUT.with_name("MiiLeg.obj")
ARM_OUTPUT = OUTPUT.with_name("MiiArm.obj")
MATERIALS = {
    "skin": (1.0, 0.76, 0.52),
    "hair": (0.25, 0.12, 0.075),
    "shirt": (0.82, 0.12, 0.12),
    "pants": (0.16, 0.20, 0.29),
    "shoes": (0.10, 0.11, 0.14),
    "ink": (0.055, 0.035, 0.025),
    "eye_white": (0.98, 0.96, 0.90),
}
meshes = {name: [] for name in MATERIALS}


def add_mesh(material, vertices, faces):
    meshes[material].append((vertices, faces))


def add_ellipsoid(material, center, radii, segments=24, rings=16):
    cx, cy, cz = center
    rx, ry, rz = radii
    vertices = []
    for row in range(rings + 1):
        latitude = -math.pi / 2 + math.pi * row / rings
        clat, slat = math.cos(latitude), math.sin(latitude)
        for col in range(segments):
            theta = 2 * math.pi * col / segments
            vertices.append((
                cx + rx * clat * math.cos(theta),
                cy + ry * slat,
                cz + rz * clat * math.sin(theta),
            ))

    faces = []
    for row in range(rings):
        for col in range(segments):
            next_col = (col + 1) % segments
            lower = row * segments + col
            lower_next = row * segments + next_col
            upper = (row + 1) * segments + col
            upper_next = (row + 1) * segments + next_col
            faces.append((lower, upper, upper_next))
            faces.append((lower, upper_next, lower_next))
    add_mesh(material, vertices, faces)


def head_surface(latitude, theta, offset=1.0):
    normalized_y = math.sin(latitude)
    jaw_taper = 0.77 + 0.23 * ((normalized_y + 1.0) * 0.5)
    width = 0.42 * math.cos(latitude) * jaw_taper * offset
    depth = 0.35 * math.cos(latitude) * offset
    return (
        width * math.cos(theta),
        1.43 + 0.50 * normalized_y,
        depth * math.sin(theta),
    )


def add_egg_head():
    segments, rings = 64, 32
    vertices = []
    for row in range(rings + 1):
        latitude = -math.pi / 2 + math.pi * row / rings
        for col in range(segments):
            theta = 2 * math.pi * col / segments
            vertices.append(head_surface(latitude, theta))

    faces = []
    for row in range(rings):
        for col in range(segments):
            next_col = (col + 1) % segments
            lower = row * segments + col
            lower_next = row * segments + next_col
            upper = (row + 1) * segments + col
            upper_next = (row + 1) * segments + next_col
            faces.append((lower, upper, upper_next))
            faces.append((lower, upper_next, lower_next))
    add_mesh("skin", vertices, faces)


def add_hair_cap():
    segments, rows = 96, 24
    vertices = []
    for col in range(segments):
        theta = 2 * math.pi * col / segments
        front_offset = (theta - 1.5 * math.pi + math.pi) % (2 * math.pi) - math.pi
        frontness = max(0.0, math.cos(front_offset))
        lower_latitude = 0.08 + 0.43 * frontness
        for bang_center in (-0.68, -0.34, 0.0, 0.34, 0.68):
            dip = 0.29 * math.exp(-((front_offset - bang_center) / 0.16) ** 2)
            lower_latitude -= dip * frontness**3
        for row in range(rows + 1):
            fraction = row / rows
            latitude = lower_latitude + (math.pi / 2 - lower_latitude) * fraction
            vertices.append(head_surface(latitude, theta, 1.025))

    faces = []
    stride = rows + 1
    for col in range(segments):
        next_col = (col + 1) % segments
        for row in range(rows):
            a = col * stride + row
            b = a + 1
            c = next_col * stride + row + 1
            d = next_col * stride + row
            faces.append((a, b, c))
            faces.append((a, c, d))
    add_mesh("hair", vertices, faces)


def add_tube(material, points, radius, sides=8):
    vertices = []
    for index, point in enumerate(points):
        previous = points[max(index - 1, 0)]
        following = points[min(index + 1, len(points) - 1)]
        tangent = normalize(subtract(following, previous))
        reference = (0.0, 1.0, 0.0) if abs(tangent[1]) < 0.9 else (1.0, 0.0, 0.0)
        axis_a = normalize(cross(tangent, reference))
        axis_b = normalize(cross(tangent, axis_a))
        for side in range(sides):
            angle = 2 * math.pi * side / sides
            offset = add(scale(axis_a, math.cos(angle) * radius),
                         scale(axis_b, math.sin(angle) * radius))
            vertices.append(add(point, offset))

    faces = []
    for row in range(len(points) - 1):
        for side in range(sides):
            next_side = (side + 1) % sides
            a = row * sides + side
            b = row * sides + next_side
            c = (row + 1) * sides + next_side
            d = (row + 1) * sides + side
            faces.append((a, b, c))
            faces.append((a, c, d))
    add_mesh(material, vertices, faces)


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(vector, value):
    return tuple(component * value for component in vector)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def normalize(vector):
    length = math.sqrt(sum(component * component for component in vector))
    return tuple(component / length for component in vector) if length else (0.0, 1.0, 0.0)


def add_box(material, center, half_size):
    cx, cy, cz = center
    hx, hy, hz = half_size
    vertices = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    faces = [
        (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
    ]
    add_mesh(material, vertices, faces)


def write_obj(output=OUTPUT, source_meshes=None):
    source_meshes = meshes if source_meshes is None else source_meshes
    output.parent.mkdir(parents=True, exist_ok=True)
    material_path = output.with_suffix(".mtl")
    with material_path.open("w", encoding="utf-8") as file:
        used_materials = {name for name, parts in source_meshes.items() if parts}
        for name, color in MATERIALS.items():
            if name not in used_materials:
                continue
            file.write(f"newmtl {name}\nKd {color[0]} {color[1]} {color[2]}\nKa 0.15 0.15 0.15\nKs 0.05 0.05 0.05\nNs 8\n\n")

    with output.open("w", encoding="utf-8") as file:
        file.write(f"mtllib {material_path.name}\no {output.stem}\ns 1\n")
        vertex_offset = 0
        for material, parts in source_meshes.items():
            if not parts:
                continue
            file.write(f"g {material}\nusemtl {material}\n")
            for vertices, faces in parts:
                for x, y, z in vertices:
                    file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                for face in faces:
                    indices = " ".join(str(vertex_offset + index + 1) for index in face)
                    file.write(f"f {indices}\n")
                vertex_offset += len(vertices)


def build_character():
    add_egg_head()
    add_hair_cap()

    # Rectangular shirt; articulated arms and legs are separate assets.
    add_box("shirt", (0.0, 0.80, 0.0), (0.23, 0.32, 0.16))

    # Solid black eyes.
    for eye_x in (-0.14, 0.14):
        add_ellipsoid("ink", (eye_x, 1.52, -0.319), (0.052, 0.074, 0.025), 20, 12)

    # Angled brows follow the face depth, and the nose is a smooth hooked line.
    add_tube("hair", [(-0.25, 1.64, -0.24), (-0.17, 1.67, -0.30),
                       (-0.06, 1.63, -0.32)], 0.024)
    add_tube("hair", [(0.25, 1.64, -0.24), (0.17, 1.67, -0.30),
                       (0.06, 1.63, -0.32)], 0.024)
    add_tube("ink", [(0.025, 1.49, -0.345), (0.045, 1.43, -0.37),
                      (0.025, 1.38, -0.365), (-0.015, 1.37, -0.35)], 0.012)
    add_tube("ink", [(-0.10, 1.29, -0.31), (0.0, 1.29, -0.34),
                      (0.10, 1.29, -0.31)], 0.012)

    write_obj()

    # Each arm includes its unchanged hand, so both move together at the shoulder.
    meshes_before_parts = {name: list(parts) for name, parts in meshes.items()}
    meshes.clear()
    meshes.update({name: [] for name in MATERIALS})
    add_box("shirt", (0.0, -0.13, 0.0), (0.10, 0.23, 0.11))
    # Keep the hand below the sleeve cuff so it remains visible in the imported mesh.
    add_ellipsoid("skin", (0.0, -0.43, 0.0), (0.075, 0.075, 0.075))
    write_obj(ARM_OUTPUT)

    # One rectangular trouser leg and shoe, imported twice under animated pivots.
    meshes.clear()
    meshes.update({name: [] for name in MATERIALS})
    add_box("pants", (0.0, -0.215, 0.0), (0.075, 0.215, 0.075))
    add_box("shoes", (0.0, -0.445, -0.045), (0.09, 0.075, 0.14))
    write_obj(LEG_OUTPUT)
    meshes.clear()
    meshes.update(meshes_before_parts)


if __name__ == "__main__":
    build_character()
