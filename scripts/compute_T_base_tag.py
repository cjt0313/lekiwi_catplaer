"""Compute static T_base_tag from URDF and save as .npy."""

from pathlib import Path
import numpy as np
import yourdfpy

URDF_PATH = Path("/home/jim/code/VLA/LeKiwi/URDF/LeKiwi_apriltag.urdf")
OUTPUT_PATH = Path(__file__).parent.parent / "common" / "T_base_tag.npy"


def main():
    urdf = yourdfpy.URDF.load(
        str(URDF_PATH),
        build_scene_graph=True,
        load_meshes=False,
        load_collision_meshes=False,
    )
    T_base_tag = urdf.get_transform("apriltag_link", frame_from="base_plate_layer1-v5")
    print("T_base_tag:")
    print(T_base_tag)
    print(f"\nTranslation: {T_base_tag[:3, 3]}")
    print(f"det(R): {np.linalg.det(T_base_tag[:3, :3]):.6f}")
    np.save(str(OUTPUT_PATH), T_base_tag)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
