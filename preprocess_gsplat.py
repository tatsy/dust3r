import argparse
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import quaternion
import torch
import trimesh
from plyfile import PlyData, PlyElement
from tqdm.auto import tqdm

from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
from dust3r.utils.image import load_images


def compute_frustum(K, z_far=0.5):
    fx, fy = K[0, 0], K[1, 1]
    px, py = K[0, 2], K[1, 2]
    width, height = px * 2, py * 2

    corners = np.array(
        [[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32
    )
    frustum = [[0.0, 0.0, 0.0, 1.0]]
    for x, y in corners:
        xc = (x - px) * z_far / fx
        yc = (y - py) * z_far / fy
        frustum.append([xc, yc, z_far, 1.0])

    return np.array(frustum, dtype=np.float32)


def export_gltf(intrinsics, extrinsics, points, colors, glb_file: Path):
    """
    Convert a PLY file to GLB format.
    """
    scene = trimesh.Scene()

    # Point cloud
    pcd = trimesh.PointCloud(points, colors=colors)
    scene.add_geometry([pcd], geom_name="Point Cloud")

    scene_scale = np.max(np.std(points, axis=0))

    # Camera frustums
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 3, 2],
            [1, 4, 3],
        ],
        dtype=np.int32,
    )

    for idx, (K, P) in enumerate(zip(intrinsics, extrinsics)):
        v_view = compute_frustum(K, z_far=0.05 * scene_scale)
        v_world = (P @ v_view.T).T[:, :3]
        vertex_colors = np.array([255, 0, 0] * len(v_world), dtype=np.uint8).reshape(
            (-1, 3)
        )
        frustum = trimesh.Trimesh(
            vertices=v_world, faces=faces, vertex_colors=vertex_colors, process=False
        )
        scene.add_geometry([frustum], geom_name=f"Camera #{idx + 1:d}")

    # Save glTF (binary)
    scene.export(str(glb_file), file_type="glb")


def main(args: argparse.Namespace):
    dataset_path = Path(args.input)
    image_dir = dataset_path / "images"
    out_dir = dataset_path / "dust3r"
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = out_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logging.info(f"Using CUDA device: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Continue with CPU")

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    # load_images can take a list of images or a directory
    file_glob = [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]
    image_files = []
    for f in image_dir.iterdir():
        for e in file_glob:
            if str(f).endswith(e):
                image_files.append(f)
                break
    print(f"Loading images from {image_dir} ({len(image_files)} files)")

    images = load_images([str(f) for f in image_files], size=args.resize)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=args.batch_size)

    # Global alignment
    scene = global_aligner(
        output,
        device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
        shared_focal=args.single_camera,
    )

    loss = scene.compute_global_alignment(
        init="mst", niter=args.niter, schedule=args.schedule, lr=args.lr
    )
    print(f"Global alignment loss: {loss:.4f}")

    # retrieve useful values from scene:
    scene = scene.clean_pointcloud()
    imgs = scene.imgs
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()

    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(args.conf_threshold)))
    print(f"scene.min_conf_thr = {scene.min_conf_thr}")

    intrinsics = scene.get_intrinsics()
    confidence_masks = scene.get_masks()

    poses = [t.detach().cpu().numpy() for t in poses]
    pts3d = [t.detach().cpu().numpy() for t in pts3d]
    intrinsics = [t.detach().cpu().numpy() for t in intrinsics]
    confidence_masks = [t.detach().cpu().numpy() for t in confidence_masks]

    # Calculate image scaling factor
    org_image = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
    org_height, org_width = org_image.shape[:2]
    height, width = imgs[0].shape[:2]
    scale = min(org_height / height, org_width / width)

    # Save COLMAP cameras
    cam_txt = sparse_dir / "cameras.txt"
    with open(cam_txt, mode="w") as f:
        if args.single_camera:
            K = np.mean(intrinsics, axis=0)
            px = org_width * 0.5
            py = org_height * 0.5
            fx = px / K[0, 2] * K[0, 0]
            fy = py / K[1, 2] * K[1, 1]
            f.write(
                f"1 PINHOLE {org_width:d} {org_height:d} {fx:f} {fy:f} {px:f} {py:f}\n"
            )
        else:
            for i in tqdm(range(len(intrinsics)), desc="Saving COLMAP cameras"):
                K = intrinsics[i]
                px = org_width * 0.5
                py = org_height * 0.5
                fx = px / K[0, 2] * K[0, 0]
                fy = py / K[1, 2] * K[1, 1]

                # CAMERA_ID, TYPE, WIDTH, HEIGHT, FX, FY, PX, PY (for PINHOLE)
                f.write(
                    f"{i + 1:d} PINHOLE {org_width:d} {org_height:d} {fx:f} {fy:f} {px:f} {py:f}\n"
                )

    # Save COLMAP images
    img_txt = sparse_dir / "images.txt"
    with open(img_txt, mode="w") as f:
        for i in tqdm(range(len(imgs)), desc="Saving COLMAP images"):
            pose = np.linalg.inv(poses[i])
            tv = pose[:3, 3]
            qv = quaternion.from_rotation_matrix(pose[:3, :3])
            name = image_files[i].name
            cam_id = 1 if args.single_camera else i + 1
            f.write(
                f"{i + 1:d} {qv.w:f} {qv.x:f} {qv.y:f} {qv.z:f} {tv[0]:f} {tv[1]:f} {tv[2]:f} {cam_id:d} {name:s}\n"
            )
            f.write(
                "\n"
            )  # every 2nd line is for the point data, not needed for GSplats.

    # Save COLMAP points
    pts_txt = sparse_dir / "points3D.txt"
    point_id = 1
    points = []
    colors = []
    with open(pts_txt, mode="w") as f:
        for i in tqdm(range(len(pts3d)), desc="Saving COLMAP points"):
            conf_i = confidence_masks[i]
            pts = pts3d[i][conf_i]
            rgb = imgs[i][conf_i]
            rgb = (rgb * 255.0).astype(np.uint8)
            err = 0.0
            for j, p in enumerate(pts):
                # POINT_ID, X, Y, Z, R, G, B, ERR
                f.write(
                    f"{point_id:d} {p[0]} {p[1]} {p[2]} {rgb[j, 0]:d} {rgb[j, 1]:d} {rgb[j, 2]:d} {err:f}\n"
                )
                point_id += 1

            points.append(pts)
            colors.append(rgb)

    points = np.concatenate(points, axis=0).astype(np.float32)
    normals = np.zeros_like(points)
    colors = np.concatenate(colors, axis=0).astype(np.uint8)
    print(f"Total {len(points)} are detected!")

    # Save PLY
    point_data = [
        (x, y, z, nx, ny, nz, r, g, b)
        for (x, y, z), (nx, ny, nz), (r, g, b) in zip(points, normals, colors)
    ]
    point_data = np.array(
        point_data,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    elements = PlyElement.describe(point_data, "vertex")
    ply_data = PlyData([elements], text=False, byte_order="<")
    ply_data.write(sparse_dir / "points3D.ply")

    # Copy images
    out_image_dir = out_dir / "images"
    out_image_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(image_files):
        shutil.copy(f, out_image_dir / f.name)

    # Save glTF file for sanity check
    export_gltf(intrinsics, poses, points, colors, sparse_dir / "points3D.glb")

    # Save COLMAP database, if requested
    if args.save_db:
        # Delete old database file if exists
        database_path = out_dir / "database.db"
        if database_path.exists():
            database_path.unlink()

        # Create an empty database
        database = pycolmap.Database()
        database.open(str(database_path))
        database.close()

        # Import images
        camera_mode = pycolmap.CameraMode.SINGLE
        options = pycolmap.ImageReaderOptions()
        image_names = [f.name for f in image_files]
        with pycolmap.ostream():
            pycolmap.import_images(
                str(database_path),
                str(image_dir),
                camera_mode,
                image_names,
                options,
            )

        # Retrieve image ids
        database = pycolmap.Database()
        database.open(str(database_path))
        image_ids = []
        for img in database.read_all_images():
            image_ids.append(img.image_id)

        database.close()

        # Import features
        database = pycolmap.Database()
        database.open(str(database_path))
        for i, image_id in enumerate(image_ids):
            conf_i = confidence_masks[i]
            H, W = imgs[i].shape[:2]
            keypoints = np.array(xy_grid(W, H)[conf_i], dtype=np.float64)
            keypoints = (keypoints + 0.5) * scale  # COLMAP origin
            database.write_keypoints(image_id, keypoints)

        database.close()

        # Import matches
        database = pycolmap.Database()
        database.open(str(database_path))
        skip_geometric_verification = True

        matched: set[tuple[int, int]] = set()
        for pair in tqdm(pairs, desc="Exporting matches for COLMAP database"):
            id0 = pair[0]["idx"]
            id1 = pair[1]["idx"]
            if (id0, id1) in matched or (id1, id0) in matched:
                continue

            matched = matched.union({(id0, id1), (id1, id0)})

            pts2d_list, pts3d_list = [], []
            for i in [id0, id1]:
                conf_i = confidence_masks[i]
                H, W = imgs[i].shape[:2]
                pts2d_list.append(xy_grid(W, H)[conf_i])
                pts3d_list.append(pts3d[i][conf_i])

            reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(
                pts3d_list[0], pts3d_list[1]
            )

            matches0 = np.arange(len(pts2d_list[0]))[nn2_in_P1][reciprocal_in_P2]
            matches1 = np.arange(len(pts2d_list[1]))[reciprocal_in_P2]

            matches = np.stack([matches1, matches0], axis=1).astype(np.int32)
            # print(f'found {num_matches} matches ({id0} vs {id1})')

            database.write_matches(image_ids[id0], image_ids[id1], matches.tolist())
            if skip_geometric_verification:
                two_view_geo = pycolmap.TwoViewGeometry()
                two_view_geo.inlier_matches = matches.tolist()
                database.write_two_view_geometry(
                    image_ids[id1], image_ids[id0], two_view_geo
                )

        database.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-r", "--resize", type=int, default=512)
    parser.add_argument("--gpu", type=int, default=0, help="Device to run the model on")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for inference"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="Learning rate schedule",
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--niter", type=int, default=1000, help="Number of iterations")
    parser.add_argument(
        "--conf_threshold", type=float, default=3.0, help="Confidence threshold"
    )
    parser.add_argument(
        "--single_camera", action="store_true", help="Use single camera mode"
    )
    parser.add_argument(
        "--save_db", action="store_true", help="Save COLMAP database.db"
    )

    args = parser.parse_args()
    main(args)
