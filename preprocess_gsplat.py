import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import pycolmap
import quaternion
from tqdm.auto import tqdm

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.geometry import xy_grid, find_reciprocal_matches


def main(args: argparse.Namespace):
    dataset_path = Path(args.input)
    image_dir = dataset_path / 'images'
    out_dir = dataset_path / 'dust3r'
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir = out_dir / 'sparse' / '0'
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        raise RuntimeError('CUDA is not available. Please check your setup.')

    model_name = 'naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt'
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    # load_images can take a list of images or a directory
    image_files = sorted(list(image_dir.glob('*.JPG')))
    print(f'Loading images from {image_dir} ({len(image_files)} files)')

    images = load_images([str(f) for f in image_files], size=args.resize)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=False)
    output = inference(pairs, model, device, batch_size=args.batch_size)

    # Global alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init='mst', niter=args.niter, schedule=args.schedule, lr=args.lr)
    print(f'Global alignment loss: {loss:.4f}')

    # retrieve useful values from scene:
    scene = scene.clean_pointcloud()
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # Calculate image scaling factor
    org_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    H, W = org_image.shape[:2]
    h, w = imgs[0].shape[:2]
    scale = min(H / h, W / w)

    # Save COLMAP cameras
    cam_txt = sparse_dir / 'cameras.txt'
    avg_focal = np.mean(focals.detach().cpu().numpy())
    with open(cam_txt, mode='w') as f:
        f.write(f'1 PINHOLE {W:d} {H:d} {avg_focal * scale:f} {W / 2:f} {H / 2:f}\n')

    # Save COLMAP images
    img_txt = sparse_dir / 'images.txt'
    global_point_id = 1
    with open(img_txt, mode='w') as f:
        for i in tqdm(range(len(imgs))):
            pose = poses[i].detach().cpu().numpy()
            tv = pose[:3, 3]
            qv = quaternion.from_rotation_matrix(pose[:3, :3])
            name = image_files[i].name
            f.write(f'{i + 1:d} {qv.w:f} {qv.x:f} {qv.y:f} {qv.z:f} {tv[0]:f} {tv[1]:f} {tv[2]:f} 1 {name:s}\n')

            conf_i = confidence_masks[i].detach().cpu().numpy()
            H, W = imgs[i].shape[:2]
            pts2d = xy_grid(W, H)[conf_i]
            pts2d_txt = [f'{p[0]} {p[1]} {k + global_point_id}' for k, p in enumerate(pts2d)]
            f.write(' '.join(pts2d_txt) + '\n')

            global_point_id += len(pts2d)

    # Save COLMAP points
    pts_txt = sparse_dir / 'points3D.txt'
    global_point_id = 1
    with open(pts_txt, mode='w') as f:
        for i in tqdm(range(len(pts3d))):
            pts = pts3d[i].reshape((-1, 3))
            rgb = imgs[i].reshape((-1, 3))
            rgb = (rgb * 255.0).astype(np.uint8)
            err = 0.0
            for k, p in enumerate(pts):
                pid = k + global_point_id
                f.write(
                    f'{pid} {p[0]} {p[1]} {p[2]} 1 {rgb[k, 0]:d} {rgb[k, 1]:d} {rgb[k, 2]:d} {err:f} {i + 1:d} {k:d}\n'
                )

            global_point_id += len(pts)

    # Copy images
    out_image_dir = out_dir / 'images'
    out_image_dir.mkdir(parents=True, exist_ok=True)

    for f in tqdm(image_files):
        shutil.copy(f, out_image_dir / f.name)

    # Delete old database file if exists
    database_path = out_dir / 'database.db'
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
        conf_i = confidence_masks[i].cpu().numpy()
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
    for pair in tqdm(pairs):
        id0 = pair[0]['idx']
        id1 = pair[1]['idx']
        if (id0, id1) in matched or (id1, id0) in matched:
            continue

        matched = matched.union({(id0, id1), (id1, id0)})

        pts2d_list, pts3d_list = [], []
        for i in [id0, id1]:
            conf_i = confidence_masks[i].cpu().numpy()
            H, W = imgs[i].shape[:2]
            pts2d_list.append(xy_grid(W, H)[conf_i])
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])

        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(pts3d_list[0], pts3d_list[1])

        matches0 = np.arange(len(pts2d_list[0]))[nn2_in_P1][reciprocal_in_P2]
        matches1 = np.arange(len(pts2d_list[1]))[reciprocal_in_P2]

        matches = np.stack([matches1, matches0], axis=1).astype(np.int32)
        print(f'found {len(matches)} matches ({id0} vs {id1})')

        database.write_matches(image_ids[id0], image_ids[id1], matches.tolist())
        if skip_geometric_verification:
            two_view_geo = pycolmap.TwoViewGeometry()
            two_view_geo.inlier_matches = matches.tolist()
            database.write_two_view_geometry(image_ids[id1], image_ids[id0], two_view_geo)

    database.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-r', '--resize', type=int, default=512)
    parser.add_argument('--gpu', type=int, default=0, help='Device to run the model on')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--schedule', type=str, default='cosine', help='Learning rate schedule')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--niter', type=int, default=300, help='Number of iterations')

    args = parser.parse_args()
    main(args)
