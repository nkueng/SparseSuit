import time
import pyrender
import trimesh
import numpy as np


def vis_smpl(
    faces,
    vertices: list,
    vertex_colors: list = None,
    play_frames: int = 100,
    playback_speed: float = 1,
    joints: list = None,
    sensors: list = None,
    accs: list = None,
    oris: list = None,
    export: bool = False,
    add_captions: bool = False,
    side_by_side: bool = False,
):
    """Displays play_frames number of frames of the given vertices with the given SMPL model."""
    # constants
    fps = 60 * playback_speed
    num_models = len(vertices)
    num_vertices = vertices[0].shape[1]
    num_frames = vertices[0].shape[0]
    colors = {
        "yellow": [1, 1, 0, 0.8],  # raw prediction
        "orange": [1, 0.5, 0, 0.8],  # aligned prediction
        "green": [0.17, 1, 0, 0.8],  # ground truth
        "black": [0, 0, 0, 0.8],  # black for text
    }

    # compute translation vectors in case of side-by-side comparison
    if side_by_side:
        dist = num_models - 1  # total distance between models furthest out
        pos = np.arange(0, dist + 1) - dist / 2
        transl = np.array([[0, i, 0] for i in pos])
    else:
        transl = np.zeros([num_models, 3])

    # first compute all objects for visualization
    print("Computing visuals...")
    nodes_buffer = []
    if play_frames == -1 or play_frames > num_frames:
        play_frames = num_frames
    for i in range(play_frames):

        nodes = []

        # vertices of SMPL model
        for v, vertex in enumerate(vertices):
            vertex_color = (
                np.ones([num_vertices, 4]) * colors[vertex_colors[v]]
                if vertex_colors is not None
                else [0.3, 0.3, 0.3, 0.8]
            )
            tri_mesh = trimesh.Trimesh(vertex[i], faces, vertex_colors=vertex_color)
            mesh = pyrender.Mesh.from_trimesh(tri_mesh)
            nodes.append(pyrender.Node(mesh=mesh, translation=transl[v]))

            # if export:
            #     # export mesh to ply file
            #     trimesh.exchange.export.export_mesh(tri_mesh, 'smpl_mesh.ply', 'ply')

        # joints of SMPL model
        if joints is not None:
            for j, joint in enumerate(joints):
                sm = trimesh.creation.uv_sphere(radius=0.005)
                sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
                tfs = np.tile(np.eye(4), (joint.shape[1], 1, 1))
                tfs[:, :3, 3] = joint[i]
                joint_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
                nodes.append(pyrender.Node(mesh=joint_pcl, translation=transl[j]))

        # IMU sensors placed on SMPL model (optionally rotated by orientation measurements)
        if sensors is not None:
            for s, sensor in enumerate(sensors):
                if oris is not None:
                    sens = trimesh.creation.axis(
                        origin_size=0.005, axis_radius=0.003, axis_length=0.03
                    )
                else:
                    sens = trimesh.creation.uv_sphere(radius=0.015)
                sens.visual.vertex_colors = [0, 0, 0, 1.0]
                tfs = np.tile(np.eye(4), (sensor.shape[1], 1, 1))
                if oris is not None:
                    tfs[:, :3, :3] = oris[s][i]
                tfs[:, :3, 3] = sensor[i]
                sens_pcl = pyrender.Mesh.from_trimesh(sens, poses=tfs)
                nodes.append(pyrender.Node(mesh=sens_pcl, translation=transl[s]))

        # visualize accelerometer measurements as green cylinders with origins at sensors
        if accs is not None:
            for a, acc in enumerate(accs):
                cyls = []
                tfs = []
                for m, meas in enumerate(acc[i]):
                    cyl_orig = sensors[a][i, m]
                    cyl_end = cyl_orig + meas / 100
                    endpoints = np.array([cyl_orig, cyl_end])
                    cyl = trimesh.creation.cylinder(radius=0.005, segment=endpoints)
                    cyl.visual.vertex_colors = [0.1, 0.9, 0.1, 1.0]
                    cyls.append(cyl)
                cyl_pcl = pyrender.Mesh.from_trimesh(cyls)
                nodes.append(pyrender.Node(mesh=cyl_pcl, translation=transl[a]))

        nodes_buffer.append(nodes)

    # add captions
    captions = []
    caption_template = {
        "text": "my caption",
        "color": colors["black"],
        "location": pyrender.constants.TextAlign.BOTTOM_CENTER,
        "font_name": "OpenSans-Regular",
        "font_pt": 20,
        "scale": 1,
    }
    caption_playback_speed = caption_template.copy()
    caption_playback_speed["text"] = str(playback_speed) + "x"
    captions.append(caption_playback_speed)

    if add_captions:
        caption_gt = {
            "text": "ground truth",
            "color": colors["green"],
            "location": pyrender.constants.TextAlign.BOTTOM_LEFT,
            "font_name": "OpenSans-Regular",
            "font_pt": 20,
            "scale": 1,
        }
        captions.append(caption_gt)
        caption_pred = caption_gt.copy()
        caption_pred["text"] = "prediction"
        caption_pred["color"] = colors["orange"]
        caption_pred["location"] = pyrender.constants.TextAlign.BOTTOM_RIGHT
        captions.append(caption_pred)

    # then render objects (first remove, then add new to reduce lag)
    scene = pyrender.Scene()
    v = pyrender.Viewer(
        scene,
        run_in_thread=True,
        use_raymond_lighting=True,
        refresh_rate=fps,
        caption=captions,
        use_perspective_cam=False,
    )

    old_nodes = []
    for i in range(play_frames):
        print("playing frame {}/{}".format(i + 1, play_frames))

        v.render_lock.acquire()
        # remove old nodes
        for node in old_nodes:
            scene.remove_node(node)

        # add new nodes
        for nodes in nodes_buffer[i]:
            scene.add_node(nodes)

        v.render_lock.release()
        # keep track of new nodes to remove next iteration
        old_nodes = nodes_buffer[i]

        time.sleep(1 / fps)

    v.close_external()
