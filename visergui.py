from threading import Thread
import torch
import numpy as np
import time
import viser
import viser.transforms as tf
from omegaconf import OmegaConf
from utils import qvec2rotmat
import cv2
from utils import Timer
from collections import deque


def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c

class RenderThread(Thread):
    pass


class ViserViewer:
    def __init__(self, device, viewer_port):
        self.device = device
        self.port = viewer_port

        self.render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        self.reset_view_button = self.server.add_gui_button("Reset View")

        self.need_update = False

        self.pause_training = False
        self.train_viewer_update_period_slider = self.server.add_gui_slider(
            "Train Viewer Update Period",
            min=1,
            max=100,
            step=1,
            initial_value=10,
            disabled=self.pause_training,
        )

        self.pause_training_button = self.server.add_gui_button("Pause Training")
        self.sh_order = self.server.add_gui_slider(
            "SH Order", min=1, max=4, step=1, initial_value=1
        )
        self.resolution_slider = self.server.add_gui_slider(
            "Resolution", min=384, max=4096, step=2, initial_value=1024
        )
        self.near_plane_slider = self.server.add_gui_slider(
            "Near", min=0.1, max=30, step=0.5, initial_value=0.1
        )
        self.far_plane_slider = self.server.add_gui_slider(
            "Far", min=30.0, max=1000.0, step=10.0, initial_value=1000.0
        )

        self.show_train_camera = self.server.add_gui_checkbox(
            "Show Train Camera", initial_value=False
        )

        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

        @self.show_train_camera.on_update
        def _(_):
            self.need_update = True

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.near_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.far_plane_slider.on_update
        def _(_):
            self.need_update = True

        @self.pause_training_button.on_click
        def _(_):
            self.pause_training = not self.pause_training
            self.train_viewer_update_period_slider.disabled = not self.pause_training
            self.pause_training_button.name = (
                "Resume Training" if self.pause_training else "Pause Training"
            )

        @self.reset_view_button.on_click
        def _(_):
            self.need_update = True
            for client in self.server.get_clients().values():
                client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                    [0.0, -1.0, 0.0]
                )

        self.c2ws = []
        self.camera_infos = []

        @self.resolution_slider.on_update
        def _(_):
            self.need_update = True

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_):
                self.need_update = True

        self.debug_idx = 0

    def set_renderer(self, renderer):
        self.renderer = renderer

    @torch.no_grad()
    def update(self):
        if self.need_update:
            start = time.time()
            for client in self.server.get_clients().values():
                camera = client.camera
                w2c = get_w2c(camera)
                try:
                    W = self.resolution_slider.value
                    H = int(self.resolution_slider.value/camera.aspect)
                    focal_x = W/2/np.tan(camera.fov/2)
                    focal_y = H/2/np.tan(camera.fov/2)

                    start_cuda = torch.cuda.Event(enable_timing=True)
                    end_cuda = torch.cuda.Event(enable_timing=True)
                    start_cuda.record()

                    outputs = self.renderer.test(
                        None,
                        extrinsics={
                            "rot": w2c[:3,:3], 
                            "tran": w2c[:3, 3],
                        },
                        intrinsics={
                            "width": W,
                            "height": H,
                            "focal_x": focal_x,
                            "focal_y": focal_y,
                        }
                    )
                    end_cuda.record()
                    torch.cuda.synchronize()
                    interval = start_cuda.elapsed_time(end_cuda)/1000.

                    out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                except RuntimeError as e:
                    print(e)
                    continue
                client.set_background_image(out, format="jpeg")
                self.debug_idx += 1
                # if self.debug_idx % 100 == 0:
                #     cv2.imwrite(
                #         f"./tmp/viewer/debug_{self.debug_idx}.png",
                #         cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                #     )

            self.render_times.append(interval)
            self.fps.value = f"{1.0 / np.mean(self.render_times):.3g}"
            # print(f"Update time: {end - start:.3g}")

