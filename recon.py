from __future__ import annotations

import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.data import MeshViewSynthesisDataset
from rfstudio.data.dataparser import RFMaskedRealDataparser
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Cameras, RGBAImages
from rfstudio.io import load_float32_image
from rfstudio.model import GSplatter
from rfstudio.trainer import GSplatTrainer
from rfstudio.visualization import Visualizer


@dataclass
class Script(Task):

    name: str = ...
    idx: int = 0
    base_dir: Path = Path('output')
    vis: bool = False

    def run(self) -> None:

        images = load_float32_image(self.base_dir / self.name / f'{self.idx}.png')
        assert images.shape == (256, 4096, 3)
        images = images.view(256, 16, 256, 3).permute(1, 0, 2, 3).contiguous()
        gt_outputs = RGBAImages(torch.cat((images, torch.ones_like(images[..., :1])), dim=-1))
        with open('meta_info/camera-16.pkl', 'rb') as f:
            K, azimuths, elevations, distances, cam_poses = pickle.load(f)
        extrinsics = torch.from_numpy(cam_poses).float()
        intrinsic = torch.from_numpy(K).float().expand(16, 3, 3).contiguous()
        c2w = torch.cat((extrinsics[:, :3, :3].transpose(1, 2), extrinsics[:, :3, :3].transpose(1, 2) @ -extrinsics[:, :3, 3:]), dim=-1)
        c2w[:, :, 1] *= -1
        c2w[:, :, 2] *= -1
        cameras = Cameras(
            c2w=c2w,
            fx=intrinsic[:, 0, 0],
            fy=intrinsic[:, 1, 1],
            cx=intrinsic[:, 0, 2],
            cy=intrinsic[:, 1, 2],
            width=torch.tensor(256).long().repeat(16),
            height=torch.tensor(256).long().repeat(16),
            near=torch.tensor(1e-2).float().repeat(16),
            far=torch.tensor(1e2).float().repeat(16),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            RFMaskedRealDataparser.dump(cameras, gt_outputs, None, path=Path(tmpdir), split='all')
            if self.vis:
                dataset = MeshViewSynthesisDataset(path=Path(tmpdir))
                dataset.__setup__()
                Visualizer().show(dataset=dataset)
            else:
                gs_task = TrainTask(
                    dataset=MeshViewSynthesisDataset(path=Path(tmpdir)),
                    model=GSplatter(
                        background_color='white',
                        sh_degree=3,
                        prepare_densification=True,
                    ),
                    experiment=Experiment(name='recon', timestamp=f'{self.name}_{self.idx}', output_dir=self.base_dir),
                    trainer=GSplatTrainer(
                        num_steps=30000,
                        batch_size=1,
                        num_steps_per_val=1000,
                        mixed_precision=False,
                        full_test_after_train=False,
                    ),
                    cuda=0,
                    seed=1,
                )
                gs_task.join()

tasks = { 'custom': Script(name=...) }
for filename in Path('testset').glob("*.png"):
    if (Path('output') / filename.stem).exists():
        tasks[filename.stem] = Script(name=filename.stem)

if __name__ == '__main__':
    TaskGroup(**tasks).run()
