"""
Real BEATs encoder wrapper — SKELETON ONLY, not runnable in this sandbox.

Why not runnable here: needs `torch` + `torchaudio` and a pretrained
checkpoint (~350-400MB) from Microsoft's release, which this sandbox's
network allowlist can't reach.

TO ACTIVATE THIS ON YOUR OWN MACHINE (Day 1 task):
  1. pip install torch torchaudio
  2. git clone https://github.com/microsoft/unilm && cd unilm/beats
  3. Download a pretrained checkpoint from that repo's README (their
     released checkpoints, e.g. "BEATs_iter3_plus_AS2M.pt")
  4. Set BEATS_CHECKPOINT_PATH below to that file's path
  5. Run scripts/day1_embedding_shape.py with --encoder beats

Once this class is filled in and satisfies the same AudioEncoder
interface, nothing else in the pipeline needs to change — scoring and
evaluation code is written against the interface, not this class.
"""

import numpy as np
from .base import AudioEncoder

BEATS_CHECKPOINT_PATH = "path/to/BEATs_checkpoint.pt"  # <-- set this locally


class BEATsEncoder(AudioEncoder):
    def __init__(self, checkpoint_path: str = BEATS_CHECKPOINT_PATH, device: str = "cpu"):
        try:
            import torch  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "torch is required for BEATsEncoder. Install with: "
                "pip install torch torchaudio"
            ) from e

        # TODO (Day 1, on your own machine):
        #   from BEATs import BEATs, BEATsConfig
        #   checkpoint = torch.load(checkpoint_path)
        #   cfg = BEATsConfig(checkpoint['cfg'])
        #   self.model = BEATs(cfg)
        #   self.model.load_state_dict(checkpoint['model'])
        #   self.model.eval()
        #   self._embedding_dim = cfg.encoder_embed_dim
        raise NotImplementedError(
            "Fill in BEATs model loading here once running with real "
            "network/GPU access. See module docstring for setup steps."
        )

    @property
    def embedding_dim(self) -> int:
        raise NotImplementedError

    def embed(self, window: np.ndarray, sr: int) -> np.ndarray:
        # TODO: resample to 16kHz if needed, convert to torch tensor,
        # run self.model.extract_features(...), mean-pool over time,
        # return as a numpy array.
        raise NotImplementedError
