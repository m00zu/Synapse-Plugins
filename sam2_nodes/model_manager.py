"""
model_manager.py — Download and locate SAM2 + GroundingDINO ONNX model files.

Image-only models are stored in plugins/sam2_nodes/models/{variant}/
(fetched from HuggingFace as a zip).

Video models (with memory attention) are stored in
plugins/sam2_nodes/models/video_{variant}/ (fetched individually from
Google Cloud Storage, ailia pre-exports).

GroundingDINO Tiny ONNX is stored in plugins/sam2_nodes/models/grounding_dino_tiny/
(fetched from HuggingFace onnx-community repo).
"""
from __future__ import annotations

import logging
import pathlib
import zipfile

logger = logging.getLogger(__name__)

_PLUGIN_DIR = pathlib.Path(__file__).parent
_MODELS_DIR = _PLUGIN_DIR / "models"


class SAM2ModelManager:
    REPO_ID = "vietanhdev/segment-anything-2.1-onnx-models"
    VARIANTS: dict[str, tuple[str, str]] = {
        # variant_key: (zip_filename, model_prefix)
        "tiny":      ("sam2.1_hiera_tiny_20260221.zip",      "sam2.1_hiera_tiny"),
        "small":     ("sam2.1_hiera_small_20260221.zip",     "sam2.1_hiera_small"),
        "base_plus": ("sam2.1_hiera_base_plus_20260221.zip", "sam2.1_hiera_base_plus"),
        "large":     ("sam2.1_hiera_large_20260221.zip",     "sam2.1_hiera_large"),
    }

    SIZE_ESTIMATES: dict[str, str] = {
        "tiny": "~111 MB",
        "small": "~150 MB",
        "base_plus": "~250 MB",
        "large": "~550 MB",
    }

    # -- Video (memory attention) models from ailia pre-exports ----------
    VIDEO_BASE_URL = (
        "https://storage.googleapis.com/ailia-models/segment-anything-2")
    VIDEO_MODELS = [
        'image_encoder', 'prompt_encoder', 'mask_decoder',
        'memory_encoder', 'memory_attention', 'mlp',
    ]
    VIDEO_VARIANT_MAP: dict[str, str] = {
        "tiny":      "hiera_t",
        "small":     "hiera_s",
        "base_plus": "hiera_b+",
        "large":     "hiera_l",
    }
    VIDEO_SIZE_ESTIMATES: dict[str, str] = {
        "tiny": "~100 MB",
        "small": "~165 MB",
        "base_plus": "~290 MB",
        "large": "~600 MB",
    }

    def _variant_dir(self, variant: str) -> pathlib.Path:
        return _MODELS_DIR / variant

    def _onnx_paths(self, variant: str) -> tuple[pathlib.Path, pathlib.Path]:
        _, prefix = self.VARIANTS[variant]
        d = self._variant_dir(variant)
        return d / f"{prefix}.encoder.onnx", d / f"{prefix}.decoder.onnx"

    def is_downloaded(self, variant: str) -> bool:
        enc, dec = self._onnx_paths(variant)
        return enc.is_file() and dec.is_file()

    def get_model_paths(self, variant: str = "tiny") -> tuple[pathlib.Path, pathlib.Path]:
        """Return (encoder_path, decoder_path), downloading if needed."""
        if variant not in self.VARIANTS:
            raise ValueError(f"Unknown variant '{variant}'. "
                             f"Choose from: {list(self.VARIANTS)}")
        if not self.is_downloaded(variant):
            self.download_model(variant)
        return self._onnx_paths(variant)

    def download_model(self, variant: str) -> tuple[pathlib.Path, pathlib.Path]:
        """Download zip from HuggingFace, extract encoder + decoder ONNX."""
        from huggingface_hub import hf_hub_download

        zip_name, prefix = self.VARIANTS[variant]
        logger.info("Downloading SAM2 '%s' from %s …", variant, self.REPO_ID)

        zip_path = hf_hub_download(
            repo_id=self.REPO_ID,
            filename=zip_name,
        )

        dest = self._variant_dir(variant)
        dest.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting to %s …", dest)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                # Extract flat (ignore any subdirectory inside zip)
                fname = pathlib.Path(member.filename).name
                if fname.endswith(".onnx") or fname.endswith(".yaml"):
                    with zf.open(member) as src, \
                         open(dest / fname, "wb") as dst:
                        dst.write(src.read())

        enc, dec = self._onnx_paths(variant)
        if not enc.is_file() or not dec.is_file():
            raise FileNotFoundError(
                f"Extraction failed — expected:\n  {enc}\n  {dec}")
        logger.info("SAM2 '%s' ready.", variant)
        return enc, dec

    def list_variants(self) -> list[dict]:
        """Return info about all variants."""
        result = []
        for key in self.VARIANTS:
            result.append({
                "name": key,
                "size": self.SIZE_ESTIMATES.get(key, "?"),
                "downloaded": self.is_downloaded(key),
            })
        return result

    # ── Video (memory attention) models ────────────────────────────────

    def _video_model_dir(self, variant: str) -> pathlib.Path:
        return _MODELS_DIR / f"video_{variant}"

    def _video_model_filename(self, model_name: str, variant: str) -> str:
        v = self.VIDEO_VARIANT_MAP[variant]
        if model_name == 'memory_attention':
            return f"{model_name}_{v}.opt.onnx"
        return f"{model_name}_{v}.onnx"

    def are_video_models_downloaded(self, variant: str) -> bool:
        if variant not in self.VIDEO_VARIANT_MAP:
            return False
        d = self._video_model_dir(variant)
        return all(
            (d / self._video_model_filename(m, variant)).is_file()
            for m in self.VIDEO_MODELS)

    def get_video_model_paths(
        self, variant: str = "tiny",
    ) -> dict[str, pathlib.Path]:
        """Return {model_name: path} dict, downloading if needed."""
        if variant not in self.VIDEO_VARIANT_MAP:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from: {list(self.VIDEO_VARIANT_MAP)}")
        if not self.are_video_models_downloaded(variant):
            self.download_video_models(variant)
        d = self._video_model_dir(variant)
        return {
            m: d / self._video_model_filename(m, variant)
            for m in self.VIDEO_MODELS
        }

    def download_video_models(
        self, variant: str, progress_cb=None,
    ) -> dict[str, pathlib.Path]:
        """Download individual ONNX files from Google Cloud Storage."""
        import urllib.request
        import socket
        socket.setdefaulttimeout(300)  # 5-minute timeout for large model downloads

        d = self._video_model_dir(variant)
        d.mkdir(parents=True, exist_ok=True)

        paths: dict[str, pathlib.Path] = {}
        total = len(self.VIDEO_MODELS)

        for i, model_name in enumerate(self.VIDEO_MODELS):
            fname = self._video_model_filename(model_name, variant)
            dest = d / fname
            paths[model_name] = dest

            if dest.is_file():
                logger.info("Video model '%s' already present", model_name)
            else:
                url = f"{self.VIDEO_BASE_URL}/{fname}"
                logger.info("Downloading %s from %s …", model_name, url)
                try:
                    urllib.request.urlretrieve(url, str(dest))
                    sz = dest.stat().st_size / 1e6
                    logger.info("Downloaded %s (%.1f MB)", model_name, sz)
                except Exception:
                    # Clean up partial file
                    if dest.exists():
                        dest.unlink()
                    raise

            if progress_cb:
                progress_cb((i + 1) / total)

        logger.info("SAM2 video models '%s' ready.", variant)
        return paths

    # ── GroundingDINO Tiny (text-prompted object detection) ───────────

    GDINO_HF_REPO = "onnx-community/grounding-dino-tiny-ONNX"
    GDINO_FILES = [
        "onnx/model_quantized.onnx",
        "tokenizer.json",
    ]
    GDINO_SIZE_ESTIMATE = "~204 MB"

    def _gdino_dir(self) -> pathlib.Path:
        return _MODELS_DIR / "grounding_dino_tiny"

    def is_gdino_downloaded(self) -> bool:
        d = self._gdino_dir()
        return (d / "model_quantized.onnx").is_file() and \
               (d / "tokenizer.json").is_file()

    def get_gdino_model_path(self) -> pathlib.Path:
        """Return path to GroundingDINO ONNX model, downloading if needed."""
        if not self.is_gdino_downloaded():
            self.download_gdino()
        return self._gdino_dir() / "model_quantized.onnx"

    def get_gdino_tokenizer_path(self) -> pathlib.Path:
        """Return path to tokenizer.json, downloading if needed."""
        if not self.is_gdino_downloaded():
            self.download_gdino()
        return self._gdino_dir() / "tokenizer.json"

    def download_gdino(self) -> None:
        """Download GroundingDINO Tiny ONNX from HuggingFace."""
        from huggingface_hub import hf_hub_download

        d = self._gdino_dir()
        d.mkdir(parents=True, exist_ok=True)

        for hf_path in self.GDINO_FILES:
            local_name = pathlib.Path(hf_path).name
            dest = d / local_name
            if dest.is_file():
                logger.info("GroundingDINO '%s' already present", local_name)
                continue
            logger.info("Downloading GroundingDINO '%s' from %s …",
                        local_name, self.GDINO_HF_REPO)
            cached = hf_hub_download(
                repo_id=self.GDINO_HF_REPO,
                filename=hf_path,
            )
            # Copy from HF cache to our models dir
            import shutil
            shutil.copy2(cached, dest)
            sz = dest.stat().st_size / 1e6
            logger.info("Downloaded %s (%.1f MB)", local_name, sz)

        logger.info("GroundingDINO Tiny ready.")
