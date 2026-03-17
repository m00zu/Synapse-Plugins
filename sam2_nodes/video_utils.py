"""
video_utils.py — Video utility nodes and helpers.

Contains VideoToFramesNode and the shared _get_reader helper.
"""
from __future__ import annotations

import logging
import os

from PIL import Image

import NodeGraphQt
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget

from data_models import ImageData
from nodes.base import BaseExecutionNode, PORT_COLORS, NodeFileSelector, NodeDirSelector

logger = logging.getLogger(__name__)

__all__ = ['VideoToFramesNode']


def _get_reader(video_path: str):
    """Open a video reader using imageio + ffmpeg backend."""
    import imageio
    return imageio.get_reader(video_path, 'ffmpeg')


class VideoToFramesNode(BaseExecutionNode):
    """Extract frames from a video file and save as numbered images.

    Select a video file and an output folder.  Run the node once to
    export all (or a range of) frames as individual image files.

    Keywords: video, frames, extract, split, export, mp4, avi, 影片, 幀, 拆分, 轉換
    """

    __identifier__ = 'plugins.Plugins.VideoAnalysis'
    NODE_NAME      = 'Video to Frames'
    PORT_SPEC      = {'inputs': [], 'outputs': ['path']}

    _UI_PROPS = frozenset({
        'color', 'pos', 'selected', 'name', 'progress',
    })

    def __init__(self):
        super().__init__()

        self.add_output('folder_path', color=PORT_COLORS['path'])

        file_selector = NodeFileSelector(
            self.view, name='video_path', label='Video')
        self.add_custom_widget(
            file_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties',
        )

        dir_selector = NodeDirSelector(
            self.view, name='output_folder', label='Output')
        self.add_custom_widget(
            dir_selector,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
            tab='Properties',
        )

        self.add_text_input('frame_start', 'Start Frame', text='1')
        self.add_text_input('frame_end', 'End Frame', text='')

        self.add_combo_menu(
            'output_format', 'Format',
            items=['png', 'tif', 'jpg'])

        self.add_text_input('pad_digits', 'Pad Digits', text='4')

    def evaluate(self):
        self.reset_progress()

        video_path = self.get_property('video_path')
        output_folder = self.get_property('output_folder')

        if not video_path or not os.path.exists(video_path):
            return False, f"Video not found: {video_path}"
        if not output_folder:
            return False, "No output folder specified"

        os.makedirs(output_folder, exist_ok=True)

        fmt = self.get_property('output_format') or 'png'
        pad = int(self.get_property('pad_digits') or 4)

        try:
            reader = _get_reader(video_path)
            n_frames = reader.count_frames()
        except Exception as exc:
            return False, f"Cannot open video: {exc}"

        start_1 = int(self.get_property('frame_start') or 1)
        end_str = self.get_property('frame_end') or ''
        end_1 = int(end_str) if end_str.strip() else n_frames

        start = max(0, min(start_1 - 1, n_frames - 1))
        end = max(start, min(end_1, n_frames))
        total = end - start

        if total == 0:
            reader.close()
            return False, "No frames in selected range"

        self.set_progress(5)

        saved = 0
        try:
            for i in range(start, end):
                if self.cancel_requested:
                    reader.close()
                    return False, f"Cancelled after {saved} frames"

                frame_arr = reader.get_data(i)
                pil = Image.fromarray(frame_arr)

                fname = f"frame_{str(i + 1).zfill(pad)}.{fmt}"
                out_path = os.path.join(output_folder, fname)

                if fmt == 'jpg':
                    pil.save(out_path, quality=95)
                elif fmt == 'tif':
                    pil.save(out_path, compression='tiff_lzw')
                else:
                    pil.save(out_path)

                saved += 1
                pct = 5 + int(90 * saved / total)
                self.set_progress(pct)

        except Exception as exc:
            reader.close()
            return False, f"Error at frame {i}: {exc}"

        reader.close()
        self.set_progress(100)
        self.output_values['folder_path'] = output_folder
        self.mark_clean()
        logger.info("Exported %d frames to %s", saved, output_folder)
        return True, None
