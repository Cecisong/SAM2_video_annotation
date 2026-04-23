#!/usr/bin/env python3
"""
Interactive SAM2 video annotation GUI with YOLO bbox export.

Features:
- multiple objects with separate obj_id / class_id
- positive / negative point clicks
- optional box prompt by drag
- frame navigation
- apply prompt on current frame
- propagate through the whole video
- export YOLO bbox labels from propagated masks

Notes:
- For mp4 input, this script extracts frames with OpenCV first, then runs SAM2 on
  the extracted image folder. This avoids requiring `decord`.
- Exported YOLO txt files use the same stem as the extracted image names, so the
  image-label pairing stays aligned even when frame skipping is enabled.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
from tkinter import filedialog, messagebox, ttk


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_CANVAS_W = 1280
MAX_CANVAS_H = 800


def require_module(module_name: str, install_hint: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except Exception as exc:
        raise RuntimeError(f"Missing dependency `{module_name}`. {install_hint}") from exc


def build_predictor(model_cfg: str, checkpoint: Path, device: str):
    try:
        from sam2.build_sam import build_sam2_video_predictor
    except Exception as exc:
        raise RuntimeError(
            "Unable to import `sam2.build_sam.build_sam2_video_predictor`. "
            "Make sure the SAM2 package/codebase is installed and visible to this Python."
        ) from exc
    return build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)


def _frame_stem_sort_key(filename: str) -> Tuple[int, str]:
    stem = os.path.splitext(filename)[0]
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1)), stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits), stem
    return sys.maxsize, stem


def patch_sam2_jpg_loader_for_prefixed_names() -> None:
    import sam2.utils.misc as misc

    if getattr(misc.load_video_frames_from_jpg_images, "_supports_prefixed_names", False):
        return

    def patched_load_video_frames_from_jpg_images(
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames=False,
        compute_device=None,
    ):
        if isinstance(video_path, str) and os.path.isdir(video_path):
            jpg_folder = video_path
        else:
            raise NotImplementedError(
                "Only JPEG frames are supported at this moment."
            )

        frame_names = [
            p
            for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=_frame_stem_sort_key)
        num_frames = len(frame_names)
        if num_frames == 0:
            raise RuntimeError(f"no images found in {jpg_folder}")
        img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
        img_mean_tensor = misc.torch.tensor(img_mean, dtype=misc.torch.float32)[:, None, None]
        img_std_tensor = misc.torch.tensor(img_std, dtype=misc.torch.float32)[:, None, None]

        if async_loading_frames:
            lazy_images = misc.AsyncVideoFrameLoader(
                img_paths,
                image_size,
                offload_video_to_cpu,
                img_mean_tensor,
                img_std_tensor,
                compute_device,
            )
            return lazy_images, lazy_images.video_height, lazy_images.video_width

        images = misc.torch.zeros(num_frames, 3, image_size, image_size, dtype=misc.torch.float32)
        for n, img_path in enumerate(misc.tqdm(img_paths, desc="frame loading (JPEG)")):
            images[n], video_height, video_width = misc._load_img_as_tensor(img_path, image_size)
        if not offload_video_to_cpu:
            images = images.to(compute_device)
            img_mean_tensor = img_mean_tensor.to(compute_device)
            img_std_tensor = img_std_tensor.to(compute_device)
        images -= img_mean_tensor
        images /= img_std_tensor
        return images, video_height, video_width

    patched_load_video_frames_from_jpg_images._supports_prefixed_names = True
    misc.load_video_frames_from_jpg_images = patched_load_video_frames_from_jpg_images


def infer_device(explicit_device: Optional[str]) -> str:
    if explicit_device:
        return explicit_device
    torch = require_module("torch", "Install torch first.")
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_inference_context(device: str):
    normalized_device = (device or "").lower()
    if normalized_device.startswith("cuda"):
        torch = require_module("torch", "Install torch first.")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def sort_key_for_frame(path: Path) -> Tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        return int(digits), stem
    return sys.maxsize, stem


def list_frame_paths(frames_dir: Path) -> List[Path]:
    frames = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    frames.sort(key=sort_key_for_frame)
    if not frames:
        raise RuntimeError(f"No image frames found in {frames_dir}")
    return frames


class FrameProvider:
    """Load frames on demand and keep only a small LRU cache in memory."""

    def __init__(self, frame_paths: Sequence[Path], cache_size: int):
        if cache_size < 1:
            raise RuntimeError("--ui-cache-size must be >= 1")
        self.frame_paths = list(frame_paths)
        self.cache_size = cache_size
        self._cache: "OrderedDict[int, Image.Image]" = OrderedDict()
        with Image.open(self.frame_paths[0]) as img:
            self._size = img.size

    @property
    def size(self) -> Tuple[int, int]:
        return self._size

    def __len__(self) -> int:
        return len(self.frame_paths)

    def frame_name(self, index: int) -> str:
        return self.frame_paths[index].stem

    def get_frame(self, index: int) -> Image.Image:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached.copy()

        with Image.open(self.frame_paths[index]) as img:
            frame = img.convert("RGB").copy()
        self._cache[index] = frame
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return frame.copy()

    def preload_neighbors(self, index: int, radius: int = 1) -> None:
        for candidate in range(max(0, index - radius), min(len(self.frame_paths), index + radius + 1)):
            if candidate in self._cache:
                self._cache.move_to_end(candidate)
                continue
            with Image.open(self.frame_paths[candidate]) as img:
                frame = img.convert("RGB").copy()
            self._cache[candidate] = frame
            self._cache.move_to_end(candidate)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)


def extract_frames_from_video(
    video_path: Path,
    frames_dir: Path,
    frame_stride: int,
    jpg_quality: int,
) -> List[Path]:
    cv2 = require_module(
        "cv2",
        "Install opencv-python, or use --frames-dir with pre-extracted frames.",
    )
    if frame_stride < 1:
        raise RuntimeError("--frame-stride must be >= 1")
    if frames_dir.exists():
        existing_items = list(frames_dir.iterdir())
        if existing_items:
            raise RuntimeError(
                f"Target images directory is not empty: {frames_dir}. "
                "Please use a clean output dir or remove existing images first."
            )
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    saved_paths: List[Path] = []
    frame_idx = 0
    saved_idx = 0
    video_prefix = video_path.stem
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride == 0:
                out_path = frames_dir / f"{video_prefix}_{saved_idx:06d}.jpg"
                success = cv2.imwrite(
                    str(out_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)],
                )
                if not success:
                    raise RuntimeError(f"Failed to write extracted frame: {out_path}")
                saved_paths.append(out_path)
                saved_idx += 1
            frame_idx += 1
    finally:
        cap.release()

    if not saved_paths:
        raise RuntimeError(
            f"No frames extracted from {video_path}. Check video readability and frame stride."
        )
    return saved_paths


def mask_tensor_to_numpy(mask_like) -> np.ndarray:
    if hasattr(mask_like, "detach"):
        array = mask_like.detach().cpu().float().numpy()
    else:
        array = np.asarray(mask_like)
    return np.squeeze(array)


def binary_mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_to_yolo_line(
    bbox: Tuple[int, int, int, int],
    class_id: int,
    image_width: int,
    image_height: int,
) -> str:
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    center_x = x1 + box_w / 2.0
    center_y = y1 + box_h / 2.0
    return (
        f"{class_id} "
        f"{center_x / image_width:.6f} "
        f"{center_y / image_height:.6f} "
        f"{box_w / image_width:.6f} "
        f"{box_h / image_height:.6f}"
    )


@dataclass
class PromptState:
    points: List[Tuple[float, float, int]] = field(default_factory=list)
    box: Optional[Tuple[float, float, float, float]] = None


@dataclass
class ObjectMeta:
    class_id: int
    class_name: Optional[str] = None


class InteractiveAnnotator:
    def __init__(
        self,
        root: tk.Tk,
        predictor,
        inference_state,
        source_path: str,
        output_dir: Path,
        frame_provider: FrameProvider,
        mask_threshold: float,
        inference_context_factory,
        memory_window: int,
    ):
        self.root = root
        self.predictor = predictor
        self.state = inference_state
        self.source_path = source_path
        self.output_dir = output_dir
        self.frame_provider = frame_provider
        self.mask_threshold = mask_threshold
        self.inference_context_factory = inference_context_factory
        self.memory_window = memory_window

        self.frame_count = len(self.frame_provider)
        self.current_frame_idx = 0
        self.current_mode = tk.StringVar(value="positive")
        self.current_obj_id = tk.StringVar(value="1")
        self.current_class_id = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="Ready")
        self.all_classes_listbox: Optional[tk.Listbox] = None
        self.frame_classes_listbox: Optional[tk.Listbox] = None
        self.frame_objects_listbox: Optional[tk.Listbox] = None
        self.all_class_ids: List[int] = []
        self.frame_class_ids: List[int] = []
        self.frame_object_ids: List[int] = []

        self.prompt_store: Dict[int, Dict[int, PromptState]] = {}
        self.object_meta: Dict[int, ObjectMeta] = {}
        self.results_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        self.selected_pred_obj_id: Optional[int] = None
        self.selected_pred_frame_idx: Optional[int] = None
        self.drag_start: Optional[Tuple[float, float]] = None
        self.drag_current: Optional[Tuple[float, float]] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.display_scale = 1.0
        self.display_size = self.frame_provider.size

        self._build_ui()
        self._render_frame()

    def _build_ui(self) -> None:
        self.root.title("SAM2 Interactive Video Annotator")
        self.root.geometry("1600x960")

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        center = ttk.Frame(main, padding=8)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        class_panel = ttk.Frame(main, padding=8, width=280)
        class_panel.pack(side=tk.RIGHT, fill=tk.Y)
        class_panel.pack_propagate(False)

        ttk.Label(left, text="Object ID").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=self.current_obj_id, width=12).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(left, text="Class ID").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=self.current_class_id, width=12).pack(anchor=tk.W, pady=(0, 8))

        ttk.Button(left, text="Set Object/Class", command=self._set_current_object).pack(fill=tk.X, pady=2)
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        ttk.Label(left, text="Mode").pack(anchor=tk.W)
        for label, value in [
            ("Positive Point", "positive"),
            ("Negative Point", "negative"),
            ("Box Drag", "box"),
        ]:
            ttk.Radiobutton(left, text=label, value=value, variable=self.current_mode).pack(anchor=tk.W)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="Apply Current Prompt", command=self._apply_current_prompt).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Update Memory", command=self._update_memory).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Clear Current Obj Prompt On Frame", command=self._clear_current_prompt).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Propagate Whole Video", command=self._propagate_all).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Propagate Current -> Target", command=self._propagate_range).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Export YOLO", command=self._export_yolo).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="Prev Frame", command=self._prev_frame).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Next Frame", command=self._next_frame).pack(fill=tk.X, pady=2)

        self.frame_entry = tk.StringVar(value="0")
        ttk.Label(left, text="Jump To Frame").pack(anchor=tk.W, pady=(8, 0))
        ttk.Entry(left, textvariable=self.frame_entry, width=12).pack(anchor=tk.W, pady=(0, 4))
        ttk.Button(left, text="Go", command=self._jump_to_frame).pack(fill=tk.X, pady=2)
        self.target_frame_entry = tk.StringVar(value=str(self.frame_count - 1))
        ttk.Label(left, text="Target Frame").pack(anchor=tk.W, pady=(8, 0))
        ttk.Entry(left, textvariable=self.target_frame_entry, width=12).pack(anchor=tk.W, pady=(0, 4))

        tips = (
            "Left click: add point in current mode\n"
            "Right click: select predicted bbox\n"
            "Box mode: press and drag\n"
            "Apply prompt: run SAM2 on current frame\n"
            "Update Memory: add a new tracked object on this frame\n"
            "Delete: remove selected label on current frame\n"
            "Propagate Current -> Target: track from current frame to target frame\n"
            "Propagate Whole Video: track across the whole video\n"
            "Export YOLO: write txt labels"
        )
        ttk.Label(left, text=tips, wraplength=240, justify=tk.LEFT).pack(anchor=tk.W, pady=(12, 0))

        self.canvas = tk.Canvas(center, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<Button-3>", self._on_right_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)

        ttk.Label(class_panel, text="All Classes").pack(anchor=tk.W)
        self.all_classes_listbox = tk.Listbox(class_panel, exportselection=False, height=14)
        self.all_classes_listbox.pack(fill=tk.BOTH, expand=False, pady=(4, 10))
        self.all_classes_listbox.bind("<<ListboxSelect>>", self._on_select_all_class)
        self.all_classes_listbox.bind("<Double-Button-1>", self._on_rename_all_class)

        ttk.Label(class_panel, text="Classes In Current Frame").pack(anchor=tk.W)
        self.frame_classes_listbox = tk.Listbox(class_panel, exportselection=False, height=14)
        self.frame_classes_listbox.pack(fill=tk.BOTH, expand=False, pady=(4, 10))
        self.frame_classes_listbox.bind("<<ListboxSelect>>", self._on_select_frame_class)

        ttk.Label(class_panel, text="Objects In Current Frame").pack(anchor=tk.W)
        self.frame_objects_listbox = tk.Listbox(class_panel, exportselection=False, height=16)
        self.frame_objects_listbox.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.frame_objects_listbox.bind("<<ListboxSelect>>", self._on_select_frame_object)

        bottom = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status_var).pack(anchor=tk.W)

        self.root.bind("<Left>", lambda _e: self._prev_frame())
        self.root.bind("<Right>", lambda _e: self._next_frame())
        self.root.bind("a", lambda _e: self._prev_frame())
        self.root.bind("A", lambda _e: self._prev_frame())
        self.root.bind("d", lambda _e: self._next_frame())
        self.root.bind("D", lambda _e: self._next_frame())
        self.root.bind("<Delete>", self._on_delete_selected)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def _safe_int(self, value: str, label: str) -> int:
        try:
            return int(value)
        except ValueError as exc:
            raise RuntimeError(f"{label} must be an integer.") from exc

    def _set_current_object(self) -> None:
        obj_id = self._safe_int(self.current_obj_id.get(), "Object ID")
        class_id = self._safe_int(self.current_class_id.get(), "Class ID")
        self.object_meta[obj_id] = ObjectMeta(class_id=class_id)
        self._set_status(f"Current object set: obj_id={obj_id}, class_id={class_id}")
        self._refresh_class_lists()
        self._render_frame()

    def _sync_current_selection(self, obj_id: int, class_id: int) -> None:
        self.current_obj_id.set(str(obj_id))
        self.current_class_id.set(str(class_id))
        if obj_id not in self.object_meta or self.object_meta[obj_id].class_id != class_id:
            self.object_meta[obj_id] = ObjectMeta(class_id=class_id)
        self._refresh_class_lists()

    def _class_for_obj(self, obj_id: int) -> int:
        return self.object_meta.get(obj_id, ObjectMeta(class_id=0)).class_id

    def _class_name_for_id(self, class_id: int) -> Optional[str]:
        for meta in self.object_meta.values():
            if meta.class_id == class_id and meta.class_name:
                return meta.class_name
        return None

    def _objects_for_class(self, class_id: int) -> List[int]:
        return sorted(obj_id for obj_id, meta in self.object_meta.items() if meta.class_id == class_id)

    def _frame_objects_for_class(self, class_id: int) -> List[int]:
        frame_obj_ids = set()
        for obj_id, prompt in self._iter_frame_prompts(self.current_frame_idx):
            if self._class_for_obj(obj_id) == class_id and (prompt.points or prompt.box is not None):
                frame_obj_ids.add(obj_id)
        for obj_id, mask in self.results_by_frame.get(self.current_frame_idx, {}).items():
            if self._class_for_obj(obj_id) == class_id and binary_mask_to_bbox(mask) is not None:
                frame_obj_ids.add(obj_id)
        return sorted(frame_obj_ids)

    def _next_available_obj_id(self) -> int:
        if not self.object_meta:
            return 1
        return max(self.object_meta) + 1

    def _select_class(self, class_id: int) -> None:
        existing_obj_id = self._safe_int(self.current_obj_id.get(), "Object ID")
        if existing_obj_id in self.object_meta and self._class_for_obj(existing_obj_id) == class_id:
            selected_obj_id = existing_obj_id
        else:
            frame_obj_ids = self._frame_objects_for_class(class_id)
            if frame_obj_ids:
                selected_obj_id = frame_obj_ids[0]
            else:
                class_obj_ids = self._objects_for_class(class_id)
                selected_obj_id = class_obj_ids[0] if class_obj_ids else self._next_available_obj_id()
        self._sync_current_selection(selected_obj_id, class_id)
        self._set_status(f"Selected class {class_id} for obj {selected_obj_id}")
        self._render_frame()

    def _select_object(self, obj_id: int) -> None:
        class_id = self._class_for_obj(obj_id)
        self._sync_current_selection(obj_id, class_id)
        self.selected_pred_obj_id = obj_id
        self.selected_pred_frame_idx = self.current_frame_idx
        self._set_status(f"Selected object {obj_id} in class {class_id}")
        self._render_frame()

    def _on_select_all_class(self, _event) -> None:
        if self.all_classes_listbox is None:
            return
        selection = self.all_classes_listbox.curselection()
        if not selection:
            return
        self._select_class(self.all_class_ids[selection[0]])

    def _on_select_frame_class(self, _event) -> None:
        if self.frame_classes_listbox is None:
            return
        selection = self.frame_classes_listbox.curselection()
        if not selection:
            return
        self._select_class(self.frame_class_ids[selection[0]])

    def _on_select_frame_object(self, _event) -> None:
        if self.frame_objects_listbox is None:
            return
        selection = self.frame_objects_listbox.curselection()
        if not selection:
            return
        self._select_object(self.frame_object_ids[selection[0]])

    def _rename_class(self, class_id: int) -> None:
        current_name = self._class_name_for_id(class_id) or ""
        popup = tk.Toplevel(self.root)
        popup.title(f"Rename class {class_id}")
        popup.transient(self.root)
        popup.grab_set()
        popup.resizable(False, False)

        ttk.Label(popup, text=f"Class {class_id} name").pack(anchor=tk.W, padx=12, pady=(12, 4))
        name_var = tk.StringVar(value=current_name)
        entry = ttk.Entry(popup, textvariable=name_var, width=32)
        entry.pack(fill=tk.X, padx=12, pady=(0, 12))
        entry.focus_set()
        entry.selection_range(0, tk.END)

        def submit():
            new_name = name_var.get().strip() or None
            updated = False
            for meta in self.object_meta.values():
                if meta.class_id == class_id:
                    meta.class_name = new_name
                    updated = True
            if not updated:
                temp_obj_id = self._next_available_obj_id()
                self.object_meta[temp_obj_id] = ObjectMeta(class_id=class_id, class_name=new_name)
            popup.destroy()
            self._set_status(
                f"Updated class {class_id} name to `{new_name}`" if new_name else f"Cleared class {class_id} name"
            )
            self._refresh_class_lists()
            self._render_frame()

        buttons = ttk.Frame(popup)
        buttons.pack(fill=tk.X, padx=12, pady=(0, 12))
        ttk.Button(buttons, text="OK", command=submit).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Cancel", command=popup.destroy).pack(side=tk.RIGHT)
        popup.bind("<Return>", lambda _e: submit())
        popup.bind("<Escape>", lambda _e: popup.destroy())

    def _on_rename_all_class(self, _event) -> None:
        if self.all_classes_listbox is None:
            return
        selection = self.all_classes_listbox.curselection()
        if not selection:
            return
        self._rename_class(self.all_class_ids[selection[0]])

    def _get_or_create_prompt_state(self, obj_id: int, frame_idx: int) -> PromptState:
        return self.prompt_store.setdefault(obj_id, {}).setdefault(frame_idx, PromptState())

    def _current_prompt_state(self) -> PromptState:
        obj_id = self._safe_int(self.current_obj_id.get(), "Object ID")
        if obj_id not in self.object_meta:
            self._set_current_object()
        return self._get_or_create_prompt_state(obj_id, self.current_frame_idx)

    def _on_left_down(self, event) -> None:
        x, y = self._canvas_to_image_coords(event.x, event.y)
        mode = self.current_mode.get()
        if mode == "box":
            self.drag_start = (x, y)
            self.drag_current = (x, y)
            self._render_frame()
        else:
            label = 1 if mode == "positive" else 0
            prompt = self._current_prompt_state()
            prompt.points.append((x, y, label))
            self._set_status(
                f"Added {'positive' if label == 1 else 'negative'} point at ({int(x)}, {int(y)})"
            )
            self._render_frame()

    def _on_right_down(self, event) -> None:
        x, y = self._canvas_to_image_coords(event.x, event.y)
        hit_obj_id = self._find_prediction_at_point(self.current_frame_idx, x, y)
        if hit_obj_id is None:
            self.selected_pred_obj_id = None
            self.selected_pred_frame_idx = None
            self._set_status("No predicted bbox selected.")
            self._render_frame()
            return
        self.selected_pred_obj_id = hit_obj_id
        self.selected_pred_frame_idx = self.current_frame_idx
        self._sync_current_selection(hit_obj_id, self._class_for_obj(hit_obj_id))
        self._set_status(
            f"Selected predicted label: cls {self._class_for_obj(hit_obj_id)} / obj {hit_obj_id}. "
            "Use Delete to remove it, or add prompts to refine it."
        )
        self._render_frame()

    def _on_left_drag(self, event) -> None:
        if self.current_mode.get() != "box" or self.drag_start is None:
            return
        self.drag_current = self._canvas_to_image_coords(event.x, event.y)
        self._render_frame()

    def _on_left_up(self, event) -> None:
        if self.current_mode.get() != "box" or self.drag_start is None:
            return
        end = self._canvas_to_image_coords(event.x, event.y)
        x1, y1 = self.drag_start
        x2, y2 = end
        prompt = self._current_prompt_state()
        prompt.box = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.drag_start = None
        self.drag_current = None
        self._set_status("Box prompt updated.")
        self._render_frame()

    def _find_prediction_at_point(self, frame_idx: int, x: float, y: float) -> Optional[int]:
        candidates: List[Tuple[int, int]] = []
        for obj_id, mask in self.results_by_frame.get(frame_idx, {}).items():
            bbox = binary_mask_to_bbox(mask)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1 + 1) * (y2 - y1 + 1)
                candidates.append((area, obj_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> Tuple[float, float]:
        img_w, img_h = self.frame_provider.size
        disp_w, disp_h = self.display_size
        offset_x = max((self.canvas.winfo_width() - disp_w) // 2, 0)
        offset_y = max((self.canvas.winfo_height() - disp_h) // 2, 0)
        x = (canvas_x - offset_x) / self.display_scale
        y = (canvas_y - offset_y) / self.display_scale
        x = min(max(x, 0), img_w - 1)
        y = min(max(y, 0), img_h - 1)
        return x, y

    def _iter_frame_prompts(self, frame_idx: int):
        for obj_id, frame_prompts in self.prompt_store.items():
            prompt = frame_prompts.get(frame_idx)
            if prompt is not None:
                yield obj_id, prompt

    def _classes_in_frame(self, frame_idx: int) -> List[int]:
        class_ids = set()
        for obj_id, prompt in self._iter_frame_prompts(frame_idx):
            if prompt.points or prompt.box is not None:
                class_ids.add(self._class_for_obj(obj_id))
        for obj_id, mask in self.results_by_frame.get(frame_idx, {}).items():
            if binary_mask_to_bbox(mask) is not None:
                class_ids.add(self._class_for_obj(obj_id))
        return sorted(class_ids)

    def _objects_in_frame(self, frame_idx: int) -> List[int]:
        obj_ids = set()
        for obj_id, prompt in self._iter_frame_prompts(frame_idx):
            if prompt.points or prompt.box is not None:
                obj_ids.add(obj_id)
        for obj_id, mask in self.results_by_frame.get(frame_idx, {}).items():
            if binary_mask_to_bbox(mask) is not None:
                obj_ids.add(obj_id)
        return sorted(obj_ids)

    def _format_class_label(self, class_id: int, obj_ids: Sequence[int]) -> str:
        obj_text = ", ".join(str(obj_id) for obj_id in obj_ids)
        class_name = self._class_name_for_id(class_id)
        class_title = f"class {class_id}"
        if class_name:
            class_title += f" ({class_name})"
        return f"{class_title}    objs [{obj_text}]"

    def _format_object_label(self, obj_id: int) -> str:
        class_id = self._class_for_obj(obj_id)
        class_name = self._class_name_for_id(class_id)
        class_text = f"class {class_id}"
        if class_name:
            class_text += f" ({class_name})"
        return f"obj {obj_id}    {class_text}"

    def _set_listbox_color(self, listbox: tk.Listbox, index: int, class_id: int) -> None:
        color = "#%02x%02x%02x" % self._color_for_class(class_id)
        try:
            listbox.itemconfig(index, foreground=color)
        except tk.TclError:
            pass

    def _sync_listbox_selection(
        self,
        listbox: Optional[tk.Listbox],
        class_ids: Sequence[int],
        current_class_id: int,
    ) -> None:
        if listbox is None:
            return
        listbox.selection_clear(0, tk.END)
        if current_class_id in class_ids:
            idx = class_ids.index(current_class_id)
            listbox.selection_set(idx)
            listbox.see(idx)

    def _refresh_class_lists(self) -> None:
        try:
            current_class_id = self._safe_int(self.current_class_id.get(), "Class ID")
        except RuntimeError:
            current_class_id = 0

        all_class_ids = sorted({meta.class_id for meta in self.object_meta.values()})
        frame_class_ids = self._classes_in_frame(self.current_frame_idx)
        frame_object_ids = self._objects_in_frame(self.current_frame_idx)
        self.all_class_ids = all_class_ids
        self.frame_class_ids = frame_class_ids
        self.frame_object_ids = frame_object_ids

        if self.all_classes_listbox is not None:
            self.all_classes_listbox.delete(0, tk.END)
            for idx, class_id in enumerate(all_class_ids):
                label = self._format_class_label(class_id, self._objects_for_class(class_id))
                self.all_classes_listbox.insert(tk.END, label)
                self._set_listbox_color(self.all_classes_listbox, idx, class_id)
            self._sync_listbox_selection(self.all_classes_listbox, all_class_ids, current_class_id)

        if self.frame_classes_listbox is not None:
            self.frame_classes_listbox.delete(0, tk.END)
            for idx, class_id in enumerate(frame_class_ids):
                label = self._format_class_label(class_id, self._frame_objects_for_class(class_id))
                self.frame_classes_listbox.insert(tk.END, label)
                self._set_listbox_color(self.frame_classes_listbox, idx, class_id)
            self._sync_listbox_selection(self.frame_classes_listbox, frame_class_ids, current_class_id)

        if self.frame_objects_listbox is not None:
            self.frame_objects_listbox.delete(0, tk.END)
            for idx, obj_id in enumerate(frame_object_ids):
                label = self._format_object_label(obj_id)
                self.frame_objects_listbox.insert(tk.END, label)
                self._set_listbox_color(self.frame_objects_listbox, idx, self._class_for_obj(obj_id))
            self.frame_objects_listbox.selection_clear(0, tk.END)
            if self.selected_pred_frame_idx == self.current_frame_idx and self.selected_pred_obj_id in frame_object_ids:
                selected_idx = frame_object_ids.index(self.selected_pred_obj_id)
                self.frame_objects_listbox.selection_set(selected_idx)
                self.frame_objects_listbox.see(selected_idx)

    def _render_frame(self) -> None:
        self.frame_provider.preload_neighbors(self.current_frame_idx)
        frame = self.frame_provider.get_frame(self.current_frame_idx).convert("RGBA")
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for obj_id, frame_results in self.results_by_frame.get(self.current_frame_idx, {}).items():
            bbox = binary_mask_to_bbox(frame_results)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            color = self._color_for_class(self._class_for_obj(obj_id))
            is_selected = (
                self.selected_pred_obj_id == obj_id
                and self.selected_pred_frame_idx == self.current_frame_idx
            )
            draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=5 if is_selected else 3)
            if is_selected:
                draw.rectangle((x1 - 2, y1 - 2, x2 + 2, y2 + 2), outline=(255, 255, 255, 255), width=2)
            draw.text((x1 + 3, y1 + 3), f"cls {self._class_for_obj(obj_id)} / obj {obj_id}", fill=color + (255,))

        current_obj_id: Optional[int] = None
        try:
            current_obj_id = self._safe_int(self.current_obj_id.get(), "Object ID")
        except RuntimeError:
            current_obj_id = None

        for obj_id, prompt in self._iter_frame_prompts(self.current_frame_idx):
            color = self._color_for_class(self._class_for_obj(obj_id))
            if prompt.box is not None:
                width = 3 if current_obj_id == obj_id else 2
                draw.rectangle(prompt.box, outline=color + (255,), width=width)
            for x, y, label in prompt.points:
                r = 7 if current_obj_id == obj_id else 5
                point_color = (0, 255, 0, 255) if label == 1 else (255, 0, 0, 255)
                draw.ellipse((x - r, y - r, x + r, y + r), fill=point_color, outline=(255, 255, 255, 255))

        if self.drag_start is not None and self.drag_current is not None:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_current
            draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0, 255), width=2)

        composed = Image.alpha_composite(frame, overlay).convert("RGB")
        composed = self._fit_image_to_canvas(composed)
        self.tk_image = ImageTk.PhotoImage(composed)
        self.canvas.delete("all")
        self.canvas.create_image(
            max((self.canvas.winfo_width() - composed.width) // 2, 0),
            max((self.canvas.winfo_height() - composed.height) // 2, 0),
            image=self.tk_image,
            anchor=tk.NW,
        )
        self.frame_entry.set(str(self.current_frame_idx))
        self._refresh_class_lists()
        self.root.title(
            f"SAM2 Interactive Video Annotator - frame {self.current_frame_idx + 1}/{self.frame_count}"
        )

    def _fit_image_to_canvas(self, image: Image.Image) -> Image.Image:
        canvas_w = max(self.canvas.winfo_width(), MAX_CANVAS_W)
        canvas_h = max(self.canvas.winfo_height(), MAX_CANVAS_H)
        img_w, img_h = image.size
        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        disp_w = max(int(img_w * scale), 1)
        disp_h = max(int(img_h * scale), 1)
        self.display_scale = scale
        self.display_size = (disp_w, disp_h)
        if scale == 1.0:
            return image
        return image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)

    def _color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        palette = [
            (255, 190, 11),
            (251, 86, 7),
            (255, 0, 110),
            (131, 56, 236),
            (58, 134, 255),
            (46, 196, 182),
            (233, 196, 106),
            (244, 162, 97),
        ]
        return palette[class_id % len(palette)]

    def _prev_frame(self) -> None:
        self.current_frame_idx = max(self.current_frame_idx - 1, 0)
        self._render_frame()

    def _next_frame(self) -> None:
        self.current_frame_idx = min(self.current_frame_idx + 1, self.frame_count - 1)
        self._render_frame()

    def _jump_to_frame(self) -> None:
        idx = self._safe_int(self.frame_entry.get(), "Frame index")
        idx = min(max(idx, 0), self.frame_count - 1)
        self.current_frame_idx = idx
        self._render_frame()

    def _apply_current_prompt(self) -> None:
        self._run_prompt(apply_mode="apply")

    def _update_memory(self) -> None:
        obj_id = self._safe_int(self.current_obj_id.get(), "Object ID")
        if obj_id in self.object_meta:
            should_continue = messagebox.askyesno(
                "Object ID already exists",
                f"Object ID {obj_id} already exists.\n\n"
                "If you are introducing a brand-new target, it is safer to use a new Object ID.\n"
                "If you continue, the new prompt will be added to the existing object.",
                parent=self.root,
            )
            if not should_continue:
                self._set_status("Update Memory cancelled. Please choose a new Object ID for the new target.")
                return
        self._run_prompt(apply_mode="update_memory")

    def _run_prompt(self, apply_mode: str) -> None:
        obj_id = self._safe_int(self.current_obj_id.get(), "Object ID")
        if obj_id not in self.object_meta:
            self._set_current_object()
        prompt = self._get_or_create_prompt_state(obj_id, self.current_frame_idx)
        if not prompt.points and prompt.box is None:
            raise RuntimeError("Current object has no point or box prompt on this frame.")

        points = None
        labels = None
        if prompt.points:
            points = np.asarray([[x, y] for x, y, _ in prompt.points], dtype=np.float32)
            labels = np.asarray([label for _, _, label in prompt.points], dtype=np.int32)
        box = None
        if prompt.box is not None:
            box = np.asarray(prompt.box, dtype=np.float32)

        action_text = "Updating memory" if apply_mode == "update_memory" else "Applying prompt"
        self._set_status(f"{action_text} on frame {self.current_frame_idx} for obj {obj_id}...")
        with self.inference_context_factory():
            _frame_idx, obj_ids, video_res_masks = self.predictor.add_new_points_or_box(
                inference_state=self.state,
                frame_idx=self.current_frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                clear_old_points=True,
                normalize_coords=True,
                box=box,
            )
        current_frame_results: Dict[int, np.ndarray] = {}
        for idx, out_obj_id in enumerate(obj_ids):
            mask = mask_tensor_to_numpy(video_res_masks[idx]) > self.mask_threshold
            current_frame_results[int(out_obj_id)] = mask.astype(np.uint8)
        existing_frame_results = self.results_by_frame.get(self.current_frame_idx, {}).copy()
        existing_frame_results.update(current_frame_results)
        self.results_by_frame[self.current_frame_idx] = existing_frame_results
        self.selected_pred_obj_id = obj_id
        self.selected_pred_frame_idx = self.current_frame_idx
        self._trim_inference_state(current_frame_idx=self.current_frame_idx)
        if apply_mode == "update_memory":
            self._set_status(
                f"Memory updated on frame {self.current_frame_idx} for obj {obj_id}. "
                "It will participate in the next propagation."
            )
        else:
            self._set_status(
                f"Prompt applied on frame {self.current_frame_idx}. Existing annotations were kept."
            )
        self._render_frame()

    def _clear_current_prompt(self) -> None:
        obj_id = self._safe_int(self.current_obj_id.get(), "Object ID")
        if obj_id in self.prompt_store:
            self.prompt_store[obj_id].pop(self.current_frame_idx, None)
        try:
            with self.inference_context_factory():
                self.predictor.clear_all_prompts_in_frame(
                    self.state,
                    frame_idx=self.current_frame_idx,
                    obj_id=obj_id,
                    need_output=False,
                )
        except Exception:
            pass
        frame_results = self.results_by_frame.get(self.current_frame_idx)
        if frame_results is not None:
            frame_results.pop(obj_id, None)
            if not frame_results:
                self.results_by_frame.pop(self.current_frame_idx, None)
        if self.selected_pred_obj_id == obj_id and self.selected_pred_frame_idx == self.current_frame_idx:
            self.selected_pred_obj_id = None
            self.selected_pred_frame_idx = None
        self._set_status(f"Cleared prompts for obj {obj_id} on frame {self.current_frame_idx}.")
        self._render_frame()

    def _delete_current_frame_label(self, obj_id: int) -> None:
        frame_idx = self.current_frame_idx

        has_prompt = (
            obj_id in self.prompt_store
            and frame_idx in self.prompt_store[obj_id]
            and (
                self.prompt_store[obj_id][frame_idx].points
                or self.prompt_store[obj_id][frame_idx].box is not None
            )
        )
        if has_prompt:
            self._sync_current_selection(obj_id, self._class_for_obj(obj_id))
            self._clear_current_prompt()
            return

        frame_results = self.results_by_frame.get(frame_idx)
        if frame_results is not None:
            frame_results.pop(obj_id, None)
            if not frame_results:
                self.results_by_frame.pop(frame_idx, None)

        obj_idx = self.state.get("obj_id_to_idx", {}).get(obj_id)
        if obj_idx is not None:
            obj_output_dict = self.state.get("output_dict_per_obj", {}).get(obj_idx, {})
            obj_output_dict.get("cond_frame_outputs", {}).pop(frame_idx, None)
            obj_output_dict.get("non_cond_frame_outputs", {}).pop(frame_idx, None)
            obj_temp_output_dict = self.state.get("temp_output_dict_per_obj", {}).get(obj_idx, {})
            obj_temp_output_dict.get("cond_frame_outputs", {}).pop(frame_idx, None)
            obj_temp_output_dict.get("non_cond_frame_outputs", {}).pop(frame_idx, None)
            self.state.get("point_inputs_per_obj", {}).get(obj_idx, {}).pop(frame_idx, None)
            self.state.get("mask_inputs_per_obj", {}).get(obj_idx, {}).pop(frame_idx, None)
            self.state.get("frames_tracked_per_obj", {}).get(obj_idx, {}).pop(frame_idx, None)

        if self.selected_pred_obj_id == obj_id and self.selected_pred_frame_idx == frame_idx:
            self.selected_pred_obj_id = None
            self.selected_pred_frame_idx = None

        self._set_status(f"Deleted current-frame label for obj {obj_id}.")
        self._render_frame()

    def _on_delete_selected(self, _event=None) -> None:
        if self.selected_pred_obj_id is None or self.selected_pred_frame_idx != self.current_frame_idx:
            self._set_status("No selected predicted label on the current frame.")
            return
        self._delete_current_frame_label(self.selected_pred_obj_id)

    def _earliest_seed_frame(self) -> int:
        frame_indices = [
            frame_idx
            for _obj_id, frames in self.prompt_store.items()
            for frame_idx, prompt in frames.items()
            if prompt.points or prompt.box is not None
        ]
        if not frame_indices:
            raise RuntimeError("No prompts have been added yet.")
        return min(frame_indices)

    def _propagate_all(self) -> None:
        start_frame_idx = self._earliest_seed_frame()
        self._set_status("Propagating forward...")
        forward_results = self._collect_propagation_results(
            start_frame_idx=start_frame_idx,
            reverse=False,
            max_frame_num_to_track=None,
        )
        results = dict(forward_results)
        if start_frame_idx > 0:
            self._set_status("Propagating backward...")
            backward_results = self._collect_propagation_results(
                start_frame_idx=start_frame_idx,
                reverse=True,
                max_frame_num_to_track=None,
            )
            results.update(backward_results)
        self.results_by_frame = results
        if self.selected_pred_frame_idx is not None and self.selected_pred_frame_idx != self.current_frame_idx:
            self.selected_pred_obj_id = None
            self.selected_pred_frame_idx = None
        self._trim_inference_state(current_frame_idx=self.current_frame_idx)
        self._set_status("Propagation finished.")
        self._render_frame()

    def _propagate_range(self) -> None:
        start_frame_idx = self.current_frame_idx
        target_frame_idx = self._safe_int(self.target_frame_entry.get(), "Target frame")
        target_frame_idx = min(max(target_frame_idx, 0), self.frame_count - 1)
        reverse = target_frame_idx < start_frame_idx
        max_frame_num_to_track = abs(target_frame_idx - start_frame_idx) + 1

        direction_text = "backward" if reverse else "forward"
        self._set_status(
            f"Propagating {direction_text} from frame {start_frame_idx} to frame {target_frame_idx}..."
        )
        range_results = self._collect_propagation_results(
            start_frame_idx=start_frame_idx,
            reverse=reverse,
            max_frame_num_to_track=max_frame_num_to_track,
        )
        self.results_by_frame.update(range_results)
        self._trim_inference_state(current_frame_idx=target_frame_idx)
        self._set_status(f"Range propagation finished: {start_frame_idx} -> {target_frame_idx}")
        self._render_frame()

    def _collect_propagation_results(
        self,
        start_frame_idx: int,
        reverse: bool,
        max_frame_num_to_track: Optional[int],
    ) -> Dict[int, Dict[int, np.ndarray]]:
        collected: Dict[int, Dict[int, np.ndarray]] = {}
        with self.inference_context_factory():
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.state,
                start_frame_idx=start_frame_idx,
                reverse=reverse,
                max_frame_num_to_track=max_frame_num_to_track,
            ):
                per_obj: Dict[int, np.ndarray] = {}
                for idx, obj_id in enumerate(out_obj_ids):
                    mask = mask_tensor_to_numpy(out_mask_logits[idx]) > self.mask_threshold
                    per_obj[int(obj_id)] = mask.astype(np.uint8)
                collected[int(out_frame_idx)] = per_obj
                self._trim_inference_state(current_frame_idx=int(out_frame_idx))
                self._set_status(f"Propagating... frame {out_frame_idx}")
                self.root.update_idletasks()
        return collected

    def _trim_inference_state(self, current_frame_idx: int) -> None:
        self._trim_non_condition_memory(current_frame_idx)
        self._trim_cached_features(current_frame_idx)

    def _trim_non_condition_memory(self, current_frame_idx: int) -> None:
        for obj_output_dict in self.state.get("output_dict_per_obj", {}).values():
            non_cond = obj_output_dict.get("non_cond_frame_outputs", {})
            if len(non_cond) <= self.memory_window:
                continue
            keep_keys = sorted(non_cond.keys(), key=lambda frame_idx: abs(frame_idx - current_frame_idx))[: self.memory_window]
            keep_set = set(keep_keys)
            for frame_idx in list(non_cond.keys()):
                if frame_idx not in keep_set:
                    non_cond.pop(frame_idx, None)

        for obj_frames_tracked in self.state.get("frames_tracked_per_obj", {}).values():
            cond_frames = set()
            obj_idx = None
            for maybe_idx, mapping in self.state.get("frames_tracked_per_obj", {}).items():
                if mapping is obj_frames_tracked:
                    obj_idx = maybe_idx
                    break
            if obj_idx is not None:
                cond_frames = set(
                    self.state["output_dict_per_obj"][obj_idx].get("cond_frame_outputs", {}).keys()
                )
            non_cond_tracked = [
                frame_idx for frame_idx in obj_frames_tracked.keys() if frame_idx not in cond_frames
            ]
            if len(non_cond_tracked) <= self.memory_window:
                continue
            keep_keys = set(
                sorted(non_cond_tracked, key=lambda frame_idx: abs(frame_idx - current_frame_idx))[: self.memory_window]
            )
            for frame_idx in list(obj_frames_tracked.keys()):
                if frame_idx in cond_frames:
                    continue
                if frame_idx not in keep_keys:
                    obj_frames_tracked.pop(frame_idx, None)

    def _trim_cached_features(self, current_frame_idx: int) -> None:
        cached_features = self.state.get("cached_features", {})
        if len(cached_features) <= self.memory_window:
            return
        keep_keys = set(
            sorted(cached_features.keys(), key=lambda frame_idx: abs(frame_idx - current_frame_idx))[: self.memory_window]
        )
        for frame_idx in list(cached_features.keys()):
            if frame_idx not in keep_keys:
                cached_features.pop(frame_idx, None)

    def _export_yolo(self) -> None:
        if not self.results_by_frame:
            raise RuntimeError("No propagated results to export. Run `Propagate Whole Video` first.")
        labels_dir = self.output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_width, image_height = self.frame_provider.size
        for frame_idx in range(self.frame_count):
            lines: List[str] = []
            frame_results = self.results_by_frame.get(frame_idx, {})
            for obj_id, mask in sorted(frame_results.items()):
                bbox = binary_mask_to_bbox(mask)
                if bbox is None:
                    continue
                class_id = self.object_meta.get(obj_id, ObjectMeta(class_id=0)).class_id
                lines.append(bbox_to_yolo_line(bbox, class_id, image_width, image_height))
            (labels_dir / f"{self.frame_provider.frame_name(frame_idx)}.txt").write_text(
                "\n".join(lines),
                encoding="utf-8",
            )

        class_lines = []
        all_class_ids = sorted({meta.class_id for meta in self.object_meta.values()})
        for class_id in all_class_ids:
            class_name = self._class_name_for_id(class_id) or f"class_{class_id}"
            class_lines.append(class_name)
        (self.output_dir / "classes.txt").write_text("\n".join(class_lines), encoding="utf-8")
        self._set_status(f"YOLO labels exported to {labels_dir}")
        messagebox.showinfo("Export complete", f"YOLO labels written to:\n{labels_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive SAM2 video annotation GUI.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", type=Path, help="Input mp4 video path.")
    source_group.add_argument("--frames-dir", type=Path, help="Input frame directory.")
    parser.add_argument("--model-cfg", required=True, help="SAM2 config yaml path.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="SAM2 checkpoint path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir for YOLO labels.")
    parser.add_argument("--device", default=None, help="torch device, default cuda then cpu.")
    parser.add_argument("--mask-threshold", type=float, default=0.0, help="Mask threshold for bbox export.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth frame when extracting from mp4. 1 means no skipping.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=95,
        help="JPEG quality used for extracted frames.",
    )
    parser.add_argument(
        "--ui-cache-size",
        type=int,
        default=5,
        help="Number of decoded UI frames to keep in the in-memory LRU cache.",
    )
    parser.add_argument(
        "--memory-window",
        type=int,
        default=5,
        help="Keep only this many non-conditioning frames in SAM2 memory.",
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to offload SAM2 loaded video frames to CPU memory. Default: enabled.",
    )
    parser.add_argument(
        "--offload-state-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to offload SAM2 tracking state to CPU memory. Default: enabled.",
    )
    parser.add_argument(
        "--async-loading-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether SAM2 should lazily load JPEG frames instead of preloading them all. Default: enabled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise RuntimeError(f"Checkpoint not found: {args.checkpoint}")

    if args.video:
        video_path = args.video.resolve()
        if not video_path.exists():
            raise RuntimeError(f"Video not found: {video_path}")
        extracted_frames_dir = output_dir / "images"
        frame_paths = extract_frames_from_video(
            video_path=video_path,
            frames_dir=extracted_frames_dir,
            frame_stride=args.frame_stride,
            jpg_quality=args.jpg_quality,
        )
        source_path = str(extracted_frames_dir)
    else:
        frames_dir = args.frames_dir.resolve()
        if not frames_dir.exists():
            raise RuntimeError(f"Frames dir not found: {frames_dir}")
        frame_paths = list_frame_paths(frames_dir)
        source_path = str(frames_dir)
    frame_provider = FrameProvider(frame_paths=frame_paths, cache_size=args.ui_cache_size)

    device = infer_device(args.device)
    inference_context_factory = lambda: build_inference_context(device)
    with inference_context_factory():
        patch_sam2_jpg_loader_for_prefixed_names()
        predictor = build_predictor(args.model_cfg, args.checkpoint.resolve(), device=device)
        inference_state = predictor.init_state(
            video_path=source_path,
            offload_video_to_cpu=args.offload_video_to_cpu,
            offload_state_to_cpu=args.offload_state_to_cpu,
            async_loading_frames=args.async_loading_frames,
        )

    root = tk.Tk()
    app = InteractiveAnnotator(
        root=root,
        predictor=predictor,
        inference_state=inference_state,
        source_path=source_path,
        output_dir=output_dir,
        frame_provider=frame_provider,
        mask_threshold=args.mask_threshold,
        inference_context_factory=inference_context_factory,
        memory_window=args.memory_window,
    )
    root.after(100, app._render_frame)
    root.mainloop()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
