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
import sys
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
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    saved_paths: List[Path] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride == 0:
                out_path = frames_dir / f"{frame_idx:06d}.jpg"
                success = cv2.imwrite(
                    str(out_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)],
                )
                if not success:
                    raise RuntimeError(f"Failed to write extracted frame: {out_path}")
                saved_paths.append(out_path)
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


class InteractiveAnnotator:
    def __init__(
        self,
        root: tk.Tk,
        predictor,
        inference_state,
        source_path: str,
        output_dir: Path,
        frames: Sequence[Image.Image],
        frame_names: Sequence[str],
        mask_threshold: float,
        inference_context_factory,
    ):
        self.root = root
        self.predictor = predictor
        self.state = inference_state
        self.source_path = source_path
        self.output_dir = output_dir
        self.frames = list(frames)
        self.frame_names = list(frame_names)
        self.mask_threshold = mask_threshold
        self.inference_context_factory = inference_context_factory

        self.frame_count = len(self.frames)
        self.current_frame_idx = 0
        self.current_mode = tk.StringVar(value="positive")
        self.current_obj_id = tk.StringVar(value="1")
        self.current_class_id = tk.StringVar(value="0")
        self.status_var = tk.StringVar(value="Ready")
        self.all_classes_listbox: Optional[tk.Listbox] = None
        self.frame_classes_listbox: Optional[tk.Listbox] = None
        self.all_class_ids: List[int] = []
        self.frame_class_ids: List[int] = []

        self.prompt_store: Dict[int, Dict[int, PromptState]] = {}
        self.object_meta: Dict[int, ObjectMeta] = {}
        self.results_by_frame: Dict[int, Dict[int, np.ndarray]] = {}
        self.drag_start: Optional[Tuple[float, float]] = None
        self.drag_current: Optional[Tuple[float, float]] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.display_scale = 1.0
        self.display_size = self.frames[0].size

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
        ttk.Button(left, text="Clear Current Obj Prompt On Frame", command=self._clear_current_prompt).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Propagate Whole Video", command=self._propagate_all).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Export YOLO", command=self._export_yolo).pack(fill=tk.X, pady=2)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Button(left, text="Prev Frame", command=self._prev_frame).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Next Frame", command=self._next_frame).pack(fill=tk.X, pady=2)

        self.frame_entry = tk.StringVar(value="0")
        ttk.Label(left, text="Jump To Frame").pack(anchor=tk.W, pady=(8, 0))
        ttk.Entry(left, textvariable=self.frame_entry, width=12).pack(anchor=tk.W, pady=(0, 4))
        ttk.Button(left, text="Go", command=self._jump_to_frame).pack(fill=tk.X, pady=2)

        tips = (
            "Left click: add point in current mode\n"
            "Box mode: press and drag\n"
            "Apply prompt: run SAM2 on current frame\n"
            "Propagate: track through the video\n"
            "Export YOLO: write txt labels"
        )
        ttk.Label(left, text=tips, wraplength=240, justify=tk.LEFT).pack(anchor=tk.W, pady=(12, 0))

        self.canvas = tk.Canvas(center, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)

        ttk.Label(class_panel, text="All Classes").pack(anchor=tk.W)
        self.all_classes_listbox = tk.Listbox(class_panel, exportselection=False, height=14)
        self.all_classes_listbox.pack(fill=tk.BOTH, expand=False, pady=(4, 10))
        self.all_classes_listbox.bind("<<ListboxSelect>>", self._on_select_all_class)

        ttk.Label(class_panel, text="Classes In Current Frame").pack(anchor=tk.W)
        self.frame_classes_listbox = tk.Listbox(class_panel, exportselection=False, height=14)
        self.frame_classes_listbox.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.frame_classes_listbox.bind("<<ListboxSelect>>", self._on_select_frame_class)

        bottom = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        bottom.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status_var).pack(anchor=tk.W)

        self.root.bind("<Left>", lambda _e: self._prev_frame())
        self.root.bind("<Right>", lambda _e: self._next_frame())

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

    def _canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> Tuple[float, float]:
        img_w, img_h = self.frames[self.current_frame_idx].size
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

    def _format_class_label(self, class_id: int, obj_ids: Sequence[int]) -> str:
        obj_text = ", ".join(str(obj_id) for obj_id in obj_ids)
        return f"class {class_id}    objs [{obj_text}]"

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
        self.all_class_ids = all_class_ids
        self.frame_class_ids = frame_class_ids

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

    def _render_frame(self) -> None:
        frame = self.frames[self.current_frame_idx].copy().convert("RGBA")
        overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for obj_id, frame_results in self.results_by_frame.get(self.current_frame_idx, {}).items():
            bbox = binary_mask_to_bbox(frame_results)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            color = self._color_for_class(self._class_for_obj(obj_id))
            draw.rectangle((x1, y1, x2, y2), outline=color + (255,), width=3)
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

        self._set_status(f"Applying prompt on frame {self.current_frame_idx} for obj {obj_id}...")
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
        self.results_by_frame[self.current_frame_idx] = current_frame_results
        self._set_status(f"Prompt applied on frame {self.current_frame_idx}.")
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
        self._set_status(f"Cleared prompts for obj {obj_id} on frame {self.current_frame_idx}.")
        self._render_frame()

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
        forward_results = self._collect_propagation_results(start_frame_idx=start_frame_idx, reverse=False)
        results = dict(forward_results)
        if start_frame_idx > 0:
            self._set_status("Propagating backward...")
            backward_results = self._collect_propagation_results(start_frame_idx=start_frame_idx, reverse=True)
            results.update(backward_results)
        self.results_by_frame = results
        self._set_status("Propagation finished.")
        self._render_frame()

    def _collect_propagation_results(self, start_frame_idx: int, reverse: bool) -> Dict[int, Dict[int, np.ndarray]]:
        collected: Dict[int, Dict[int, np.ndarray]] = {}
        with self.inference_context_factory():
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.state,
                start_frame_idx=start_frame_idx,
                reverse=reverse,
            ):
                per_obj: Dict[int, np.ndarray] = {}
                for idx, obj_id in enumerate(out_obj_ids):
                    mask = mask_tensor_to_numpy(out_mask_logits[idx]) > self.mask_threshold
                    per_obj[int(obj_id)] = mask.astype(np.uint8)
                collected[int(out_frame_idx)] = per_obj
                self._set_status(f"Propagating... frame {out_frame_idx}")
                self.root.update_idletasks()
        return collected

    def _export_yolo(self) -> None:
        if not self.results_by_frame:
            raise RuntimeError("No propagated results to export. Run `Propagate Whole Video` first.")
        labels_dir = self.output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        image_width, image_height = self.frames[0].size
        for frame_idx in range(self.frame_count):
            lines: List[str] = []
            frame_results = self.results_by_frame.get(frame_idx, {})
            for obj_id, mask in sorted(frame_results.items()):
                bbox = binary_mask_to_bbox(mask)
                if bbox is None:
                    continue
                class_id = self.object_meta.get(obj_id, ObjectMeta(class_id=0)).class_id
                lines.append(bbox_to_yolo_line(bbox, class_id, image_width, image_height))
            (labels_dir / f"{self.frame_names[frame_idx]}.txt").write_text(
                "\n".join(lines),
                encoding="utf-8",
            )

        meta_lines = []
        for obj_id, meta in sorted(self.object_meta.items()):
            meta_lines.append(f"obj_id={obj_id}, class_id={meta.class_id}")
        (self.output_dir / "objects.txt").write_text("\n".join(meta_lines), encoding="utf-8")
        frame_map_lines = [
            f"{idx}\t{frame_name}"
            for idx, frame_name in enumerate(self.frame_names)
        ]
        (self.output_dir / "frame_index_map.txt").write_text(
            "\n".join(frame_map_lines),
            encoding="utf-8",
        )
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
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--offload-state-to-cpu", action="store_true")
    parser.add_argument("--async-loading-frames", action="store_true")
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
        extracted_frames_dir = output_dir / "frames"
        frame_paths = extract_frames_from_video(
            video_path=video_path,
            frames_dir=extracted_frames_dir,
            frame_stride=args.frame_stride,
            jpg_quality=args.jpg_quality,
        )
        display_frames = [Image.open(path).convert("RGB") for path in frame_paths]
        source_path = str(extracted_frames_dir)
    else:
        frames_dir = args.frames_dir.resolve()
        if not frames_dir.exists():
            raise RuntimeError(f"Frames dir not found: {frames_dir}")
        frame_paths = list_frame_paths(frames_dir)
        display_frames = [Image.open(path).convert("RGB") for path in frame_paths]
        source_path = str(frames_dir)

    frame_names = [path.stem for path in frame_paths]

    device = infer_device(args.device)
    inference_context_factory = lambda: build_inference_context(device)
    with inference_context_factory():
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
        frames=display_frames,
        frame_names=frame_names,
        mask_threshold=args.mask_threshold,
        inference_context_factory=inference_context_factory,
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
