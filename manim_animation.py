"""A ten-minute visual lecture on the unified 192-D occupancy / parameter /
spectrum JEPA (architecture_v5.md §3-§5).

The film is one continuous argument, not a slide deck: a real test sample's
spectrum becomes a geometry; the geometry is factored out of the release's
three-channel packing; masking introduces the learning problem; the JEPA
student/teacher idea is built before the architecture; the architecture is
assembled lane by lane; the objective and the physics loop close the loop; and
the measured results are shown one experiment at a time.

Timing is driven by the measured narration. `scripts/presentation/build_narration.py`
synthesises `docs/presentation/narration.md`, measures every sentence, and writes
`docs/presentation/timeline.json`. Each scene calls `self.cue(i)` at the moment
sentence i is being spoken, so an equation appears when it is discussed rather
than at a guessed time. With no timeline present the film still renders, using
nominal per-scene beats.

Preview one scene:
    python -m manim -ql manim_animation.py S05Architecture
Full film, one file:
    python -m manim -qh --fps 30 manim_animation.py JEPAProjectPresentation
The published film is assembled per scene (see docs/presentation/README_RENDER.md),
which is what keeps the voice track frame-aligned.

There is no LaTeX on this machine, so mathematical typography is rendered with
Cambria Math through Pango markup (`math()` converts `_{...}` / `^{...}` to real
sub/superscripts). Every dimension drawn here is taken from the live code and
`configs/unified.yaml`, not from the older design documents.
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import numpy as np
from manim import *
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# Paths and data
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
PRESENTATION = ROOT / "docs" / "presentation"
TIMELINE_JSON = PRESENTATION / "timeline.json"
TEST_MAT = ROOT / "data" / "metadit" / "split_data" / "test_set.mat"

SAMPLE_INDEX = 11749
SCENE_ORDER = ["hook", "inverse", "dataset", "jepa", "architecture",
               "objective", "physics", "training", "scenarios", "results",
               "transfer", "code", "closing"]

_DATA = None


def load_data():
    """The staged test split: one real sample, its nearest-spectrum neighbour,
    and the split counts. Loaded once per process."""
    global _DATA
    if _DATA is not None:
        return _DATA
    if not TEST_MAT.exists():
        raise FileNotFoundError(
            f"presentation needs the staged test split at {TEST_MAT}")
    d = loadmat(TEST_MAT)
    pattern = np.asarray(d["pattern"][:, :, SAMPLE_INDEX]).astype(np.float32)
    spec = np.stack([np.asarray(d["real"][SAMPLE_INDEX], dtype=np.float64),
                     np.asarray(d["imag"][SAMPLE_INDEX], dtype=np.float64)])

    all_spec = np.concatenate([np.asarray(d["real"], dtype=np.float64),
                               np.asarray(d["imag"], dtype=np.float64)], axis=1)
    dist = np.abs(all_spec - all_spec[SAMPLE_INDEX]).mean(axis=1)
    dist[SAMPLE_INDEX] = np.inf
    neighbour = int(np.argmin(dist))
    n_pattern = np.asarray(d["pattern"][:, :, neighbour]).astype(np.float32)
    n_spec = np.stack([np.asarray(d["real"][neighbour], dtype=np.float64),
                       np.asarray(d["imag"][neighbour], dtype=np.float64)])
    scale = float(np.abs(all_spec[SAMPLE_INDEX]).mean())
    inter = float(((pattern > 0.5) & (n_pattern > 0.5)).sum())
    union = float(((pattern > 0.5) | (n_pattern > 0.5)).sum())

    _DATA = {
        "pattern": pattern,
        "scalars": np.asarray(d["parameter"][SAMPLE_INDEX], dtype=np.float64),
        "spectrum": spec,
        "n_test": int(np.asarray(d["pattern"]).shape[-1]),
        "n_train": 139906,
        "n_val": 17488,
        "n_total": 174883,
        "neighbour_pattern": n_pattern,
        "neighbour_spectrum": n_spec,
        "neighbour_index": neighbour,
        "spectrum_distance": float(dist[neighbour] / scale),
        "neighbour_iou": inter / max(1.0, union),
    }
    return _DATA


# ---------------------------------------------------------------------------
# Palette, type
# ---------------------------------------------------------------------------

BG = "#070E1A"
PANEL = "#0D1830"
PANEL_HI = "#132443"
FAINT = "#2B3A55"
INK = "#F4F7FF"
MUTED = "#A9B6CC"

BLUE = "#5CA8FF"      # occupancy / geometry
CYAN = "#56D9EE"      # spectrum
YELLOW = "#FFD166"    # scalar parameters + FiLM
PURPLE = "#B49BFF"    # latent representations
VIOLET = "#8E72FF"    # teacher targets
GREEN = "#68E29A"     # student / trainable
CORAL = "#FF8294"     # masked / missing
FROZEN = "#7FA6B8"    # frozen pretrained
ORANGE = "#FFAA70"    # physics / surrogate

FONT = "Segoe UI"
MATH_FONT = "Cambria Math"

config.background_color = BG


def _escape(value: str) -> str:
    return (str(value).replace("&", "&amp;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _markup(value: str) -> str:
    """`_{x}` / `^{x}` -> real Pango sub/superscripts; everything else escaped."""
    out, i = [], 0
    while i < len(value):
        c = value[i]
        if c in "_^" and i + 1 < len(value) and value[i + 1] == "{":
            tag = "sub" if c == "_" else "sup"
            close = value.index("}", i)
            out.append(f"<{tag}>{_escape(value[i + 2:close])}</{tag}>")
            i = close + 1
        else:
            out.append(_escape(c))
            i += 1
    return "".join(out)


def math(expr, size=36, color=INK, weight=NORMAL):
    """Mathematical line with real sub/superscripts (no LaTeX required)."""
    return MarkupText(_markup(expr), font=MATH_FONT, font_size=size,
                      color=color, weight=weight)


def tx(value, size=28, color=INK, weight=NORMAL, wrap=None, line_spacing=0.95):
    if wrap:
        value = "\n".join(textwrap.wrap(str(value), wrap))
    return MarkupText(_escape(value), font=FONT, font_size=size, color=color,
                      weight=weight, line_spacing=line_spacing)


def eyebrow(scene_no, text_value):
    left = tx(f"{scene_no:02d}", 22, CYAN, BOLD)
    right = tx(text_value.upper(), 20, MUTED, MEDIUM)
    return VGroup(left, right).arrange(RIGHT, buff=0.22, aligned_edge=DOWN)


def heading(number, name):
    group = VGroup(eyebrow(number, name))
    group.to_corner(UL, buff=0.42)
    return group


def title_line(text_value, size=52):
    return tx(text_value, size, INK, BOLD)


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def panel(title, detail="", color=FAINT, w=1.5, h=0.8, title_size=22,
          detail_size=17, fill=PANEL, dashed=False):
    box = RoundedRectangle(width=w, height=h, corner_radius=0.08,
                           stroke_color=color, stroke_width=1.6,
                           fill_color=fill, fill_opacity=0.96)
    if dashed:
        box = DashedVMobject(box, num_dashes=26)
    rows = [tx(title, title_size, color, BOLD)]
    if detail:
        rows.append(tx(detail, detail_size, MUTED))
    labels = VGroup(*rows).arrange(DOWN, buff=0.06)
    if labels.width > w - 0.16:
        labels.scale_to_fit_width(w - 0.16)
    if labels.height > h - 0.12:
        labels.scale_to_fit_height(h - 0.12)
    labels.move_to(box)
    return VGroup(box, labels)


def chip(value, color=FAINT, size=22, pad=0.16):
    mob = tx(value, size, color, BOLD)
    frame = SurroundingRectangle(mob, color=color, buff=pad, stroke_width=1.4,
                                 corner_radius=0.06)
    frame.set_fill(PANEL, opacity=0.9)
    return VGroup(frame, mob)


def arrow_between(start, end, color=MUTED, width=2.2, buff=0.12, dashed=False):
    arrow = Arrow(start, end, buff=buff, color=color, stroke_width=width,
                  max_tip_length_to_length_ratio=0.16)
    if dashed:
        arrow = DashedVMobject(arrow, num_dashes=18)
    return arrow


def token_grid(rows, cols, color, cell=0.16, masked=(), gap=1.22):
    masked = set(masked)
    group = VGroup()
    for r in range(rows):
        for c in range(cols):
            hidden = (r, c) in masked
            square = RoundedRectangle(
                width=cell, height=cell, corner_radius=cell * 0.14,
                stroke_color=CORAL if hidden else color, stroke_width=1.1,
                fill_color="#3A1524" if hidden else color,
                fill_opacity=0.95 if hidden else 0.82)
            square.move_to(((c - (cols - 1) / 2) * cell * gap,
                            ((rows - 1) / 2 - r) * cell * gap, 0))
            group.add(square)
    return group


def occupancy_image(pattern, width=2.4, masked=None, color="blue"):
    """A real 64x64 occupancy map as pixels; optional 16x16 mask overlay."""
    pixels = np.asarray(pattern > 0.5)
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[:] = (9, 16, 30)
    on = (92, 168, 255) if color == "blue" else (255, 170, 112)
    rgb[pixels] = on
    image = ImageMobject(rgb)
    image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    image.set_width(width)
    frame = SurroundingRectangle(image, color=BLUE if color == "blue" else ORANGE,
                                 buff=0.05, stroke_width=1.5)
    group = Group(image, frame)
    if masked is not None:
        overlay = VGroup()
        hidden = np.asarray(masked) < 0.5            # mask: 1 visible, 0 masked
        cell = width / 16
        for r in range(16):
            for c in range(16):
                if hidden[r, c]:
                    square = Square(side_length=cell, stroke_width=0,
                                    fill_color=CORAL, fill_opacity=0.30)
                    square.move_to(image.get_corner(UL) +
                                   RIGHT * cell * (c + 0.5) +
                                   DOWN * cell * (r + 0.5))
                    overlay.add(square)
        group.add(overlay)
    return group


def spectrum_curve(real, imag, width=4.6, height=1.8, axes=True,
                   colors=(CYAN, PURPLE), stroke=2.4, labels=None):
    values = np.concatenate([real, imag])
    top = float(np.abs(values).max())
    top = top if top > 1e-6 else 1.0
    y_lim = np.ceil(top * 10) / 10
    ax = Axes(x_range=[0, 300, 100], y_range=[-y_lim, y_lim, y_lim],
              x_length=width, y_length=height,
              axis_config={"color": FAINT, "stroke_width": 1.2,
                           "include_ticks": False, "include_numbers": False},
              tips=False)
    x = np.arange(len(real))
    real_line = VMobject(color=colors[0], stroke_width=stroke)
    real_line.set_points_smoothly([ax.c2p(i, float(v)) for i, v in zip(x, real)])
    imag_line = VMobject(color=colors[1], stroke_width=stroke * 0.9)
    imag_line.set_points_smoothly([ax.c2p(i, float(v)) for i, v in zip(x, imag)])
    group = VGroup()
    if axes:
        group.add(ax)
    group.add(real_line, imag_line)
    group.lines = VGroup(real_line, imag_line)
    group.axes = ax
    if labels:
        legend = VGroup(
            VGroup(Dot(radius=0.055, color=colors[0]),
                   tx(labels[0], 18, colors[0])).arrange(RIGHT, buff=0.1),
            VGroup(Dot(radius=0.055, color=colors[1]),
                   tx(labels[1], 18, colors[1])).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.3)
        legend.next_to(ax, UP, buff=0.16)
        group.add(legend)
        group.legend = legend
    return group


def dashed_bracket(start, end, color=MUTED, width=1.4):
    return DashedLine(start, end, color=color, stroke_width=width,
                      dash_length=0.08)


def source_note(text_value):
    return tx(text_value, 17, MUTED).to_corner(DR, buff=0.34)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def load_timeline():
    if TIMELINE_JSON.exists():
        with open(TIMELINE_JSON, encoding="utf-8") as f:
            data = json.load(f)
        return {s["id"]: s for s in data["scenes"]}
    return {}


def nominal_timeline():
    """Fallback so the file always renders: 2.2 words/second + a tail."""
    if not (PRESENTATION / "narration.md").exists():
        return {}
    scenes, current = {}, None
    for raw in (PRESENTATION / "narration.md").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("## SCENE"):
            current = SCENE_ORDER[len(scenes)]
            scenes[current] = {"id": current, "sentences": [], "duration": 0.0}
            continue
        if current and line[:2].strip().isdigit() and ". " in line:
            text_value = line.split(". ", 1)[1]
            scenes[current]["sentences"].append({"text": text_value, "start": 0.0,
                                                 "end": 0.0})
    for scene in scenes.values():
        cursor = 0.0
        for sentence in scene["sentences"]:
            words = len(sentence["text"].split())
            sentence["start"] = round(cursor, 3)
            cursor += words / 2.15 + 0.3
            sentence["end"] = round(cursor, 3)
        scene["duration"] = round(cursor + 1.2, 3)
    return scenes


# ---------------------------------------------------------------------------
# Base scene: clock-driven synchronisation with the voice track
# ---------------------------------------------------------------------------

class JEPAScene(MovingCameraScene):
    scene_id = "hook"
    EXTRA_TAIL = {"closing": 4.0}

    def setup(self):
        super().setup()
        self.camera.background_color = BG
        self._clock = 0.0

    # -- clock ------------------------------------------------------------
    def play(self, *args, **kwargs):
        super().play(*args, **kwargs)
        self._clock = float(getattr(self.renderer, "time", self._clock + 1.0))

    def wait(self, duration=1.0, **kwargs):
        super().wait(duration, **kwargs)
        self._clock = float(getattr(self.renderer, "time", self._clock + duration))

    def elapsed(self):
        return float(getattr(self.renderer, "time", self._clock)) - self._start

    def sync_to(self, when):
        gap = when - self.elapsed()
        if gap > 0.03:
            self.wait(gap)

    def cue(self, index):
        """Hold until sentence `index` of this scene begins."""
        if index < len(self.sentences):
            self.sync_to(self.sentences[index]["start"])

    # -- lifecycle --------------------------------------------------------
    def run_scene(self, scene_id):
        timeline = load_timeline() or nominal_timeline()
        info = timeline.get(scene_id, {})
        self.sentences = info.get("sentences", [])
        self.scene_duration = float(info.get("duration", 30.0))
        self._start = float(getattr(self.renderer, "time", 0.0))
        self._clock = self._start
        getattr(self, f"scene_{scene_id}")()
        self.sync_to(self.scene_duration + self.EXTRA_TAIL.get(scene_id, 0.0))

    def construct(self):
        self.run_scene(self.scene_id)

    # -- camera -----------------------------------------------------------
    def zoom_to(self, mob, scale=0.55, run_time=1.5):
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(scale).move_to(mob.get_center()),
                  run_time=run_time, rate_func=smooth)

    def restore_camera(self, run_time=1.3):
        self.play(Restore(self.camera.frame), run_time=run_time, rate_func=smooth)

    def reveal(self, *mobjects, direction=RIGHT, shift=0.14, run_time=0.9,
               lag=0.12):
        """Staggered slide-and-fade entrance (softer than a single FadeIn)."""
        animations = [FadeIn(m, shift=direction * shift) for m in mobjects]
        self.play(LaggedStart(*animations, lag_ratio=lag), run_time=run_time)

    def drift(self, direction, amount=0.12, run_time=3.0):
        """A very slow camera drift during a hold, so the frame is never dead."""
        self.play(self.camera.frame.animate.shift(direction * amount),
                  run_time=run_time, rate_func=linear)

    # -- motion flourishes -------------------------------------------------
    def signal(self, path, color, run_time=0.7, radius=0.075):
        dot = Dot(path.get_start(), radius=radius, color=color).set_z_index(9)
        halo = Circle(radius=radius * 2.4, stroke_color=color, stroke_width=2,
                      stroke_opacity=0.55).move_to(dot)
        halo.add_updater(lambda m: m.move_to(dot.get_center()))
        self.add(halo, dot)
        self.play(MoveAlongPath(dot, path, rate_func=linear), run_time=run_time)
        halo.clear_updaters()
        self.play(FadeOut(VGroup(halo, dot), scale=0.4), run_time=0.25)


# ---------------------------------------------------------------------------
# SCENE 1 — the hook (26.8 s)
# ---------------------------------------------------------------------------

class S01Hook(JEPAScene):
    scene_id = "hook"

    def scene_hook(self):
        data = load_data()

        # ---- 1. the physical object: a patterned sheet ----------------------
        slab, pillars = self.metasurface()
        slab_tag = VGroup(
            tx("meta-atoms", 16, BLUE, BOLD),
            tx("substrate", 15, MUTED),
        )
        pill_tag = tx("each pillar is smaller than the wavelength", 18, MUTED)
        pill_tag.to_edge(DOWN, buff=1.5)
        self.cue(0)
        self.play(FadeIn(slab), FadeIn(pillars, lag_ratio=0.03), run_time=1.6)
        self.play(FadeIn(pill_tag), run_time=0.7)

        ruler = self.wavelength_ruler()
        self.play(GrowFromCenter(ruler[0]), FadeIn(ruler[1]), FadeIn(ruler[2]),
                  run_time=1.0)

        # ---- 2. what it does: a flat lens -----------------------------------
        self.cue(1)
        rays, bent, focus = self.lens_rays()
        self.play(LaggedStart(*[Create(r) for r in rays], lag_ratio=0.15),
                  run_time=1.0)
        self.play(LaggedStart(*[Transform(r, b) for r, b in zip(rays, bent)],
                              lag_ratio=0.15), run_time=1.6)
        self.play(FadeIn(focus), run_time=0.6)
        lens_tag = tx("a flat lens: one thin sheet focuses the wave", 18, CYAN)
        lens_tag.to_edge(DOWN, buff=1.5)
        self.play(ReplacementTransform(pill_tag, lens_tag), run_time=0.7)

        # ---- 3. where they are used -----------------------------------------
        self.cue(2)
        uses = VGroup(*[
            panel(name, detail, colour, 2.6, 0.9, 18, 14)
            for name, detail, colour in (
                ("flat lenses", "focus without curvature", CYAN),
                ("holograms", "shape the wavefront", PURPLE),
                ("beam steering", "point without moving parts", GREEN),
            )]).arrange(RIGHT, buff=0.4).move_to(DOWN * 2.1)
        self.play(FadeOut(Group(ruler, lens_tag, rays, bent, focus)), run_time=0.6)
        self.play(FadeIn(uses, lag_ratio=0.2), run_time=1.0)

        # ---- 4. designing one = choosing the cells ---------------------------
        self.cue(3)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(0.62).move_to(slab.get_center()),
                  run_time=1.4, rate_func=smooth)
        top_view = Square(side_length=2.3, stroke_color=BLUE, stroke_width=1.4,
                          fill_color=PANEL, fill_opacity=0.98)
        top_view.move_to(slab.get_center() + UP * 0.05)
        cells = VGroup()
        span = 2.2 / 32
        for r in range(32):
            for c in range(32):
                filled = ((r * 7919 + c * 104729) % 997) / 997 < 0.42
                square = Square(side_length=span * 0.94, stroke_width=0,
                                fill_color=BLUE if filled else PANEL,
                                fill_opacity=0.95 if filled else 0.95)
                square.move_to(((c - 15.5) * span, (15.5 - r) * span, 0))
                cells.add(square)
        cells.move_to(top_view)
        view_tag = tx("top view", 17, MUTED).next_to(top_view, UP, buff=0.16)
        self.play(FadeOut(Group(pillars, uses, slab, ruler)), run_time=0.6)
        self.play(FadeIn(top_view), FadeIn(cells), FadeIn(view_tag), run_time=1.0)
        design_tag = math("64 × 64 cells · each empty or filled", 24, INK)
        design_tag.next_to(top_view, DOWN, buff=0.32)
        self.play(FadeIn(design_tag), run_time=0.7)

        # ---- 5. the response we want -----------------------------------------
        self.cue(4)
        self.play(Restore(self.camera.frame), run_time=1.4, rate_func=smooth)
        self.play(FadeOut(Group(slab, top_view, view_tag, design_tag, cells)),
                  run_time=0.6)
        design = occupancy_image(data["pattern"], width=2.3)
        design.move_to(RIGHT * 3.6 + UP * 0.9)
        spec = spectrum_curve(*data["spectrum"], width=4.8, height=2.1)
        spec.move_to(LEFT * 3.6 + UP * 0.9)
        spec_tag = tx("the response we want", 19, CYAN, BOLD)
        spec_tag.next_to(spec, UP, buff=0.2)
        design_tag2 = tx("a structure to build", 18, BLUE, BOLD)
        design_tag2.next_to(design, UP, buff=0.2)
        axes_tag = VGroup(
            tx("frequency", 15, MUTED),
            tx("amplitude", 15, MUTED),
        )
        axes_tag[0].next_to(spec[0], DOWN, buff=0.1)
        axes_tag[1].next_to(spec[0], LEFT, buff=0.1).rotate(PI / 2)
        design_cells = token_grid(16, 16, BLUE, 0.105, gap=1.0).move_to(design)
        self.reveal(design_cells, spec, spec_tag, direction=UP, run_time=1.2)
        self.play(FadeIn(axes_tag[0]), FadeIn(axes_tag[1]), run_time=0.6)
        self.play(FadeOut(design_cells), FadeIn(design), FadeIn(design_tag2),
                  run_time=1.0)

        # ---- 6. why the direction matters ------------------------------------
        self.cue(5)
        forward = arrow_between(design.get_left(), spec.get_right(), GREEN, 2.6)
        forward_tag = tx("forward: computable", 16, GREEN, BOLD)
        forward_tag.next_to(forward, UP, buff=0.12)
        self.play(Create(forward), FadeIn(forward_tag), run_time=0.9)
        self.signal(forward, GREEN, run_time=0.9)
        self.play(FadeOut(Group(forward, forward_tag)), run_time=0.5)
        reverse = arrow_between(spec.get_right(), design.get_left(), CORAL, 2.4,
                                dashed=True)
        reverse_tag = tx("inverse: not unique", 16, CORAL, BOLD)
        reverse_tag.next_to(reverse, DOWN, buff=0.12)
        self.play(Create(reverse), FadeIn(reverse_tag), run_time=0.9)
        self.play(Circumscribe(design, color=CORAL, time_width=0.6), run_time=1.0)

        # ---- 7. the question, and the title ----------------------------------
        self.cue(6)
        title = tx("Learning to Design Metamaterials with JEPA", 40, INK, BOLD)
        title.move_to(DOWN * 1.5)
        question = tx("Can a machine learn to design a structure from the "
                      "behaviour we ask for?", 27, YELLOW, BOLD, wrap=64)
        question.to_edge(DOWN, buff=0.7)
        self.play(FadeIn(title, shift=UP * 0.1), run_time=0.9)
        self.play(Flash(title.get_center(), color=CYAN, flash_radius=2.8),
                  run_time=0.6)
        self.play(FadeIn(question, shift=UP * 0.1), run_time=0.9)

    # -- opening pieces -------------------------------------------------------

    def metasurface(self):
        """A metasurface seen from the side: substrate plus meta-atom pillars."""
        substrate = Rectangle(width=5.4, height=0.16, stroke_color=BLUE,
                              stroke_width=1.2, fill_color=PANEL_HI,
                              fill_opacity=1.0)
        substrate.move_to(UP * 0.55)
        rng = np.random.RandomState(5)
        pillars = VGroup()
        for i in range(26):
            height = 0.3 + 0.45 * rng.rand()
            pillar = Rectangle(width=0.105, height=height, stroke_width=0,
                               fill_color=BLUE, fill_opacity=0.9)
            pillar.move_to((-2.47 + i * 0.1975, 0.63 + height / 2 + 0.08, 0))
            pillars.add(pillar)
        return substrate, pillars

    def wavelength_ruler(self):
        """A λ bracket under the sheet, obviously longer than the pillar spacing."""
        start = np.array([-0.9, 0.12, 0])
        end = np.array([0.9, 0.12, 0])
        bracket = DoubleArrow(start, end, buff=0, color=CYAN, stroke_width=1.6,
                              max_tip_length_to_length_ratio=0.08)
        label = math("λ", 26, CYAN).move_to((0, -0.12, 0))
        note = tx("wavelength", 16, MUTED).move_to((0, -0.5, 0))
        return VGroup(bracket, label, note)

    def lens_rays(self):
        """Three parallel rays that bend at the sheet and meet at one focus."""
        ys = (1.05, 0.72, 0.39)
        start_x, left_face, right_face = -6.4, -2.75, 2.75
        focus = np.array([3.55, 0.72, 0])
        rays = VGroup(*[Line([start_x, y, 0], [left_face, y, 0], color=MUTED,
                             stroke_width=2.0) for y in ys])
        bent = VGroup(*[Line([right_face, y, 0], focus, color=CYAN, stroke_width=2.0)
                        for y in ys])
        dot = VGroup(
            Dot(focus, radius=0.09, color=CYAN),
            Circle(radius=0.24, stroke_color=CYAN, stroke_width=1.2,
                   stroke_opacity=0.6).move_to(focus),
        )
        return rays, bent, dot

# ---------------------------------------------------------------------------
# SCENE 2 — the inverse problem (44.4 s)
# ---------------------------------------------------------------------------

class S02Inverse(JEPAScene):
    scene_id = "inverse"

    def scene_inverse(self):
        data = load_data()
        top = heading(1, "the inverse problem")

        geometry = occupancy_image(data["pattern"], width=2.2)
        geometry.move_to(LEFT * 5.0 + UP * 0.9)
        spec = spectrum_curve(*data["spectrum"], width=4.4, height=1.8,
                              labels=("real", "imaginary"))
        spec.move_to(RIGHT * 3.5 + UP * 0.9)
        geo_tag = tx("geometry  G", 20, BLUE, BOLD).next_to(geometry, UP, buff=0.18)
        spec_tag = tx("response  S", 20, CYAN, BOLD).next_to(spec, UP, buff=0.22)

        self.cue(0)
        self.play(FadeIn(top), FadeIn(geometry), FadeIn(geo_tag),
                  FadeIn(spec), FadeIn(spec_tag), run_time=1.3)
        forward = arrow_between(geometry.get_right(), spec.get_left(), GREEN, 2.6)
        forward_label = tx("forward model", 18, GREEN, BOLD).next_to(forward, UP, buff=0.12)
        self.play(Create(forward), FadeIn(forward_label), run_time=1.0)
        self.signal(forward, GREEN, run_time=0.9)

        eq_forward = math("(G, l, h, r)  →  S", 34, INK)
        eq_forward.move_to(LEFT * 0.6 + DOWN * 0.9)
        self.cue(1)
        self.play(FadeIn(eq_forward, shift=UP * 0.1), run_time=0.8)
        params = VGroup(*[
            VGroup(chip(f"{n} = {v:.2f}", YELLOW, 18), tx(name, 13, MUTED))
            .arrange(DOWN, buff=0.08)
            for n, v, name in zip(("l", "h", "r"), data["scalars"],
                                  ("lattice period", "meta-atom height",
                                   "refractive index"))])
        params.arrange(RIGHT, buff=0.3).next_to(eq_forward, DOWN, buff=0.4)
        self.play(FadeIn(params, lag_ratio=0.15), run_time=0.8)

        self.cue(2)
        inverse = arrow_between(spec.get_left(), geometry.get_right(), CORAL,
                                2.6, dashed=True)
        inverse_label = tx("inverse design", 18, CORAL, BOLD).next_to(inverse, DOWN, buff=0.12)
        self.play(FadeOut(forward), FadeOut(forward_label),
                  FadeIn(inverse), FadeIn(inverse_label), run_time=1.2)
        eq_inverse = math("S_{target}  →  ( Ĝ, l̂, ĥ, r̂ )", 34, CORAL)
        eq_inverse.move_to(LEFT * 0.6 + DOWN * 1.75)
        self.play(FadeOut(eq_forward), FadeIn(eq_inverse, shift=UP * 0.1),
                  run_time=1.0)
        self.play(FadeOut(params), run_time=0.5)

        # Many-to-one: two real designs whose spectra almost coincide.
        self.cue(3)
        pair = Group(
            occupancy_image(data["pattern"], 1.9),
            occupancy_image(data["neighbour_pattern"], 1.9),
        ).arrange(RIGHT, buff=0.6)
        pair.move_to(LEFT * 4.8 + UP * 0.35)
        pair_tag = tx("two different designs", 19, BLUE, BOLD)
        pair_tag.next_to(pair, DOWN, buff=0.28)

        top = float(np.abs(data["spectrum"]).max())
        axes = Axes(x_range=[0, 300, 100], y_range=[-top, top, top],
                    x_length=4.4, y_length=1.9,
                    axis_config={"color": FAINT, "stroke_width": 1.1,
                                 "include_ticks": False, "include_numbers": False},
                    tips=False)
        axes.move_to(RIGHT * 3.6 + UP * 0.35)
        own = VMobject(color=CYAN, stroke_width=2.4)
        own.set_points_smoothly([axes.c2p(i, float(v))
                                 for i, v in enumerate(data["spectrum"][0])])
        other = VMobject(color=YELLOW, stroke_width=2.0)
        other.set_points_smoothly([axes.c2p(i, float(v))
                                   for i, v in enumerate(data["neighbour_spectrum"][0])])
        other = DashedVMobject(other, num_dashes=70)
        legend = VGroup(
            VGroup(Dot(radius=0.05, color=CYAN), tx("design 1", 17, CYAN)).arrange(RIGHT, buff=0.1),
            VGroup(Dot(radius=0.05, color=YELLOW), tx("design 2", 17, YELLOW)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.35).next_to(axes, UP, buff=0.2)
        distance = tx(f"their spectra differ by "
                      f"{100 * data['spectrum_distance']:.1f} %", 21, YELLOW, BOLD)
        distance.next_to(axes, DOWN, buff=0.35)
        self.play(FadeOut(Group(geometry, geo_tag, eq_inverse, spec, spec_tag,
                                inverse, inverse_label)), run_time=0.7)
        self.play(FadeIn(pair), FadeIn(pair_tag), run_time=1.0)
        self.play(Create(axes), Create(own), run_time=1.2)
        self.play(Create(other), FadeIn(legend), FadeIn(distance), run_time=1.4)

        self.cue(4)
        consequence = tx("so the model must use the target response, the visible "
                         "structure, and the parameters it knows", 24, MUTED, wrap=66)
        consequence.to_edge(DOWN, buff=0.55)
        self.play(FadeIn(consequence, shift=UP * 0.1), run_time=0.9)


# ---------------------------------------------------------------------------
# SCENE 3 — the dataset (44.5 s)
# ---------------------------------------------------------------------------

class S03Dataset(JEPAScene):
    scene_id = "dataset"

    def scene_dataset(self):
        data = load_data()
        top = heading(2, "the dataset")

        # The release's packing: three image channels.
        packed = VGroup()
        for name, detail, color in (("ch0", "r / 5 · occupied", YELLOW),
                                    ("ch1", "h · occupied", YELLOW),
                                    ("ch2", "l / 3 · dense", YELLOW)):
            packed.add(panel(name, detail, color, 1.6, 0.9, 20, 15))
        packed.arrange(DOWN, buff=0.18).move_to(LEFT * 5.6 + UP * 0.45)
        packed_tag = tx("as released:", 19, MUTED).next_to(packed, UP, buff=0.2)

        geometry = occupancy_image(data["pattern"], width=2.1)
        geometry.move_to(LEFT * 3.0 + UP * 0.85)
        geometry_tag = tx("occupancy  M", 19, BLUE, BOLD).next_to(geometry, UP, buff=0.2)
        scalars = VGroup(*[
            panel(n, f"{v:.2f} · known", YELLOW, 1.9, 0.72, 21, 15)
            for n, v in zip(("l", "h", "r"), data["scalars"])])
        scalars.arrange(DOWN, buff=0.18).move_to(LEFT * 0.6 + UP * 0.5)
        scalar_tag = tx("three global scalars", 19, YELLOW, BOLD)
        scalar_tag.next_to(scalars, UP, buff=0.2)
        spec = spectrum_curve(*data["spectrum"], width=3.2, height=1.6,
                              labels=("real", "imaginary"))
        spec.move_to(RIGHT * 4.35 + UP * 0.65)
        spec_tag = tx("complex spectrum  S", 19, CYAN, BOLD)
        spec_tag.next_to(spec, DOWN, buff=0.22)

        self.cue(0)
        self.play(FadeIn(top), FadeIn(packed), FadeIn(packed_tag), run_time=1.0)

        self.cue(1)
        self.reveal(geometry, geometry_tag, direction=UP, run_time=0.9)
        self.reveal(scalar_tag, scalars, direction=RIGHT, run_time=0.9, lag=0.18)
        self.play(Create(spec), FadeIn(spec_tag), run_time=1.3)

        counts = VGroup(*[
            panel(label, value, color, 1.7, 0.62, 18, 16)
            for label, value, color in (("train", "139,906", GREEN),
                                        ("val", "17,488", YELLOW),
                                        ("test", "17,489", CYAN))])
        counts.arrange(RIGHT, buff=0.28)
        counts.to_edge(DOWN, buff=0.42)
        self.play(FadeIn(counts, lag_ratio=0.15), run_time=0.7)

        # Factorising the packing, with the real numbers.
        self.cue(2)
        eq_open = math("M ∈ {0,1}^{64×64}      p ∈ ℝ³      S ∈ ℝ^{2×301}", 30, INK)
        eq_open.move_to(DOWN * 1.75)
        self.play(FadeIn(eq_open), run_time=0.7)

        self.cue(3)
        split_arrows = VGroup(
            arrow_between(packed.get_right(), geometry.get_left(), BLUE),
            CurvedArrow(packed.get_corner(DR) + DOWN * 0.05,
                        scalars.get_corner(DL) + LEFT * 0.05,
                        color=YELLOW, angle=-TAU / 7, stroke_width=2.2),
        )
        self.play(LaggedStart(*[Create(a) for a in split_arrows], lag_ratio=0.2),
                  run_time=1.2)
        self.play(Indicate(geometry, color=BLUE), run_time=0.8)
        flags = tx("each scalar carries its own known / unknown flag", 19, YELLOW)
        flags.next_to(scalars, DOWN, buff=0.3)
        self.play(FadeIn(flags), run_time=0.7)

        self.cue(4)
        claim = tx("spatial uncertainty and parameter uncertainty are separable",
                   25, MUTED, wrap=62).move_to(DOWN * 2.55)
        self.play(FadeIn(claim, shift=UP * 0.1), run_time=1.0)


# ---------------------------------------------------------------------------
# SCENE 4 — the core idea (53.0 s)
# ---------------------------------------------------------------------------

class S04CoreIdea(JEPAScene):
    scene_id = "jepa"

    def scene_jepa(self):
        data = load_data()
        top = heading(3, "predicting representations")

        source = occupancy_image(data["pattern"], width=1.8)
        source.move_to(LEFT * 5.5 + UP * 1.15)
        grid = token_grid(16, 16, BLUE, 0.098, gap=1.0)
        grid.move_to(source)
        self.cue(0)
        self.play(FadeIn(top), FadeIn(source), run_time=1.0)
        self.play(FadeIn(grid, lag_ratio=0.002), run_time=1.0)
        self.play(FadeOut(source), run_time=0.4)

        self.cue(1)
        token_tag = tx("16 × 16 patches → 256 tokens", 18, BLUE, BOLD)
        token_tag.next_to(grid, DOWN, buff=0.3)
        token_tag.align_to(grid, LEFT)
        self.play(FadeIn(token_tag), run_time=0.6)

        self.cue(2)
        hidden = {(r, c) for r in range(16) for c in range(16)
                  if (r * 7 + c * 5) % 11 < 5}
        masked_grid = token_grid(16, 16, BLUE, 0.098, masked=hidden, gap=1.0)
        masked_grid.move_to(grid)
        self.play(Transform(grid, masked_grid), run_time=1.4)
        student = panel("Student", "visible tokens + S + known p", GREEN, 2.7, 0.95, 21, 14)
        student.move_to(LEFT * 2.3 + UP * 1.15)
        self.play(FadeIn(student, shift=RIGHT * 0.15),
                  Create(arrow_between(grid.get_right(), student.get_left(), BLUE,
                                       buff=0.2)), run_time=0.9)

        self.cue(3)
        teacher = panel("Teacher", "complete geometry + true p", VIOLET, 2.9, 0.95, 21, 14)
        teacher.move_to(RIGHT * 1.3 + UP * 2.55)
        full = occupancy_image(data["pattern"], 0.85)
        full.next_to(teacher, RIGHT, buff=0.3)
        self.play(FadeIn(teacher, shift=LEFT * 0.15), FadeIn(full), run_time=0.9)

        self.cue(4)
        z_s = token_grid(3, 5, GREEN, 0.14).move_to(LEFT * 2.3 + DOWN * 0.75)
        z_t = token_grid(3, 5, VIOLET, 0.14).move_to(RIGHT * 1.3 + DOWN * 0.75)
        z_s_tag = tx("Ẑ masked", 18, GREEN, BOLD).next_to(z_s, DOWN, buff=0.2)
        z_t_tag = tx("Z target", 18, VIOLET, BOLD).next_to(z_t, DOWN, buff=0.2)
        self.play(FadeIn(z_s), FadeIn(z_s_tag), FadeIn(z_t), FadeIn(z_t_tag),
                  run_time=0.9)
        compare = DashedLine(z_s.get_right() + RIGHT * 0.15,
                             z_t.get_left() + LEFT * 0.15,
                             color=INK, stroke_width=1.6, dash_length=0.1)
        self.play(Create(compare), run_time=0.6)

        self.cue(5)
        loss = math("L_{JEPA} = ‖ Ẑ_{masked} − Z_{target} ‖²", 38, INK)
        loss.move_to(DOWN * 2.35)
        self.play(FadeIn(loss, shift=UP * 0.1), run_time=0.9)
        detail = tx("+ variance and covariance terms that prevent collapse",
                    20, PURPLE, wrap=52).next_to(loss, DOWN, buff=0.2)
        self.play(FadeIn(detail), run_time=0.6)

        self.cue(6)
        decoder = panel("Decoder", "latents → occupancy", ORANGE, 2.4, 0.95, 21, 14)
        decoder.move_to(RIGHT * 5.3 + DOWN * 0.75)
        self.play(FadeIn(decoder, shift=LEFT * 0.15),
                  Create(arrow_between(z_t.get_right(), decoder.get_left(), ORANGE,
                                       buff=0.2)), run_time=0.9)


# ---------------------------------------------------------------------------
# SCENE 5 — the architecture (106 s)
# ---------------------------------------------------------------------------

class S05Architecture(JEPAScene):
    scene_id = "architecture"

    def scene_architecture(self):
        data = load_data()
        top = heading(4, "the unified 192-d architecture")
        self.cue(0)
        self.play(FadeIn(top), run_time=0.8)

        # ---------------- ACT 1: occupancy encoder + scalar encoder ----------
        y_occ = 2.35
        img = occupancy_image(data["pattern"], width=1.2)
        img.move_to(LEFT * 6.45 + UP * y_occ)
        conv = panel("Conv2d", "1→192 · k4 · s4", BLUE, 1.7, 0.88, 19, 14)
        conv.move_to(LEFT * 4.85 + UP * y_occ)
        tokens = token_grid(16, 16, BLUE, 0.055, gap=1.25)
        tokens.move_to(LEFT * 3.15 + UP * y_occ)
        tokens_tag = tx("256 × 192", 16, BLUE, BOLD).next_to(tokens, DOWN, buff=0.14)
        blocks = VGroup(*[RoundedRectangle(width=1.35, height=0.2, corner_radius=0.04,
                                           stroke_color=BLUE, stroke_width=1.2,
                                           fill_color=PANEL_HI, fill_opacity=0.95)
                          for _ in range(6)]).arrange(DOWN, buff=0.06)
        blocks.move_to(LEFT * 1.35 + UP * y_occ)
        z_x = panel("z_x", "256 × 192", PURPLE, 1.35, 0.8, 18, 14)
        z_x.move_to(RIGHT * 0.6 + UP * y_occ)
        lane_arrows = VGroup(
            arrow_between(img.get_right(), conv.get_left(), BLUE),
            arrow_between(conv.get_right(), tokens.get_left(), BLUE),
            arrow_between(tokens.get_right(), blocks.get_left(), BLUE),
            arrow_between(blocks.get_right(), z_x.get_left(), BLUE),
        )

        self.cue(1)
        self.play(FadeIn(img), FadeIn(conv), FadeIn(tokens), FadeIn(tokens_tag),
                  run_time=0.9)
        self.play(LaggedStart(*[Create(a) for a in lane_arrows[:3]], lag_ratio=0.2),
                  run_time=1.0)

        self.cue(2)
        self.play(FadeIn(blocks), FadeIn(z_x),
                  Create(lane_arrows[3]), run_time=1.0)
        self.play(Indicate(blocks, color=BLUE, scale_factor=1.02), run_time=1.4)

        # scalar lane
        self.cue(3)
        scal = VGroup(*[
            panel(n, f"{v:.2f}", YELLOW, 0.85, 0.56, 17, 14)
            for n, v in zip(("l", "h", "r"), data["scalars"])])
        scal.arrange(DOWN, buff=0.09).move_to(LEFT * 6.45 + UP * -0.25)
        six = panel("6 → 128 → 128", "value + flag", YELLOW, 1.85, 0.82, 18, 13)
        six.move_to(LEFT * 4.7 + UP * -0.25)
        six_tag = math("[ l, k_l, h, k_h, r, k_r ]", 20, YELLOW)
        six_tag.next_to(six, DOWN, buff=0.18)
        self.play(FadeIn(scal), FadeIn(six), run_time=0.9)
        scal_arrow = arrow_between(scal.get_right(), six.get_left(), YELLOW)
        self.play(Create(scal_arrow), run_time=0.6)
        self.play(FadeIn(six_tag), run_time=0.5)

        self.cue(4)
        film_pairs = panel("6 FiLM pairs", "one per block", YELLOW, 1.7, 0.68, 17, 13)
        film_pairs.move_to(LEFT * 1.55 + UP * 0.6)
        summary = panel("summary token", "1 × 192", YELLOW, 1.7, 0.68, 17, 13)
        summary.move_to(LEFT * 1.55 + UP * -0.55)
        self.play(FadeIn(film_pairs), FadeIn(summary), run_time=0.8)
        pair_arrows = VGroup(
            arrow_between(six.get_right(), film_pairs.get_left(), YELLOW),
            arrow_between(six.get_right(), summary.get_left(), YELLOW))
        self.play(Create(pair_arrows), run_time=0.8)
        feed = VGroup(*[dashed_bracket(film_pairs.get_top() + LEFT * 0.25 * i,
                                       blocks[i].get_left(), YELLOW, 1.0)
                        for i in range(0, 6, 2)])
        self.play(LaggedStart(*[Create(f) for f in feed], lag_ratio=0.2), run_time=1.0)

        self.cue(5)
        self.play(Indicate(film_pairs, color=YELLOW),
                  Indicate(summary, color=YELLOW), run_time=1.0)

        self.cue(6)
        film_eq = math("FiLM(x) = γ ⊙ x + β", 26, YELLOW)
        film_eq.move_to(LEFT * 4.7 + DOWN * 1.55)
        self.play(FadeIn(film_eq), run_time=0.7)

        # ---------------- ACT 2: spectrum path and fusion --------------------
        act1 = Group(img, conv, tokens, tokens_tag, blocks, z_x, lane_arrows,
                     scal, six, six_tag, film_pairs, summary, pair_arrows,
                     scal_arrow, feed, film_eq)
        self.cue(7)
        self.play(FadeOut(act1), run_time=0.9)

        spec = spectrum_curve(*data["spectrum"], width=1.7, height=0.9)
        spec.move_to(LEFT * 5.95 + UP * 0.9)
        frozen = panel("released encoder", "frozen · 301 × 256", FROZEN, 2.0, 0.8,
                       16, 12, dashed=True)
        frozen.move_to(LEFT * 4.05 + UP * 0.9)
        local = panel("A_local", "301 × 256", CYAN, 1.4, 0.7, 17, 13)
        local.move_to(LEFT * 2.15 + UP * 0.9)
        lane2 = VGroup(
            arrow_between(spec.get_right(), frozen.get_left(), CYAN),
            arrow_between(frozen.get_right(), local.get_left(), CYAN),
        )
        self.play(FadeIn(spec), FadeIn(frozen), FadeIn(local), run_time=1.0)
        self.play(Create(lane2), run_time=0.8)

        self.cue(8)
        c_phys = panel("c_physics", "384 · every block", CYAN, 1.75, 0.62, 17, 13)
        c_phys.move_to(LEFT * 0.55 + UP * 1.65)
        goals = panel("a_goal", "16 × 384", CYAN, 1.45, 0.62, 18, 14)
        goals.move_to(LEFT * 0.55 + UP * 0.2)
        readout = VGroup(
            arrow_between(local.get_right(), c_phys.get_left(), CYAN),
            arrow_between(local.get_right(), goals.get_left(), CYAN),
        )
        self.play(FadeIn(c_phys), FadeIn(goals), run_time=0.8)
        self.play(Create(readout), run_time=0.8)

        self.cue(9)
        context = panel("context", "256 occupancy + 1 scalar", BLUE, 2.5, 0.75, 17, 13)
        context.move_to(RIGHT * 1.35 + UP * 2.65)
        fusion = panel("Fusion", "273 × 192", PURPLE, 1.75, 1.15, 20, 15)
        fusion.move_to(RIGHT * 2.45 + UP * 0.9)
        fusion_in = VGroup(
            tx("256 occupancy", 14, BLUE, BOLD),
            tx("+ 16 goals → 192", 14, CYAN, BOLD),
            tx("+ 1 scalar", 14, YELLOW, BOLD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
        fusion_in.next_to(fusion, DOWN, buff=0.3)
        self.play(FadeIn(context), FadeIn(fusion), FadeIn(fusion_in), run_time=1.0)
        into_fusion = VGroup(
            arrow_between(context.get_bottom(), fusion.get_top() + LEFT * 0.35, BLUE),
            arrow_between(goals.get_right(), fusion.get_left() + DOWN * 0.25, CYAN),
        )
        self.play(LaggedStart(*[Create(a) for a in into_fusion], lag_ratio=0.25),
                  run_time=1.1)

        self.cue(10)
        two_layer = tx("2 layers", 15, PURPLE, BOLD).next_to(fusion, DOWN, buff=1.35)
        self.play(FadeIn(two_layer), Indicate(fusion, color=PURPLE), run_time=1.0)

        # ---------------- ACT 3: the GCLCT predictor -------------------------
        act2 = Group(spec, frozen, local, lane2, c_phys, goals, readout,
                     context, fusion, fusion_in, into_fusion, two_layer)
        self.cue(11)
        self.play(FadeOut(act2), run_time=0.9)

        # One frame: the three contexts (with the spectral goals among them), the
        # query set, and the block that makes them interact.
        context = VGroup(
            self.token_band("256 occupancy tokens", "from z_x", BLUE, 2.7, 0.62),
            self.token_band("16 goal tokens", "from the spectrum", CYAN, 2.7, 0.58),
            self.token_band("1 scalar token", "from the scalar encoder", YELLOW, 2.7, 0.42),
        ).arrange(DOWN, buff=0.1).move_to(LEFT * 4.35 + UP * 1.15)
        context_tag = tx("K, V · 273 context", 17, PURPLE, BOLD)
        context_tag.next_to(context, DOWN, buff=0.22)

        queries = VGroup(
            self.token_band("256 occupancy queries", "one per position", GREEN, 2.7, 0.62),
            self.token_band("1 scalar query", "", YELLOW, 2.7, 0.42),
        ).arrange(DOWN, buff=0.1).move_to(RIGHT * 4.35 + UP * 1.15)
        queries_tag = math("Q · 257", 20, GREEN)
        queries_tag.next_to(queries, UP, buff=0.3)

        block = panel("GCLCT × 8", "self-attn · cross-attn · MLP", GREEN, 2.1, 1.1,
                      20, 13)
        block.move_to(UP * 1.15)
        physics = panel("c_physics", "384 → 192", CYAN, 1.7, 0.58, 17, 13)
        physics.move_to(UP * -1.05)

        self.play(FadeIn(context), FadeIn(context_tag), run_time=0.9)
        self.play(FadeIn(queries), FadeIn(queries_tag), run_time=0.8)
        self.play(FadeIn(block), run_time=0.6)

        # cross-attention: every context band is read, the spectral goals highlighted
        cross = VGroup(
            arrow_between(context[0].get_right(), block.get_left() + UP * 0.3,
                          PURPLE, 1.6, dashed=True),
            arrow_between(context[1].get_right(), block.get_left(), CYAN, 2.6),
            arrow_between(context[2].get_right(), block.get_left() + DOWN * 0.3,
                          PURPLE, 1.6, dashed=True),
        )
        cross_tag = tx("cross-attention reads all 273", 15, PURPLE)
        cross_tag.move_to(LEFT * 2.15 + UP * 2.55)
        # self-attention among the queries
        self_loop = CurvedArrow(queries[0].get_top() + LEFT * 0.9,
                                queries[0].get_top() + RIGHT * 0.9,
                                color=GREEN, angle=-TAU / 3, stroke_width=2.0)
        self_tag = tx("queries attend to each other", 15, GREEN)
        self_tag.next_to(self_loop, UP, buff=0.08)
        phys_arrow = dashed_bracket(physics.get_top(), block.get_bottom(), CYAN, 1.6)

        self.play(Create(cross), FadeIn(cross_tag), run_time=1.0)
        q_into_block = arrow_between(queries[0].get_left(), block.get_right(),
                                     GREEN, 2.6)
        self.play(Create(q_into_block), run_time=0.6)
        self.play(Create(self_loop), FadeIn(self_tag), run_time=0.9)
        self.play(FadeIn(physics), Create(phys_arrow), run_time=0.7)

        self.cue(12)
        detail = self.gclct_detail()
        self.play(FadeOut(Group(context, context_tag, queries, queries_tag, block,
                                physics, cross, cross_tag, self_loop, self_tag,
                                phys_arrow, q_into_block)), run_time=0.6)
        self.play(FadeIn(detail), run_time=1.2)

        self.cue(13)
        out = VGroup(
            panel("Ẑ_occ", "256 × 192", GREEN, 1.75, 0.7, 18, 14),
            panel("ẑ_scalar", "1 × 192", GREEN, 1.75, 0.7, 18, 14),
        ).arrange(RIGHT, buff=0.3).move_to(RIGHT * 3.2 + DOWN * 2.5)
        self.play(FadeIn(out, shift=UP * 0.1), run_time=0.9)

    # -- small diagrams -------------------------------------------------------

    def token_band(self, label, detail, colour, width=2.6, height=0.6):
        box = RoundedRectangle(width=width, height=height, corner_radius=0.06,
                               stroke_color=colour, stroke_width=1.3,
                               fill_color=colour, fill_opacity=0.16)
        rows = [tx(label, 15, colour, BOLD)]
        if detail:
            rows.append(tx(detail, 13, MUTED))
        text = VGroup(*rows).arrange(DOWN, buff=0.02).move_to(box)
        return VGroup(box, text)

    def query_icon(self, colour=GREEN, scalar_highlight=True):
        grid = token_grid(4, 4, colour, 0.19, gap=1.15)
        if scalar_highlight:
            query = Square(side_length=0.19, stroke_width=0, fill_color=YELLOW,
                           fill_opacity=0.95)
            query.next_to(grid, DOWN, buff=0.1)
            grid.add(query)
        return grid

    def context_icon(self):
        occupancy = token_grid(4, 4, BLUE, 0.19, gap=1.15)
        goals = VGroup(*[Square(side_length=0.19, stroke_width=0,
                                fill_color=CYAN, fill_opacity=0.9)
                         for _ in range(4)]).arrange(RIGHT, buff=0.04)
        goals.next_to(occupancy, DOWN, buff=0.1)
        scalar = Square(side_length=0.19, stroke_width=0, fill_color=YELLOW,
                        fill_opacity=0.95)
        scalar.next_to(goals, DOWN, buff=0.1)
        goals.add(scalar)
        return VGroup(occupancy, goals)

    def gclct_detail(self):
        cards = VGroup(*[
            panel(title, detail, colour, 1.6, 1.0, 17, 12)
            for title, detail, colour in (
                ("norm + FiLM", "c_physics", CYAN),
                ("self-attention", "queries ↔ queries", GREEN),
                ("norm + FiLM", "c_physics", CYAN),
                ("cross-attention", "Q ← K,V", PURPLE),
                ("norm + FiLM", "c_physics", CYAN),
                ("MLP", "192 → 768 → 192", PURPLE),
            )])
        for index, card in enumerate(cards):
            card.move_to(LEFT * 5.0 + RIGHT * index * 2.0 + UP * 0.85)
        arrows = VGroup(*[
            arrow_between(cards[i].get_right(), cards[i + 1].get_left(),
                          (GREEN, CYAN, PURPLE, CYAN, PURPLE)[i], 2.0)
            for i in range(5)])

        bottom = cards[0].get_bottom()[1] - 0.6
        left = cards[0].get_left()[0] - 0.18
        right = cards[5].get_right()[0] + 0.18
        residual = VMobject(color=ORANGE, stroke_width=1.8)
        residual.set_points_as_corners([
            [left, cards[0].get_bottom()[1], 0],
            [left, bottom, 0],
            [right, bottom, 0],
            [right, cards[5].get_bottom()[1], 0],
        ])
        residual.add(Triangle(color=ORANGE, fill_opacity=1.0, stroke_width=0)
                     .scale(0.09).move_to([right, cards[5].get_bottom()[1], 0]))
        residual_tag = tx("residual around the block", 15, ORANGE)
        residual_tag.next_to(residual, DOWN, buff=0.12).shift(LEFT * 2.0)

        label = tx("one GCLCT block · repeated 8 times", 20, GREEN, BOLD)
        label.move_to(UP * 2.75)

        bus_y = cards[0].get_top()[1] + 0.5
        bus = DashedLine([-5.0, bus_y, 0], [3.0, bus_y, 0], color=CYAN,
                         stroke_width=1.3, dash_length=0.1)
        drops = VGroup(*[dashed_bracket([x, bus_y, 0], [x, cards[i].get_top()[1], 0],
                                        CYAN, 1.3)
                         for i, x in ((0, -5.0), (2, -1.0), (4, 3.0))])
        bus_tag = math("c_physics  384 → 192", 17, CYAN)
        bus_tag.move_to([-1.6, bus_y + 0.28, 0])
        return VGroup(cards, arrows, residual, residual_tag, label, bus, drops,
                      bus_tag)


# ---------------------------------------------------------------------------
# SCENE 6 — teacher, EMA, objective (58.8 s)
# ---------------------------------------------------------------------------

class S06Objective(JEPAScene):
    scene_id = "objective"

    def scene_objective(self):
        data = load_data()
        top = heading(5, "teacher, ema and objective")

        student = panel("Student", "updated by AdamW", GREEN, 2.2, 0.95, 21, 15)
        student.move_to(LEFT * 4.4 + UP * 1.85)
        teacher = panel("Teacher", "EMA copy · no gradient", VIOLET, 2.4, 0.95, 21, 15)
        teacher.move_to(RIGHT * 3.9 + UP * 1.85)
        full = occupancy_image(data["pattern"], 0.85)
        full.next_to(teacher, RIGHT, buff=0.3)
        self.cue(0)
        self.play(FadeIn(top), FadeIn(student), FadeIn(teacher), FadeIn(full),
                  run_time=1.0)

        self.cue(1)
        ema = math("θ_{t}  ←  τ θ_{t} + (1 − τ) θ_{s}", 38, VIOLET)
        ema.move_to(UP * 0.55)
        self.play(FadeIn(ema, shift=DOWN * 0.1), run_time=0.9)
        tau_track = Line(LEFT * 3.2 + DOWN * 0.25, RIGHT * 3.2 + DOWN * 0.25,
                         color=FAINT, stroke_width=5)
        knob = Dot(LEFT * 3.2 + DOWN * 0.25, radius=0.09, color=VIOLET)
        tau_labels = VGroup(tx("τ = 0.996", 18, MUTED).move_to(LEFT * 3.2 + DOWN * 0.55),
                            tx("τ = 0.999", 18, MUTED).move_to(RIGHT * 3.2 + DOWN * 0.55))
        self.play(Create(tau_track), FadeIn(knob), FadeIn(tau_labels), run_time=0.6)
        self.play(knob.animate.move_to(RIGHT * 3.2 + DOWN * 0.25), run_time=3.0,
                  rate_func=linear)

        self.cue(2)
        ema_arrow = arrow_between(student.get_right(), teacher.get_left(), VIOLET, 2.4)
        self.play(Create(ema_arrow), run_time=0.8)
        stop = tx("no gradient crosses this boundary", 20, VIOLET, BOLD)
        stop.next_to(ema_arrow, DOWN, buff=0.22)
        self.play(FadeIn(stop), run_time=0.7)
        self.signal(ema_arrow, VIOLET, run_time=0.9)

        self.cue(3)
        targets = VGroup(
            panel("z_y_raw", "stable geometry target", VIOLET, 2.5, 0.7, 18, 15),
            panel("z_y_occ_spec", "spectrum-conditioned", VIOLET, 2.5, 0.7, 18, 15),
            panel("z_y_scal", "scalar target", VIOLET, 2.5, 0.7, 18, 15),
        ).arrange(RIGHT, buff=0.25).move_to(DOWN * 1.85)
        self.play(FadeIn(targets, lag_ratio=0.15), run_time=1.0)

        # --- the objective, in three groups ---------------------------------
        self.cue(4)
        groups = VGroup(
            self.loss_group("latent", "L_inv  L_var  L_cov", PURPLE,
                            ("invariance", "variance", "covariance")),
            self.loss_group("supervision", "L_occ  L_scalar  L_summary  L_cond  L_scal_t",
                            BLUE, ("masked pixels", "unknown scalars",
                                   "summary readout", "cosine targets")),
            self.loss_group("physics", "L_phys", ORANGE,
                            ("surrogate spectrum",)),
        ).arrange(RIGHT, buff=0.35)
        groups.move_to(UP * 0.85)
        self.play(FadeOut(Group(student, teacher, full, ema, tau_track, knob,
                                tau_labels, stop, ema_arrow, targets)),
                  run_time=0.8)
        self.play(FadeIn(groups[0]), run_time=0.7)

        self.cue(5)
        self.play(FadeIn(groups[1]), run_time=0.8)
        self.cue(6)
        self.play(FadeIn(groups[2]), run_time=0.7)

        self.cue(7)
        total = math("L_{total} = Σ λ_{i} L_{i}", 34, INK).move_to(DOWN * 2.5)
        hybrid = tx("a hybrid objective — not pure self-supervision", 22, MUTED)
        hybrid.move_to(DOWN * 3.25)
        self.play(FadeIn(total), FadeIn(hybrid), run_time=0.9)

    def loss_group(self, name, formula, color, rows):
        head = tx(name.upper(), 18, color, BOLD)
        eq = math(formula, 22, color)
        items = VGroup(*[tx(f"· {row}", 17, MUTED) for row in rows]
                       ).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        body = VGroup(head, eq, items).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        frame = SurroundingRectangle(body, color=color, buff=0.22, stroke_width=1.4,
                                     corner_radius=0.08)
        frame.set_fill(PANEL, opacity=0.55)
        return VGroup(frame, body)


# ---------------------------------------------------------------------------
# SCENE 7 — the physics loop (60.0 s)
# ---------------------------------------------------------------------------

class S07Physics(JEPAScene):
    scene_id = "physics"

    def scene_physics(self):
        data = load_data()
        top = heading(6, "the physics loop")

        occ_latent = panel("Ẑ_occ", "256 × 192", GREEN, 1.0, 0.64, 15, 12)
        occ_latent.move_to(LEFT * 6.45 + UP * 2.3)
        scal_latent = panel("ẑ_scalar", "1 × 192", GREEN, 1.0, 0.64, 15, 12)
        scal_latent.move_to(LEFT * 6.45 + UP * 1.2)

        self.cue(0)
        self.play(FadeIn(top), FadeIn(occ_latent), FadeIn(scal_latent), run_time=0.9)

        self.cue(1)
        decoder = VGroup(
            panel("reshape", "16 × 16", ORANGE, 1.05, 0.64, 16, 12),
            panel("conv + FiLM", "32 × 32", ORANGE, 1.05, 0.64, 16, 12),
            panel("conv + FiLM", "64 × 64", ORANGE, 1.05, 0.64, 16, 12),
        ).arrange(RIGHT, buff=0.16).move_to(LEFT * 3.55 + UP * 2.3)
        self.play(FadeIn(decoder, lag_ratio=0.15), run_time=1.0)
        self.play(Create(arrow_between(occ_latent.get_right(), decoder.get_left(),
                                       GREEN)), run_time=0.7)
        decoded = occupancy_image(data["pattern"], 1.0)
        decoded.move_to(LEFT * 0.55 + UP * 2.3)
        decoded_tag = tx("occupancy  M̂", 17, BLUE, BOLD).next_to(decoded, DOWN, buff=0.1)
        self.play(FadeIn(decoded), FadeIn(decoded_tag),
                  Create(arrow_between(decoder.get_right(), decoded.get_left(), ORANGE)),
                  run_time=0.9)

        self.cue(2)
        heads = VGroup(*[panel(n, "head", YELLOW, 0.75, 0.58, 17, 12)
                         for n in ("l", "h", "r")]).arrange(RIGHT, buff=0.12)
        heads.move_to(LEFT * 4.4 + UP * 1.2)
        self.play(FadeIn(heads),
                  Create(arrow_between(scal_latent.get_right(), heads.get_left(),
                                       YELLOW)), run_time=0.9)
        rule = VGroup(
            tx("known → retained", 17, YELLOW),
            tx("unknown → predicted", 17, YELLOW),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        rule.next_to(heads, DOWN, buff=0.22)
        self.play(FadeIn(rule), run_time=0.7)

        self.cue(3)
        channels = Group(*[
            occupancy_image(data["pattern"], 0.85, color="orange")
            for _ in range(3)]).arrange(RIGHT, buff=0.14)
        channels.move_to(RIGHT * 2.2 + UP * 2.3)
        chan_tag = math("G_{3ch} ∈ ℝ^{3×64×64}", 22, ORANGE)
        chan_tag.next_to(channels, UP, buff=0.18)
        chan_labels = VGroup(*[tx(t, 14, MUTED) for t in ("r/5", "h", "l/3")])
        for label, channel in zip(chan_labels, channels):
            label.next_to(channel, DOWN, buff=0.06)
        self.play(FadeIn(channels), FadeIn(chan_tag), FadeIn(chan_labels),
                  run_time=1.0)
        self.play(Create(arrow_between(decoded.get_right(), channels.get_left(),
                                       ORANGE)), run_time=0.7)

        self.cue(4)
        surrogate = panel("frozen EM surrogate", "weights frozen", FROZEN, 2.1, 0.8,
                          18, 13, dashed=True)
        surrogate.move_to(RIGHT * 5.35 + UP * 2.3)
        self.play(FadeIn(surrogate), run_time=0.8)
        self.play(Create(arrow_between(channels.get_right(), surrogate.get_left(),
                                       ORANGE, buff=0.15)), run_time=0.7)

        self.cue(5)
        predicted = spectrum_curve(*data["spectrum"], width=2.6, height=0.95,
                                   stroke=1.8, colors=(ORANGE, ORANGE))
        predicted.move_to(RIGHT * 5.25 + UP * 0.55)
        target = spectrum_curve(*data["spectrum"], width=2.6, height=0.95,
                                stroke=1.1, colors=(CYAN, CYAN))
        target.move_to(predicted)
        predicted.set_opacity(0.9)
        target.set_opacity(0.5)
        self.play(Create(predicted), Create(target),
                  Create(arrow_between(surrogate.get_bottom(), predicted.get_top(),
                                       CYAN, buff=0.1)), run_time=1.4)
        legend = VGroup(
            VGroup(Dot(radius=0.045, color=ORANGE), tx("Ŝ", 16, ORANGE)).arrange(RIGHT, buff=0.08),
            VGroup(Dot(radius=0.045, color=CYAN), tx("S target", 16, CYAN)).arrange(RIGHT, buff=0.08),
        ).arrange(RIGHT, buff=0.25)
        legend.next_to(predicted, DOWN, buff=0.14)
        loss = VGroup(
            math("L_{phys} =", 22, ORANGE),
            math("SmoothL1( Ŝ/σ , S_{tgt}/σ )", 21, ORANGE),
        ).arrange(DOWN, buff=0.1)
        loss.next_to(legend, DOWN, buff=0.22)
        self.play(FadeIn(legend), FadeIn(loss), run_time=0.9)

        lock = VGroup(
            tx("weights frozen — no parameter update", 13, FROZEN),
            tx("gradient still reaches the geometry", 13, CORAL, BOLD),
        ).arrange(DOWN, buff=0.06)
        lock.next_to(surrogate, DOWN, buff=0.15)
        self.play(FadeIn(lock), run_time=0.9)
        self.play(Flash(surrogate.get_bottom(), color=CORAL, flash_radius=0.4),
                  run_time=1.0)

        self.cue(6)
        notes = VGroup(
            tx("straight-through estimator keeps occupancy hard", 17, MUTED),
            tx("full-wave Meep is a separate evaluation, not a training step", 17, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
        notes.to_corner(DL, buff=0.7)
        self.play(FadeIn(notes, lag_ratio=0.2), run_time=0.9)


# ---------------------------------------------------------------------------
# SCENE 8 — training and inference (54.3 s)
# ---------------------------------------------------------------------------

class S08Training(JEPAScene):
    scene_id = "training"

    def scene_training(self):
        top = heading(7, "training and inference")

        self.cue(0)
        grids = VGroup()
        for ratio in (0.25, 0.50, 0.75, 1.00):
            cells = {(r, c) for r in range(8) for c in range(8)
                     if ((r * 8 + c) * 37 % 100) < ratio * 100}
            grid = token_grid(8, 8, BLUE, 0.13, masked=cells)
            label = tx(f"{int(ratio * 100)} % masked", 17, BLUE, BOLD)
            grids.add(VGroup(grid, label).arrange(DOWN, buff=0.14))
        grids.arrange(RIGHT, buff=0.42).move_to(UP * 1.65)
        self.play(FadeIn(top), FadeIn(grids, lag_ratio=0.12), run_time=1.2)

        regimes = VGroup(*[
            panel(label, detail, YELLOW, 2.0, 0.66, 18, 14)
            for label, detail in (("all known", "three flags = 1"),
                                  ("mixed", "sampled per flag"),
                                  ("all unknown", "three flags = 0"))])
        regimes.arrange(RIGHT, buff=0.3).move_to(UP * 0.15)
        self.play(FadeIn(regimes, lag_ratio=0.12), run_time=1.0)

        self.cue(1)
        steps = ("sample", "mask", "drop scalars", "student", "teacher",
                 "losses", "backward", "AdamW", "EMA")
        ring = VGroup(*[panel(s, "", GREEN if i < 8 else VIOLET, 1.35, 0.5, 14, 12)
                        for i, s in enumerate(steps)])
        ring.arrange(RIGHT, buff=0.12).move_to(DOWN * 0.8).scale(0.92)
        self.play(FadeIn(ring, lag_ratio=0.05), run_time=1.2)
        tracer = Dot(ring[0].get_top() + UP * 0.12, radius=0.06, color=WHITE)
        self.add(tracer)
        self.play(tracer.animate.move_to(ring[-1].get_top() + UP * 0.12),
                  run_time=3.0, rate_func=linear)
        self.play(FadeOut(tracer), run_time=0.2)

        self.cue(2)
        axes = Axes(x_range=[0, 20000, 5000], y_range=[0, 1.1, 0.5],
                    x_length=6.4, y_length=1.5,
                    axis_config={"color": FAINT, "stroke_width": 1.2,
                                 "include_numbers": False, "include_ticks": False},
                    tips=False).move_to(DOWN * 2.62)
        physics_curve = axes.plot(lambda s: 0 if s < 2000 else min(1.0, (s - 2000) / 3000),
                                  x_range=[0, 20000], color=ORANGE, stroke_width=2.4)
        multi_curve = axes.plot(lambda s: 0 if s < 1000 else min(1.0, (s - 1000) / 2000),
                                x_range=[0, 20000], color=PURPLE, stroke_width=2.0)
        legend = VGroup(
            VGroup(Line(ORIGIN, RIGHT * 0.3, color=ORANGE, stroke_width=3),
                   tx("physics weight", 15, ORANGE)).arrange(RIGHT, buff=0.1),
            VGroup(Line(ORIGIN, RIGHT * 0.3, color=PURPLE, stroke_width=3),
                   tx("multi-target", 15, PURPLE)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.3).next_to(axes, UP, buff=0.1)
        ticks = VGroup(*[tx(label, 13, MUTED).move_to(axes.c2p(value, 0) + DOWN * 0.24)
                         for value, label in ((0, "0"), (2000, "2k"), (5000, "5k"),
                                              (10000, "10k"), (20000, "20k"))])
        marks = VGroup(*[DashedLine(axes.c2p(v, 0), axes.c2p(v, 1.0), color=ORANGE,
                                    stroke_width=1.0, dash_length=0.07)
                         for v in (2000, 5000)])
        ramp_tag = math("λ_{phys}: 0 → 3.32", 16, ORANGE)
        ramp_tag.next_to(axes.c2p(5000, 1.0), UP, buff=0.08).shift(RIGHT * 1.1)
        self.play(FadeOut(regimes), run_time=0.4)
        self.play(FadeIn(axes), FadeIn(legend), FadeIn(ticks), run_time=0.6)
        self.play(Create(multi_curve), run_time=1.0)
        self.play(Create(physics_curve), Create(marks), FadeIn(ramp_tag),
                  run_time=1.3)

        self.cue(3)
        model = panel("base-v1", "11.4 M trainable · 70k steps · 3 seeds",
                      GREEN, 4.2, 0.75, 20, 16).move_to(RIGHT * 4.3 + UP * 0.8)
        self.play(FadeIn(model, shift=LEFT * 0.15), run_time=0.9)

        self.cue(4)
        self.play(FadeOut(Group(grids, ring, axes, legend, physics_curve,
                                multi_curve, model, ticks, marks, ramp_tag)),
                  run_time=0.8)
        kept = panel("student + frozen spectrum path", "kept", GREEN, 5.0, 1.0,
                     24, 18).move_to(LEFT * 3.3 + UP * 0.6)
        gone = VGroup(
            panel("EMA teacher", "removed", VIOLET, 3.0, 0.95, 21, 17, dashed=True),
            panel("all losses", "removed", CORAL, 3.0, 0.95, 21, 17, dashed=True),
        ).arrange(DOWN, buff=0.35).move_to(RIGHT * 2.4 + UP * 0.6)
        scored = panel("surrogate", "only to score the result", FROZEN, 4.2, 0.9,
                       21, 17, dashed=True).move_to(RIGHT * 3.1 + DOWN * 1.6)
        links = arrow_between(kept.get_bottom(), scored.get_left(), FROZEN, 1.6)
        self.play(FadeIn(kept), FadeIn(gone), run_time=1.0)
        self.play(FadeIn(scored), Create(links), run_time=0.9)
        self.play(Indicate(kept, color=GREEN), run_time=1.2)


# ---------------------------------------------------------------------------
# SCENE 9 — three use cases (45.7 s)
# ---------------------------------------------------------------------------

class S09Scenarios(JEPAScene):
    scene_id = "scenarios"

    def scene_scenarios(self):
        top = heading(8, "three conditioning problems")
        self.cue(0)
        self.play(FadeIn(top), run_time=0.7)

        panels = VGroup()
        for name, ratio, known, color in (
                ("A · pure inverse design", 1.00, "all unknown", BLUE),
                ("B · partial parameters", 0.50, "one known", YELLOW),
                ("C · retrofit", 0.25, "all known", GREEN)):
            panels.add(self.scenario_panel(name, ratio, known, color))
        panels.arrange(RIGHT, buff=0.45).move_to(DOWN * 0.15)

        for slot in range(3):
            self.cue(slot + 1)
            self.play(FadeIn(panels[slot], shift=UP * 0.12), run_time=1.0)
            self.play(self.fill_animation(panels[slot]), run_time=1.6)

        self.cue(5)
        schematic = tx("schematic fill · the retention property is measured: "
                       "visible region max |Δ| = 0", 18, MUTED)
        schematic.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(schematic), run_time=0.8)

    def scenario_panel(self, name, ratio, known, color):
        frame = RoundedRectangle(width=3.9, height=4.4, corner_radius=0.12,
                                 stroke_color=color, stroke_width=1.5,
                                 fill_color=PANEL, fill_opacity=0.9)
        grid = token_grid(10, 10, BLUE, 0.24, gap=1.0)
        masked = {(r, c) for r in range(10) for c in range(10)
                  if ((r * 7919 + c * 104729) % 997) / 997 < ratio}
        masked_grid = token_grid(10, 10, BLUE, 0.24, masked=masked, gap=1.0)
        masked_grid.move_to(grid)
        title = tx(name, 19, color, BOLD)
        detail = VGroup(tx(f"occupancy masked {int(ratio * 100)} %", 16, MUTED),
                        tx(f"scalars: {known}", 16, YELLOW))
        detail.arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        content = VGroup(title, masked_grid, detail).arrange(DOWN, buff=0.24)
        content.move_to(frame)
        panel_group = VGroup(frame, content)
        panel_group.masked_cells = masked
        return panel_group

    def fill_animation(self, panel_group):
        """Fill only the hidden cells, leaving visible cells untouched."""
        masked_grid = panel_group[1][1]
        hidden = [(r, c) for (r, c) in sorted(panel_group.masked_cells)]
        animations = []
        for r, c in hidden[:40]:
            index = r * 10 + c
            cell = masked_grid[index]
            animations.append(cell.animate.set_fill(GREEN, opacity=0.75)
                              .set_stroke(GREEN, width=1.1))
        return LaggedStart(*animations, lag_ratio=0.02) if animations else Wait(0.1)


# ---------------------------------------------------------------------------
# SCENE 10 — measured results (66 s)
# ---------------------------------------------------------------------------

class S10Results(JEPAScene):
    scene_id = "results"

    def scene_results(self):
        top = heading(9, "measured results")
        self.cue(0)
        self.play(FadeIn(top), run_time=0.7)
        provenance = source_note("held-out test split · n = 17,489 · frozen surrogate")
        provenance.to_corner(UR, buff=0.42)
        self.play(FadeIn(provenance), run_time=0.6)

        # --- size: the same task, 2.9x fewer trainable parameters ------------
        size, size_note = self.size_chart()
        self.reveal(size, direction=UP, shift=0.18, run_time=1.0, lag=0.06)
        self.play(Write(size_note), run_time=0.9)

        # --- cost: one pass against 500 denoising steps ----------------------
        self.cue(1)
        cost = self.cost_chart()
        self.play(FadeIn(cost), run_time=1.0)

        # --- fidelity --------------------------------------------------------
        self.cue(2)
        self.play(FadeOut(Group(size, size_note, cost)), run_time=0.8)
        chart, value = self.mae_chart()
        self.play(FadeIn(chart), run_time=0.9)
        self.play(Write(value), run_time=1.1)
        aae = math("AAE = 29.51 ± 0.74", 30, CYAN).next_to(value, DOWN, buff=0.35)
        self.play(FadeIn(aae), run_time=0.6)

        # --- conditioning gates ----------------------------------------------
        self.cue(3)
        self.play(FadeOut(Group(chart, value, aae)), run_time=0.8)
        question = tx("is the error with the correct target lower than with a "
                      "shuffled one?", 23, MUTED, wrap=64).move_to(UP * 2.5)
        bars = self.gate_chart()
        self.play(FadeIn(question), FadeIn(bars), run_time=1.0)

        # --- objective ablation ----------------------------------------------
        self.cue(4)
        self.play(FadeOut(Group(bars, question)), run_time=0.8)
        ablation = self.ablation_chart()
        self.play(FadeIn(ablation), run_time=1.0)

        # --- head to head ----------------------------------------------------
        self.cue(5)
        matched = self.metadit_chart()
        self.play(FadeOut(ablation), run_time=0.7)
        self.play(FadeIn(matched), run_time=1.0)

    # -- charts ---------------------------------------------------------------

    def size_chart(self):
        """Trainable parameters: 11.40 M against 32.57 M (measured live)."""
        rows = (("unified JEPA", 11.40, GREEN), ("MetaDiT-S", 32.57, CYAN))
        scale = 0.25
        group = VGroup()
        ends = []
        for index, (name, value, colour) in enumerate(rows):
            y = 1.45 - index * 1.2
            bar = Rectangle(width=value * scale, height=0.52, stroke_color=colour,
                            stroke_width=1.6, fill_color=colour, fill_opacity=0.32)
            bar.move_to(np.array([-6.3 + value * scale / 2, y, 0]))
            name_text = tx(name, 18, colour, BOLD)
            name_text.move_to(np.array([-6.3, y + 0.46, 0]), aligned_edge=LEFT)
            value_text = math(f"{value:.1f} M", 22, INK)
            value_text.move_to(np.array([-6.3 + 0.18, y, 0]), aligned_edge=LEFT)
            group.add(bar, name_text, value_text)
            ends.append(value * scale)

        # bracket the two bar ends: the size ratio
        x_short = -6.3 + ends[0]
        x_long = -6.3 + ends[1]
        bracket = Line(np.array([x_short, 1.45, 0]), np.array([x_long, 1.45, 0]),
                       color=CORAL, stroke_width=1.2)
        drop = Line(np.array([x_long, 1.45, 0]), np.array([x_long, 0.25, 0]),
                    color=CORAL, stroke_width=1.2)
        ratio = math("2.9 × fewer", 22, CORAL)
        ratio.next_to(bracket, UP, buff=0.12)
        group.add(bracket, drop, ratio)

        caption = tx("trainable parameters", 18, MUTED)
        caption.move_to(np.array([-6.3, 2.45, 0]), aligned_edge=LEFT)
        group.add(caption)
        note = math("MAE 0.0490  vs  0.0803", 26, INK)
        note.move_to(DOWN * 1.35)
        return group, note

    def cost_chart(self):
        """One forward pass against a 500-step sampler."""
        group = VGroup()
        ours = VGroup(
            Dot(radius=0.09, color=GREEN),
            tx("1 forward pass", 20, GREEN, BOLD),
        ).arrange(RIGHT, buff=0.18).move_to(LEFT * 4.0 + DOWN * 2.15)
        ticks = VGroup(*[Line(ORIGIN, UP * 0.28, color=CYAN, stroke_width=1.2)
                         for _ in range(28)]).arrange(RIGHT, buff=0.055)
        ticks.next_to(ours, RIGHT, buff=1.1)
        ticks.shift(RIGHT * 0.2)
        theirs = tx("500 denoising steps", 20, CYAN, BOLD).next_to(ticks, DOWN, buff=0.16)
        group.add(ours, ticks, theirs)
        return group

    def mae_chart(self):
        seeds = [0.0500, 0.0494, 0.0476]
        axis = Line(LEFT * 5.4, RIGHT * 5.4, color=FAINT, stroke_width=1.4)
        axis.move_to(LEFT * 0.6 + DOWN * 0.75)
        origin = axis.get_start()
        span = axis.get_end()[0] - axis.get_start()[0]
        lo, hi = 0.0455, 0.0535

        def x_of(v):
            return origin[0] + span * (v - lo) / (hi - lo)

        y_axis = axis.get_center()[1]
        band = Rectangle(width=x_of(0.0503) - x_of(0.0477), height=0.7,
                         stroke_width=0, fill_color=GREEN, fill_opacity=0.14)
        band.move_to((x_of(0.0490), y_axis, 0))
        mean_line = Line(UP * 0.5, DOWN * 0.5, color=GREEN, stroke_width=2.4)
        mean_line.move_to((x_of(0.0490), y_axis, 0))
        dots = VGroup(*[Dot((x_of(v), y_axis, 0), radius=0.075, color=GREEN)
                        for v in seeds])
        ticks = VGroup(*[tx(f"{v:.3f}", 16, MUTED).move_to((x_of(v), y_axis - 0.72, 0))
                         for v in (0.046, 0.050, 0.054)])
        caption = tx("test MAE · three training seeds", 18, MUTED)
        caption.next_to(axis, UP, buff=0.35)
        seed_note = tx("per-seed:  0.0500  ·  0.0494  ·  0.0476", 17, MUTED)
        seed_note.next_to(axis, DOWN, buff=1.15)
        value = math("MAE = 0.0490 ± 0.0013", 46, GREEN)
        value.move_to(UP * 2.45)
        return VGroup(axis, band, mean_line, dots, ticks, caption, seed_note), value

    def gate_chart(self):
        values = [0.9967, 0.9772, 0.9303]
        errors = [0.0011, 0.0040, 0.0110]
        names = ["A", "B", "C"]
        colors = [BLUE, YELLOW, GREEN]
        base_y = -1.5
        height = 2.6
        group = VGroup()
        threshold = Line(LEFT * 6.2, RIGHT * 6.2, color=CORAL, stroke_width=1.4)
        threshold.move_to(UP * (base_y + height * 0.75))
        group.add(threshold)
        gate_label = tx("gate 0.75", 16, CORAL)
        gate_label.move_to(threshold.get_end() + LEFT * 0.85 + UP * 0.2)
        group.add(gate_label)
        for i, (value, error, name, color) in enumerate(zip(values, errors, names, colors)):
            bar = Rectangle(width=0.85, height=height * (value - 0.70) / 0.35,
                            stroke_color=color, stroke_width=1.6,
                            fill_color=color, fill_opacity=0.35)
            bar.move_to((LEFT * 4.4 + RIGHT * i * 2.4) + UP * (base_y + bar.height / 2))
            whisker = Line(bar.get_top() + LEFT * 0.24, bar.get_top() + RIGHT * 0.24,
                           color=color, stroke_width=2.0)
            whisker.move_to(bar.get_top() + UP * 0.05)
            value_text = math(f"{value:.4f} ± {error:.4f}", 19, color)
            value_text.next_to(bar, UP, buff=0.42)
            name_text = tx(f"scenario {name}", 19, color, BOLD).next_to(bar, DOWN, buff=0.18)
            group.add(bar, whisker, value_text, name_text)
        axis = Line(LEFT * 6.2, LEFT * 6.2 + UP * (height + 0.2), color=FAINT,
                    stroke_width=1.2).move_to(LEFT * 6.2 + UP * (base_y + height / 2))
        group.add(axis)
        return group

    def ablation_chart(self):
        group = VGroup()
        base_y = -1.9
        for i, (name, value, seeds, color) in enumerate((
                ("unified JEPA", 0.0490, [0.0500, 0.0494, 0.0476], GREEN),
                ("conventional objective", 0.0922, [0.0914, 0.0730, 0.1121], CORAL))):
            bar = Rectangle(width=1.5, height=3.0 * value / 0.10,
                            stroke_color=color, stroke_width=1.6,
                            fill_color=color, fill_opacity=0.3)
            bar.move_to((LEFT * 3.0 + RIGHT * i * 5.2) + UP * (base_y + bar.height / 2))
            value_text = math(f"{value:.4f}", 30, color).next_to(bar, UP, buff=0.3)
            name_text = tx(name, 20, color, BOLD).next_to(bar, DOWN, buff=0.2)
            dots = VGroup(*[Dot(bar.get_top() + RIGHT * ((j - 1) * 0.24) + UP * 0.02
                                + UP * 3.0 * (s - value) / 0.10,
                                radius=0.05, color=color) for j, s in enumerate(seeds)])
            group.add(bar, value_text, name_text, dots)
        group.add(tx("test MAE · architecture held fixed, only the objective changes",
                     19, MUTED).move_to(UP * 2.45))
        axis = Line(LEFT * 6.2, RIGHT * 6.2, color=FAINT, stroke_width=1.2)
        axis.move_to(UP * base_y)
        group.add(axis)
        return group

    def metadit_chart(self):
        group = VGroup()
        base_y = -1.9
        for i, (name, value, color) in enumerate((
                ("reproduced MetaDiT-S", 0.0803, CYAN),
                ("unified JEPA", 0.0490, GREEN))):
            bar = Rectangle(width=1.6, height=3.0 * value / 0.10,
                            stroke_color=color, stroke_width=1.6,
                            fill_color=color, fill_opacity=0.3)
            bar.move_to((LEFT * 2.4 + RIGHT * i * 5.0) + UP * (base_y + bar.height / 2))
            value_text = math(f"{value:.4f}", 32, color).next_to(bar, UP, buff=0.28)
            name_text = tx(name, 20, color, BOLD).next_to(bar, DOWN, buff=0.2)
            group.add(bar, value_text, name_text)
        group.add(tx("test MAE · same split, same released surrogate", 20, MUTED)
                  .move_to(UP * 2.6))
        axis = Line(LEFT * 6.0, RIGHT * 6.0, color=FAINT, stroke_width=1.2)
        axis.move_to(UP * base_y)
        group.add(axis)
        return group


# ---------------------------------------------------------------------------
# SCENE 11 — why it works, and where it transfers (55 s)
# ---------------------------------------------------------------------------

class S11Transfer(JEPAScene):
    scene_id = "transfer"

    def scene_transfer(self):
        data = load_data()
        top = heading(10, "why it works, and where it transfers")
        self.cue(0)
        self.play(FadeIn(top), run_time=0.8)

        # --- one response, many designs ---------------------------------------
        pair = Group(
            occupancy_image(data["pattern"], 1.2),
            occupancy_image(data["neighbour_pattern"], 1.2),
        ).arrange(RIGHT, buff=0.45).move_to(LEFT * 5.0 + UP * 1.3)
        pair_tag = tx("one response", 17, CYAN, BOLD).next_to(pair, UP, buff=0.18)
        distance = math(f"ΔS = {100 * data['spectrum_distance']:.1f} %", 20, CYAN)
        distance.next_to(pair, DOWN, buff=0.22)
        self.play(FadeIn(pair), FadeIn(pair_tag), FadeIn(distance), run_time=1.0)

        # --- pixels vs representation -----------------------------------------
        self.cue(1)
        pixels = self.pixel_field()
        pixels.move_to(RIGHT * 1.4 + UP * 1.3)
        pixels_tag = VGroup(
            tx("a pixel generator spends capacity here", 16, CORAL, wrap=30),
            tx("detail the physics does not fix", 16, MUTED, wrap=30),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        pixels_tag.next_to(pixels, DOWN, buff=0.2)
        self.play(FadeIn(pixels), FadeIn(pixels_tag), run_time=1.0)

        self.cue(2)
        latent = self.latent_vector()
        latent.move_to(RIGHT * 1.2 + UP * 1.15)
        latent_tag = VGroup(
            tx("JEPA predicts this instead", 16, PURPLE, BOLD),
            tx("a representation under the physics condition", 16, MUTED, wrap=34),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        latent_tag.next_to(latent, DOWN, buff=0.2)
        chips = VGroup(
            panel("inverse", "", PURPLE, 1.1, 0.42, 14, 12),
            panel("variance", "no collapse", PURPLE, 1.6, 0.42, 14, 12),
            panel("covariance", "", PURPLE, 1.3, 0.42, 14, 12),
        ).arrange(RIGHT, buff=0.16).next_to(latent_tag, DOWN, buff=0.25)
        self.play(FadeOut(Group(pixels, pixels_tag)), run_time=0.7)
        self.play(FadeIn(latent), FadeIn(latent_tag), FadeIn(chips), run_time=1.0)

        # --- the hard stratum --------------------------------------------------
        self.cue(3)
        gate = VGroup(
            math("0.9974", 54, GREEN),
            tx("real target beats a shuffled one · everything masked", 19, MUTED),
        ).arrange(DOWN, buff=0.2).move_to(DOWN * 0.2)
        gate_tag = tx("hard stratum · n = 1,536 paired items", 16, MUTED)
        gate_tag.next_to(gate, DOWN, buff=0.18)
        self.play(FadeOut(Group(pair, pair_tag, distance, latent, latent_tag, chips)),
                  run_time=0.8)
        self.play(FadeIn(gate), FadeIn(gate_tag), run_time=1.0)

        # --- end to end, with the whole model in view --------------------------
        self.cue(4)
        chain, packet_path = self.full_model_chain(data)
        self.play(FadeOut(Group(gate, gate_tag)), run_time=0.7)
        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.scale(0.45).move_to(chain[0].get_center()),
                  run_time=0.6, rate_func=smooth)
        self.play(FadeIn(chain), run_time=0.6)
        self.play(Restore(self.camera.frame), run_time=1.4, rate_func=smooth)

        packet = Dot(packet_path.get_start(), radius=0.1, color=WHITE).set_z_index(9)
        halo = Circle(radius=0.24, stroke_color=WHITE, stroke_width=2,
                      stroke_opacity=0.5).move_to(packet)
        halo.add_updater(lambda m: m.move_to(packet.get_center()))
        self.add(halo, packet)
        self.play(MoveAlongPath(packet, packet_path, rate_func=linear), run_time=2.2)
        self.play(FadeOut(VGroup(packet, halo)), run_time=0.3)
        design = occupancy_image(data["pattern"], 1.15)

        self.cue(5)
        transfer = VGroup(
            panel("inverse design", "other domains, other physics", BLUE, 3.4, 1.0, 18, 14),
            panel("scientific surrogates", "predict the state, not the field", FROZEN,
                  3.6, 1.0, 18, 14),
            panel("world models", "plan in representation space", PURPLE, 3.4, 1.0, 18, 14),
        ).arrange(RIGHT, buff=0.45).move_to(DOWN * 0.35)
        self.play(FadeOut(chain), FadeIn(transfer, lag_ratio=0.15), run_time=1.2)

    # -- pieces ---------------------------------------------------------------

    def pixel_field(self):
        """A grid where the ignored pixels visibly flicker: capacity spent on noise."""
        cells = VGroup()
        rng = np.random.RandomState(3)
        for r in range(10):
            for c in range(10):
                noise = rng.rand() < 0.35
                square = Square(side_length=0.19, stroke_width=0.8,
                                stroke_color=CORAL if noise else FAINT,
                                fill_color=CORAL if noise else PANEL,
                                fill_opacity=0.55 if noise else 0.8)
                square.move_to(((c - 4.5) * 0.21, (4.5 - r) * 0.21, 0))
                cells.add(square)
        return cells

    def latent_vector(self):
        bars = VGroup(*[Rectangle(width=0.085, height=0.45 + 0.3 * abs(np.sin(i)),
                                  stroke_width=0, fill_color=PURPLE, fill_opacity=0.85)
                        for i in range(14)]).arrange(RIGHT, buff=0.07)
        frame = SurroundingRectangle(bars, color=PURPLE, buff=0.14, stroke_width=1.2,
                                     corner_radius=0.05)
        return VGroup(frame, bars)

    def full_model_chain(self, data):
        spectrum_icon = spectrum_curve(*data["spectrum"], width=1.0, height=0.55,
                                       axes=False, stroke=1.6)
        chips = Group(
            panel("S target", "", CYAN, 1.55, 0.62, 15, 12),
            panel("fusion", "273", PURPLE, 1.45, 0.62, 15, 12),
            panel("predictor", "257 · ×8", GREEN, 1.6, 0.62, 15, 12),
            panel("decoder", "", ORANGE, 1.4, 0.62, 15, 12),
            occupancy_image(data["pattern"], 0.95),
            panel("surrogate", "frozen", FROZEN, 1.45, 0.62, 14, 12),
            spectrum_icon,
        )
        xs = [-5.95, -4.05, -2.05, -0.05, 1.85, 3.75, 5.55]
        for chip, x in zip(chips, xs):
            chip.move_to((x, 0.35, 0))
        arrows = VGroup()
        for left, right in zip(chips[:-1], chips[1:]):
            arrows.add(arrow_between(left.get_right(), right.get_left(), MUTED, 1.8))
        tag = tx("one forward pass through the whole model", 18, MUTED)
        tag.next_to(chips, UP, buff=0.55)
        path = VMobject().set_points_as_corners([
            [chips[0].get_left()[0], 1.0, 0], [chips[-1].get_right()[0], 1.0, 0]])
        return Group(chips, arrows, tag), path


# ---------------------------------------------------------------------------
# SCENE 12 — the code behind it (39 s)
# ---------------------------------------------------------------------------

MONO = "Consolas"
CODE_INK = "#D7E2F4"


class S12Code(JEPAScene):
    scene_id = "code"

    def code_line(self, text, size=20):
        """One source line; a trailing comment is set in a quieter colour."""
        if "  #" in text:
            code, comment = text.split("  #", 1)
            left = Text(code, font=MONO, font_size=size, color=CODE_INK)
            right = Text("#" + comment, font=MONO, font_size=size, color=MUTED)
            return VGroup(left, right).arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        return Text(text, font=MONO, font_size=size, color=CODE_INK).set_color(CODE_INK)

    def snippet_window(self):
        frame = RoundedRectangle(width=9.3, height=5.0, corner_radius=0.1,
                                 stroke_color=FAINT, stroke_width=1.4,
                                 fill_color="#0A1424", fill_opacity=1.0)
        frame.move_to(LEFT * 2.15 + UP * 0.15)
        bar = Line(frame.get_corner(UL) + DOWN * 0.42,
                   frame.get_corner(UR) + DOWN * 0.42,
                   color=FAINT, stroke_width=1.0)
        dots = VGroup(*[Dot(radius=0.05, color=colour)
                        for colour in ("#FF8294", "#FFD166", "#68E29A")])
        dots.arrange(RIGHT, buff=0.1)
        dots.move_to(frame.get_corner(UL) + RIGHT * 0.28 + DOWN * 0.21)
        return VGroup(frame, bar, dots)

    def build_snippet(self, path_text, lines, highlight):
        """Lay the lines out by hand so source indentation survives Pango."""
        left_x = -6.6
        char = 0.0835          # Consolas advance at 20 pt, in scene units
        top = 1.95
        step = 0.36
        mobs, bands = VGroup(), VGroup()
        for index, raw in enumerate(lines):
            indent = len(raw) - len(raw.lstrip(" "))
            mob = self.code_line(raw.strip())
            mob.move_to(np.array([left_x + indent * char, top - index * step, 0]),
                        aligned_edge=LEFT)
            mobs.add(mob)
            if index in highlight:
                band = Rectangle(width=8.7, height=0.42, stroke_width=0,
                                 fill_color=GREEN, fill_opacity=0.12)
                band.move_to(np.array([-2.2, top - index * step, 0]))
                bands.add(band)
        path = Text(path_text, font=MONO, font_size=16, color=CYAN)
        path.move_to(np.array([left_x + 0.62, 2.42, 0]), aligned_edge=LEFT)
        return VGroup(bands, mobs, path)

    def scene_code(self):
        top = heading(11, "the code behind it")
        window = self.snippet_window()
        self.cue(0)
        self.play(FadeIn(top), FadeIn(window), run_time=0.9)

        snippets = [
            ("src/encoders/occupancy_encoder.py",
             ["x = self.patch_embed(occupancy).flatten(2).transpose(1, 2)",
              "x = x + self.pos_embed",
              "for i, block in enumerate(self.blocks):",
              "    x = block(x)",
              "    gamma, beta = film_params[i]",
              "    x = gamma[:, None, :] * x + beta[:, None, :]"],
             [0, 2, 4, 5],
             "one convolution, six blocks, one FiLM per block"),
            ("src/assembly.py  ·  UnifiedJEPA.forward",
             ["z_x = self.occupancy_encoder(occupancy, film_params=film_params,",
              "                             mask=mask, mask_token=self.mask_token)",
              "fused = self.fusion_encoder(z_x, a_goal, scalar_summary)  # 273",
              "queries = torch.cat([occ_queries, scalar_query], dim=1)  # 257",
              "z_hat_raw, attn = self.predictor(queries, fused, c_physics)"],
             [2, 3, 4],
             "273 context tokens in, 257 queries out"),
            ("src/predictor/gclct.py  ·  GCLCTBlock.forward",
             ["h = self.norm1(x)",
              "h = h * (1.0 + gamma1[:, None, :]) + beta1[:, None, :]",
              "x = x + self.self_attn(h)",
              "h = self.norm2(x)",
              "h = h * (1.0 + gamma2[:, None, :]) + beta2[:, None, :]",
              "cross_out, weights = self.cross_attn(h, kv)",
              "x = x + cross_out",
              "h = self.norm3(x)",
              "x = x + self.mlp(h)"],
             [2, 5, 8],
             "normalise, condition, attend, cross-attend, MLP"),
            ("src/losses/unified_losses.py  ·  physics term",
             ["geometry, soft_occ = model.decode_geometry(",
              '    out["z_hat"], out["scalar_pred"], occ_input=occ,',
              "    mask=mask, scalar_known=sk, scalar_values=sv)",
              "result = surrogate(geometry)  # frozen EM surrogate",
              "spectrum_pred = result.prediction",
              "L_phys, _, _ = physics_loss_from_out(",
              "    model, out, self.surrogate, occupancy, scalar_values,",
              "    scalar_known, spectrum, mask, loss_type=\"smooth_l1\")"],
             [0, 3, 5],
             "decode, assemble, score with the frozen surrogate"),
        ]

        current = None
        for index, (path_text, lines, highlight, caption) in enumerate(snippets):
            self.cue(index + 1)
            if current is not None:
                self.play(FadeOut(current), run_time=0.35)
            snippet = self.build_snippet(path_text, lines, highlight)
            caption_mob = tx(caption, 18, MUTED, wrap=30)
            caption_mob.move_to(RIGHT * 5.2 + UP * 0.9)
            caption_tag = tx("what it does", 15, CYAN, BOLD)
            caption_tag.next_to(caption_mob, UP, buff=0.2)
            snippet.add(caption_mob, caption_tag)
            self.play(FadeIn(snippet[2]), run_time=0.3)
            self.play(LaggedStart(*[FadeIn(line, shift=RIGHT * 0.08)
                                    for line in snippet[1]], lag_ratio=0.08),
                      FadeIn(caption_mob), FadeIn(caption_tag), run_time=1.1)
            self.play(FadeIn(snippet[0]), run_time=0.5)
            current = snippet

        self.cue(6)
        self.play(FadeOut(current), run_time=0.6)
        files = VGroup(*[Text(name, font=MONO, font_size=18, color=CYAN)
                         for name in ("src/assembly.py", "src/encoders/occupancy_encoder.py",
                                      "src/predictor/gclct.py", "src/losses/unified_losses.py",
                                      "src/physics/physics_loop.py")])
        files.arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        files.move_to(LEFT * 2.2 + UP * 0.3)
        banner = tx("every number in this film comes from these files", 26, INK, BOLD)
        banner.next_to(files, DOWN, buff=0.45)
        self.play(LaggedStart(*[FadeIn(f, shift=RIGHT * 0.1) for f in files],
                              lag_ratio=0.12), run_time=1.2)
        self.play(FadeIn(banner, shift=UP * 0.1), run_time=0.9)


# ---------------------------------------------------------------------------
# SCENE 13 — closing (19.7 s + 4 s hold)
# ---------------------------------------------------------------------------

class S13Closing(JEPAScene):
    scene_id = "closing"

    def scene_closing(self):
        self.cue(0)
        chain = VGroup(
            panel("G + p + S", "inputs", BLUE, 2.1, 0.8, 18, 14),
            panel("fusion + predictor", "273 · 257", PURPLE, 2.5, 0.8, 18, 14),
            panel("decoder", "geometry", ORANGE, 2.1, 0.8, 18, 14),
            panel("surrogate", "physics", FROZEN, 2.1, 0.8, 18, 14),
        ).arrange(RIGHT, buff=0.85).move_to(UP * 2.65)
        arrows = VGroup(*[arrow_between(chain[i].get_right(), chain[i + 1].get_left())
                          for i in range(3)])
        equation = math("S_{target}  →  Z  →  Ĝ", 40, INK).move_to(UP * 1.35)
        self.play(FadeIn(chain, lag_ratio=0.1), run_time=1.2)
        self.play(LaggedStart(*[Create(a) for a in arrows], lag_ratio=0.15),
                  run_time=0.8)
        self.play(FadeIn(equation), run_time=0.8)

        self.cue(1)
        facts_a = VGroup(
            tx("11.4 M trainable parameters · one forward pass", 22, GREEN, BOLD),
            tx("test MAE 0.0490 ± 0.0013   ·   AAE 29.51 ± 0.74", 21, MUTED),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
        facts_a.move_to(UP * 0.15)
        self.reveal(*facts_a, direction=UP, run_time=1.2, lag=0.25)

        self.cue(2)
        facts_b = VGroup(
            tx("scenario gates A / B / C:   0.9967  ·  0.9772  ·  0.9303", 21, MUTED),
            tx("JEPA objective vs conventional:   0.0490  vs  0.0922", 21, PURPLE),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.16)
        facts_b.next_to(facts_a, DOWN, buff=0.3).align_to(facts_a, LEFT)
        self.reveal(*facts_b, direction=UP, run_time=1.1, lag=0.25)

        self.cue(3)
        motto = title_line("Learning the representations behind physical design", 34)
        motto.move_to(DOWN * 2.05)
        credits = VGroup(
            tx("Unified JEPA for Electromagnetic Metamaterial Inverse Design", 18, MUTED),
            tx("github.com/tegga8/metasurface-jepa", 17, CYAN),
        ).arrange(DOWN, buff=0.08).to_corner(DL, buff=0.4)
        self.play(FadeIn(motto, shift=UP * 0.12), run_time=1.2)
        self.play(FadeIn(credits), run_time=0.8)

        self.wait(4.0)
        self.play(FadeOut(Group(chain, arrows, equation, facts_a, facts_b, motto,
                                credits)), run_time=1.6)


# ---------------------------------------------------------------------------
# Scene classes
# ---------------------------------------------------------------------------

class JEPAProjectPresentation(JEPAScene):
    """The whole film in one file (all eleven scenes, in order)."""

    def construct(self):
        self.camera.background_color = BG
        for scene_id in SCENE_ORDER:
            self.run_scene(scene_id)


for _index, _scene_id in enumerate(SCENE_ORDER, start=1):
    _name = f"S{_index:02d}" + _scene_id.capitalize()
    if _name not in globals():
        globals()[_name] = type(_name, (JEPAScene,), {"scene_id": _scene_id})
