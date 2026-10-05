"""An implementation-faithful visual walkthrough of the unified 192-D JEPA.

Fast preview:
    python -m manim -pql manim_animation.py JEPAFullPresentation

Final render:
    python -m manim -pqh manim_animation.py JEPAFullPresentation

This film intentionally follows the active code path in ``src/assembly.py``
and ``scripts/train/train_unified.py`` rather than an abstract JEPA sketch.
"""

from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# Palette and typography
# ---------------------------------------------------------------------------

BG = "#09101F"
PANEL = "#101B32"
PANEL_2 = "#14223F"
INK = "#F3F7FF"
MUTED = "#A9B6D4"
FAINT = "#35435F"
BLUE = "#5CA8FF"       # occupancy geometry
CYAN = "#55E0F7"       # spectrum / frozen physics
YELLOW = "#FFD166"     # scalar path / FiLM
PURPLE = "#B497FF"     # latent / target
GREEN = "#68E29A"      # trainable prediction
RED = "#FF7F91"        # loss / masking
ORANGE = "#FFAA70"

FONT = "Segoe UI"
MATH_FONT = "Cambria Math"
config.background_color = BG


def txt(value, size=24, color=INK, weight=NORMAL):
    return Text(value, font=FONT, font_size=size, color=color, weight=weight)


def formula(value, size=24, color=INK):
    return Text(value, font=MATH_FONT, font_size=size, color=color)


def heading(title_text, eyebrow=None):
    items = []
    if eyebrow:
        items.append(txt(eyebrow.upper(), 15, CYAN, BOLD))
    items.append(txt(title_text, 38, INK, BOLD))
    group = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.08)
    group.to_edge(UP, buff=0.30)
    return group


def caption(value, color=MUTED, size=18):
    label = txt(value, size, color)
    label.to_edge(DOWN, buff=0.26)
    return label


def panel(width, height, color=FAINT, fill=PANEL, stroke_width=2.0):
    return RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.12,
        stroke_color=color,
        stroke_width=stroke_width,
        fill_color=fill,
        fill_opacity=0.97,
    )


def label_panel(name, detail, color, width=2.2, height=0.86):
    box = panel(width, height, color)
    title = txt(name, 17, INK, BOLD)
    sub = txt(detail, 11, MUTED)
    words = VGroup(title, sub).arrange(DOWN, buff=0.06).move_to(box)
    return VGroup(box, words)


def arrow_between(a, b, color=MUTED, width=2.2, buff=0.08):
    return Arrow(
        a, b, buff=buff, color=color, stroke_width=width,
        max_tip_length_to_length_ratio=0.15,
    )


def pulse(scene, point, color, radius=0.08, run_time=0.55):
    dot = Dot(point, radius=radius, color=color)
    halo = Dot(point, radius=radius * 2.4, color=color, fill_opacity=0.18)
    token = VGroup(halo, dot)
    scene.play(FadeIn(token, scale=0.4), Flash(point, color=color, flash_radius=0.26), run_time=run_time)
    scene.play(FadeOut(token, scale=1.7), run_time=0.18)


def travel(scene, start, end, color=GREEN, run_time=0.65):
    halo = Dot(start, radius=0.18, color=color, fill_opacity=0.17)
    dot = Dot(start, radius=0.07, color=color)
    token = VGroup(halo, dot)
    scene.add(token)
    scene.play(MoveAlongPath(token, Line(start, end)), run_time=run_time, rate_func=linear)
    scene.play(Flash(end, color=color, flash_radius=0.22, line_length=0.10), run_time=0.22)
    scene.remove(token)


def fade_scene(scene, duration=0.45):
    if scene.mobjects:
        scene.play(*[FadeOut(mob) for mob in scene.mobjects], run_time=duration)


# ---------------------------------------------------------------------------
# Real-world data visualisations
# ---------------------------------------------------------------------------


def occupancy_map(rows=12, cols=12, side=0.15, seed=7, hidden=False):
    """Readable proxy for M[64,64]; a red patch marks masked 4x4-token regions."""
    rng = np.random.default_rng(seed)
    active = rng.random((rows, cols)) > 0.57
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            in_mask = hidden and 3 <= r <= 8 and 4 <= c <= 9
            square = Square(
                side_length=side,
                stroke_color="#405276",
                stroke_width=0.45,
                fill_color=("#20233E" if in_mask else (BLUE if active[r, c] else "#11192B")),
                fill_opacity=1,
            )
            square.move_to(((c - (cols - 1) / 2) * side, ((rows - 1) / 2 - r) * side, 0))
            cells.add(square)
    if not hidden:
        return cells
    mask = RoundedRectangle(
        width=6.05 * side, height=6.05 * side, corner_radius=0.04,
        stroke_color=RED, stroke_width=2.0, fill_color="#171A30", fill_opacity=0.82,
    )
    mask.move_to(cells.get_center() + RIGHT * 1.5 * side + DOWN * 0.5 * side)
    q = txt("?", 33, RED, BOLD).move_to(mask)
    return VGroup(cells, mask, q)


def token_grid(rows=4, cols=8, color=PURPLE, side=0.13, missing=False):
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            is_missing = missing and (r + 2 * c) % 5 == 0
            cell = RoundedRectangle(
                width=side * 1.28, height=side * 1.28, corner_radius=0.025,
                stroke_color=(RED if is_missing else color), stroke_width=0.6,
                fill_color=("#281A2E" if is_missing else color),
                fill_opacity=(0.45 if is_missing else 0.88),
            )
            cell.move_to(((c - (cols - 1) / 2) * side * 1.47, ((rows - 1) / 2 - r) * side * 1.47, 0))
            cells.add(cell)
    return cells


def spectrum(width=2.4, height=0.9, color=CYAN, seed=2):
    rng = np.random.default_rng(seed)
    axes = Axes(
        x_range=[0, 1, 0.2], y_range=[0, 1, 0.5], x_length=width, y_length=height,
        axis_config={"stroke_color": FAINT, "stroke_width": 1}, tips=False,
    )
    x = np.linspace(0, 1, 100)
    y = 0.18 + 0.10 * np.sin(13 * x) + 0.06 * np.cos(29 * x)
    for center, amp in [(0.18, 0.33), (0.46, 0.28), (0.74, 0.42)]:
        y += amp * np.exp(-((x - center) ** 2) / 0.0026)
    y += rng.normal(0.0, 0.007, x.shape[0])
    y = (y - y.min()) / (y.max() - y.min())
    curve = VMobject(stroke_color=color, stroke_width=2.8)
    curve.set_points_smoothly([axes.c2p(a, b) for a, b in zip(x, y)])
    return VGroup(axes, curve)


def scalar_vector(values=("2.75", "0.75", "4.25"), flags=(1, 0, 1), compact=False):
    names = ("l", "h", "r")
    cards = VGroup()
    for name, value, flag in zip(names, values, flags):
        box = panel(0.78 if compact else 0.98, 0.48, YELLOW, "#262433", 1.4)
        value_label = txt(value, 14 if compact else 16, YELLOW, BOLD)
        marker = txt("known" if flag else "unknown", 8 if compact else 9, GREEN if flag else RED)
        name_label = txt(name, 12 if compact else 13, MUTED)
        trio = VGroup(name_label, value_label, marker).arrange(RIGHT, buff=0.05)
        trio.move_to(box)
        cards.add(VGroup(box, trio))
    cards.arrange(DOWN if not compact else RIGHT, buff=0.10)
    return cards


def scalar_six_vector():
    values = [("l", "2.75", "1"), ("h", "0", "0"), ("r", "4.25", "1")]
    cells = VGroup()
    for name, value, flag in values:
        name_tag = txt(name, 12, MUTED)
        val = label_panel(value, "value", YELLOW, 0.60, 0.42)
        known = label_panel(flag, "flag", GREEN if flag == "1" else RED, 0.56, 0.42)
        row = VGroup(name_tag, val, known).arrange(RIGHT, buff=0.08)
        cells.add(row)
    cells.arrange(DOWN, aligned_edge=LEFT, buff=0.10)
    outer = panel(cells.width + 0.30, cells.height + 0.28, YELLOW, PANEL_2, 1.5)
    cells.move_to(outer)
    return VGroup(outer, cells)


# ---------------------------------------------------------------------------
# Neural-network visual primitives
# ---------------------------------------------------------------------------


def neuron_layer(count, color, height=1.0, radius=0.055):
    nodes = VGroup(*[
        Circle(radius=radius, stroke_color=color, stroke_width=1.2,
               fill_color=color, fill_opacity=0.75)
        for _ in range(count)
    ])
    nodes.arrange(DOWN, buff=max(0.04, (height - 2 * radius * count) / max(1, count - 1)))
    return nodes


def dense_network(width=2.0, height=1.25, layers=(5, 7, 5), color=YELLOW, title_text=None):
    node_layers = VGroup(*[neuron_layer(n, color, height=height) for n in layers])
    node_layers.arrange(RIGHT, buff=width / max(1, len(layers) - 1) - 0.11)
    lines = VGroup()
    for left, right in zip(node_layers[:-1], node_layers[1:]):
        for a in left:
            for b in right:
                lines.add(Line(a.get_right(), b.get_left(), stroke_color=color, stroke_width=0.45, stroke_opacity=0.36))
    group = VGroup(lines, node_layers)
    if title_text:
        title_label = txt(title_text, 13, color, BOLD).next_to(group, UP, buff=0.12)
        group.add(title_label)
    return group


def mini_transformer(depth, color, label, width=1.55, height=1.3):
    """A block stack: each layer visibly contains attention and MLP sublayers."""
    layers = VGroup()
    for index in range(depth):
        shell = RoundedRectangle(
            width=width, height=height / depth - 0.035, corner_radius=0.05,
            stroke_color=color, stroke_width=1.2, fill_color=PANEL_2, fill_opacity=0.95,
        )
        words = txt("attention  →  MLP", 9, MUTED).move_to(shell)
        layers.add(VGroup(shell, words))
    layers.arrange(DOWN, buff=0.035)
    tag = txt(label, 12, color, BOLD).next_to(layers, UP, buff=0.10)
    return VGroup(layers, tag)


def convolution_stack(color=BLUE, label="Conv patch-4", width=1.5, height=1.2):
    filters = VGroup()
    for i in range(4):
        slab = RoundedRectangle(
            width=width, height=0.18, corner_radius=0.035,
            stroke_color=color, stroke_width=1.0, fill_color=color, fill_opacity=0.32 + i * 0.12,
        )
        filters.add(slab)
    filters.arrange(DOWN, buff=0.06)
    tag = txt(label, 12, color, BOLD).next_to(filters, UP, buff=0.10)
    return VGroup(filters, tag)


def film_cards(n=6, small=False):
    cards = VGroup()
    for i in range(n):
        card = panel(0.56 if small else 0.70, 0.38, YELLOW, "#282432", 1.0)
        label = txt(f"γ{i + 1}, β{i + 1}", 8 if small else 9, YELLOW, BOLD).move_to(card)
        cards.add(VGroup(card, label))
    cards.arrange(RIGHT, buff=0.05)
    return cards


def query_tokens(count=8):
    group = VGroup()
    for i in range(count):
        missing = i in (2, 5, 6)
        cell = RoundedRectangle(
            width=0.20, height=0.20, corner_radius=0.035,
            stroke_color=(RED if missing else GREEN), stroke_width=1.0,
            fill_color=("#2A1C31" if missing else GREEN), fill_opacity=0.72,
        )
        mark = txt("?" if missing else "•", 13, RED if missing else PANEL).move_to(cell)
        group.add(VGroup(cell, mark))
    scalar = RoundedRectangle(width=0.22, height=0.22, corner_radius=0.04, stroke_color=YELLOW,
                              stroke_width=1.2, fill_color=YELLOW, fill_opacity=0.75)
    group.add(VGroup(scalar, txt("s", 11, PANEL, BOLD).move_to(scalar)))
    group.arrange(RIGHT, buff=0.05)
    return group


def gclct_stack():
    """Eight exact predictor blocks, each drawn as LN/FiLM → self → cross → MLP."""
    rows = VGroup()
    for i in range(8):
        row = panel(2.65, 0.28, GREEN, PANEL_2, 1.0)
        labels = VGroup(
            txt("LN + FiLM", 8, YELLOW), txt("self-attn", 8, GREEN),
            txt("cross-attn", 8, CYAN), txt("MLP", 8, PURPLE),
        ).arrange(RIGHT, buff=0.13).move_to(row)
        index = txt(str(i + 1), 8, MUTED).next_to(row, LEFT, buff=0.05)
        rows.add(VGroup(row, labels, index))
    rows.arrange(DOWN, buff=0.025)
    label = txt("GCLCT predictor · 8 blocks", 14, GREEN, BOLD).next_to(rows, UP, buff=0.10)
    return VGroup(rows, label)


def loss_chip(name, detail, color, width=1.45):
    box = panel(width, 0.58, color, PANEL, 1.5)
    top = txt(name, 13, color, BOLD)
    sub = txt(detail, 8, MUTED)
    words = VGroup(top, sub).arrange(DOWN, buff=0.03).move_to(box)
    return VGroup(box, words)


# ---------------------------------------------------------------------------
# Reusable implementation-accurate module figures
# ---------------------------------------------------------------------------


def occupancy_encoder_figure():
    image = occupancy_map(side=0.10, hidden=True).scale(0.72)
    image_label = txt("M  [1, 64, 64]", 12, BLUE, BOLD).next_to(image, DOWN, buff=0.10)
    patch = convolution_stack(BLUE, "Conv2d 1→192 · k=4 s=4", 1.25, 0.92)
    patch_grid = token_grid(4, 4, BLUE, 0.13, missing=True)
    grid_label = txt("16 × 16 = 256 tokens", 11, BLUE).next_to(patch_grid, DOWN, buff=0.10)
    stack = mini_transformer(6, BLUE, "6 TransformerBlocks", 1.55, 1.38)
    film = film_cards(6, small=True)
    film_label = txt("scalar FiLM after every block", 10, YELLOW).next_to(film, DOWN, buff=0.08)
    items = VGroup(VGroup(image, image_label), patch, VGroup(patch_grid, grid_label), stack)
    items.arrange(RIGHT, buff=0.32)
    arrow1 = arrow_between(image.get_right(), patch.get_left(), BLUE)
    arrow2 = arrow_between(patch.get_right(), patch_grid.get_left(), BLUE)
    arrow3 = arrow_between(patch_grid.get_right(), stack.get_left(), BLUE)
    film.next_to(stack, DOWN, buff=0.25)
    film_label.next_to(film, DOWN, buff=0.06)
    full = VGroup(items, arrow1, arrow2, arrow3, film, film_label)
    return full, {"image": image, "patch": patch, "grid": patch_grid, "stack": stack, "film": film}


def scalar_encoder_figure():
    inputs = scalar_six_vector().scale(0.88)
    network = dense_network(width=1.60, height=1.35, layers=(6, 7, 6), color=YELLOW, title_text="6 → 128 → GELU → 128")
    film = film_cards(6)
    summary = RoundedRectangle(width=1.15, height=0.42, corner_radius=0.07, stroke_color=PURPLE,
                               stroke_width=1.5, fill_color=PURPLE, fill_opacity=0.25)
    summary_label = txt("scalar summary\n[1, 192]", 10, PURPLE, BOLD).move_to(summary)
    outputs = VGroup(
        VGroup(film, txt("six (γ, β) pairs", 10, YELLOW).next_to(film, DOWN, buff=0.06)),
        VGroup(summary, summary_label),
    ).arrange(DOWN, buff=0.30, aligned_edge=LEFT)
    line = arrow_between(inputs.get_right(), network.get_left(), YELLOW)
    split1 = arrow_between(network.get_right(), film.get_left(), YELLOW)
    split2 = arrow_between(network.get_right(), summary.get_left(), PURPLE)
    group = VGroup(inputs, network, outputs, line, split1, split2)
    inputs.move_to(LEFT * 2.7)
    network.move_to(ORIGIN)
    outputs.move_to(RIGHT * 2.75)
    line.put_start_and_end_on(inputs.get_right(), network.get_left())
    split1.put_start_and_end_on(network.get_right(), film.get_left())
    split2.put_start_and_end_on(network.get_right(), summary.get_left())
    return group, {"inputs": inputs, "network": network, "film": film, "summary": summary}


def spectrum_path_figure():
    s = spectrum(2.0, 0.72, CYAN, 12)
    s_label = txt("S [2, 301]", 12, CYAN, BOLD).next_to(s, DOWN, buff=0.10)
    feature = convolution_stack(CYAN, "released encoder · frozen", 1.35, 0.92)
    local = token_grid(3, 10, CYAN, 0.10)
    local_label = txt("A_local [301, 256]", 10, CYAN).next_to(local, DOWN, buff=0.10)
    pool = label_panel("mean pool", "301 → 1", CYAN, 0.82, 0.50)
    cphys = label_panel("c_physics", "[384]", PURPLE, 0.88, 0.50)
    queries = dense_network(width=0.95, height=0.85, layers=(4, 5, 4), color=CYAN, title_text="16 learned queries")
    goal = token_grid(2, 8, PURPLE, 0.10)
    goal_label = txt("a_goal [16, 384]", 10, PURPLE).next_to(goal, DOWN, buff=0.08)
    left = VGroup(s, s_label)
    local_group = VGroup(local, local_label)
    top = VGroup(pool, cphys).arrange(RIGHT, buff=0.18)
    lower = VGroup(queries, VGroup(goal, goal_label)).arrange(RIGHT, buff=0.20)
    result = VGroup(top, lower).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
    groups = VGroup(left, feature, local_group, result).arrange(RIGHT, buff=0.35)
    arrows = VGroup(
        arrow_between(s.get_right(), feature.get_left(), CYAN),
        arrow_between(feature.get_right(), local.get_left(), CYAN),
        arrow_between(local.get_right(), pool.get_left(), CYAN),
        arrow_between(pool.get_right(), cphys.get_left(), PURPLE),
        arrow_between(local.get_right(), queries.get_left(), CYAN),
        arrow_between(queries.get_right(), goal.get_left(), PURPLE),
    )
    return VGroup(groups, arrows), {"spectrum": s, "feature": feature, "local": local, "cphys": cphys, "goal": goal}


# ---------------------------------------------------------------------------
# 1. Input contract and masking
# ---------------------------------------------------------------------------


class DataAndMaskScene(Scene):
    def construct(self):
        h = heading("A training sample is factorized before it is masked", "Step 1 of the real loop")
        self.play(Write(h))

        occ = occupancy_map(side=0.15, hidden=False).shift(LEFT * 4.6 + UP * 0.15)
        occ_label = txt("occupancy M", 20, BLUE, BOLD).next_to(occ, DOWN, buff=0.18)
        occ_shape = txt("binary [1, 64, 64]", 13, MUTED).next_to(occ_label, DOWN, buff=0.07)
        scalars = scalar_vector(("2.75", "0.75", "4.25"), (1, 1, 1)).shift(LEFT * 0.35 + UP * 0.10)
        scalar_label = txt("three global physical scalars", 18, YELLOW, BOLD).next_to(scalars, DOWN, buff=0.18)
        spectrum_view = spectrum(3.1, 1.2, CYAN, 8).shift(RIGHT * 4.25 + UP * 0.15)
        spectrum_label = txt("target spectrum S", 20, CYAN, BOLD).next_to(spectrum_view, DOWN, buff=0.18)
        spectrum_shape = txt("real + imaginary response [2, 301]", 13, MUTED).next_to(spectrum_label, DOWN, buff=0.07)
        self.play(FadeIn(occ), Write(occ_label), FadeIn(occ_shape))
        self.play(FadeIn(scalars), Write(scalar_label))
        self.play(Create(spectrum_view), Write(spectrum_label), FadeIn(spectrum_shape))

        statement = txt("Spatial uncertainty and parameter uncertainty are different questions.", 22, INK, BOLD)
        statement.to_edge(DOWN, buff=0.33)
        self.play(Write(statement))
        self.wait(0.6)

        mask_grid = token_grid(4, 4, BLUE, 0.15, missing=True).move_to(occ.get_center())
        mask_frame = SurroundingRectangle(mask_grid, color=RED, buff=0.13, stroke_width=1.8)
        mask_title = txt("BlockMasker M [16, 16]", 15, RED, BOLD).next_to(mask_frame, UP, buff=0.15)
        masked_scalars = scalar_vector(("2.75", "0", "4.25"), (1, 0, 1)).move_to(scalars)
        flags = txt("unknown: value → 0, flag → 0", 13, RED).next_to(masked_scalars, UP, buff=0.22)
        self.play(FadeOut(occ), FadeIn(mask_grid), Create(mask_frame), Write(mask_title))
        self.play(Transform(scalars, masked_scalars), Write(flags))
        self.play(Indicate(mask_grid, color=RED), Indicate(scalars[1], color=RED))
        self.wait(0.8)


# ---------------------------------------------------------------------------
# 2. Student inputs: actual neural-network components
# ---------------------------------------------------------------------------


class EncodersScene(Scene):
    def construct(self):
        h = heading("Three inputs enter three different neural computations", "Student forward · steps 2–4")
        self.play(Write(h))

        occ_fig, occ_nodes = occupancy_encoder_figure()
        occ_fig.scale(0.78).move_to(UP * 1.15)
        occ_title = txt("1  Occupancy encoder", 19, BLUE, BOLD).next_to(occ_fig, UP, buff=0.15)
        self.play(FadeIn(occ_title), FadeIn(occ_fig, shift=RIGHT * 0.15))
        travel(self, occ_nodes["image"].get_right(), occ_nodes["patch"].get_left(), BLUE)
        travel(self, occ_nodes["patch"].get_right(), occ_nodes["grid"].get_left(), BLUE)
        self.play(Indicate(occ_nodes["film"], color=YELLOW), ApplyWave(occ_nodes["stack"][0], direction=DOWN), run_time=0.9)
        occ_note = txt("Masked patch embeddings are replaced by one learned mask token, then receive position embeddings.", 14, MUTED)
        occ_note.next_to(occ_fig, DOWN, buff=0.18)
        self.play(Write(occ_note))
        self.wait(0.5)
        self.play(FadeOut(occ_fig), FadeOut(occ_title), FadeOut(occ_note), run_time=0.35)

        scalar_fig, scalar_nodes = scalar_encoder_figure()
        scalar_fig.scale(0.86).move_to(UP * 0.35)
        scalar_title = txt("2  Scalar encoder", 19, YELLOW, BOLD).next_to(scalar_fig, UP, buff=0.22)
        self.play(FadeIn(scalar_title), FadeIn(scalar_fig, shift=UP * 0.15))
        travel(self, scalar_nodes["inputs"].get_right(), scalar_nodes["network"].get_left(), YELLOW)
        travel(self, scalar_nodes["network"].get_right(), scalar_nodes["film"].get_left(), YELLOW)
        pulse(self, scalar_nodes["summary"].get_center(), PURPLE)
        scalar_note = txt("The live MLP emits six identity-initialized FiLM pairs and one 192-D scalar-summary token.", 14, MUTED)
        scalar_note.next_to(scalar_fig, DOWN, buff=0.25)
        self.play(Write(scalar_note))
        self.wait(0.5)
        self.play(FadeOut(scalar_fig), FadeOut(scalar_title), FadeOut(scalar_note), run_time=0.35)

        spec_fig, spec_nodes = spectrum_path_figure()
        spec_fig.scale(0.76).move_to(UP * 0.35)
        spec_title = txt("3  Frozen SpectrumPath", 19, CYAN, BOLD).next_to(spec_fig, UP, buff=0.22)
        self.play(FadeIn(spec_title), FadeIn(spec_fig, shift=UP * 0.15))
        travel(self, spec_nodes["spectrum"].get_right(), spec_nodes["feature"].get_left(), CYAN)
        travel(self, spec_nodes["feature"].get_right(), spec_nodes["local"].get_left(), CYAN)
        pulse(self, spec_nodes["cphys"].get_center(), PURPLE)
        pulse(self, spec_nodes["goal"].get_center(), PURPLE)
        spec_note = txt("Released spectrum encoder is frozen: it yields c_physics [384] and 16 structured goal tokens [16, 384].", 14, MUTED)
        spec_note.next_to(spec_fig, DOWN, buff=0.22)
        self.play(Write(spec_note))
        self.wait(0.9)


# ---------------------------------------------------------------------------
# 3. Fusion and the exact GCLCT completion mechanism
# ---------------------------------------------------------------------------


class FusionPredictorScene(Scene):
    def construct(self):
        h = heading("The predictor fills latent holes, not pixels", "Student forward · steps 5–7")
        self.play(Write(h))

        occ_tokens = token_grid(4, 8, BLUE, 0.12, missing=True)
        goal_tokens = token_grid(2, 8, CYAN, 0.12)
        scalar_token = RoundedRectangle(width=0.35, height=0.35, corner_radius=0.06, stroke_color=YELLOW,
                                        stroke_width=1.5, fill_color=YELLOW, fill_opacity=0.7)
        scalar_token.add(txt("s", 15, PANEL, BOLD).move_to(scalar_token))
        left = VGroup(
            VGroup(occ_tokens, txt("256 occupancy tokens · 192-D", 12, BLUE).next_to(occ_tokens, DOWN, buff=0.10)),
            VGroup(goal_tokens, txt("16 goal tokens · 384-D", 12, CYAN).next_to(goal_tokens, DOWN, buff=0.10)),
            VGroup(scalar_token, txt("1 summary", 11, YELLOW).next_to(scalar_token, DOWN, buff=0.08)),
        ).arrange(DOWN, buff=0.34, aligned_edge=LEFT).move_to(LEFT * 4.8 + DOWN * 0.3)
        self.play(FadeIn(left, shift=RIGHT * 0.12))

        projection = label_panel("goal projection", "384 → 192", PURPLE, 1.35, 0.65).move_to(LEFT * 2.65 + DOWN * 0.15)
        fusion = mini_transformer(2, PURPLE, "Fusion transformer · 2 layers", 2.20, 1.20).move_to(ORIGIN + UP * 0.55)
        fused = token_grid(3, 12, PURPLE, 0.095).next_to(fusion, DOWN, buff=0.25)
        fused_label = txt("273 context tokens = 256 + 16 + 1", 12, PURPLE, BOLD).next_to(fused, DOWN, buff=0.08)
        arrows = VGroup(
            arrow_between(occ_tokens.get_right(), fusion.get_left(), BLUE),
            arrow_between(goal_tokens.get_right(), projection.get_left(), CYAN),
            arrow_between(projection.get_right(), fusion.get_left(), PURPLE),
            arrow_between(scalar_token.get_right(), fusion.get_left(), YELLOW),
        )
        self.play(FadeIn(projection), GrowArrow(arrows[1]), GrowArrow(arrows[0]), GrowArrow(arrows[2]), GrowArrow(arrows[3]))
        travel(self, goal_tokens.get_right(), projection.get_left(), CYAN)
        self.play(FadeIn(fusion), FadeIn(fused), Write(fused_label), Indicate(fusion, color=PURPLE))
        self.wait(0.5)

        queries = query_tokens(8).move_to(RIGHT * 2.7 + UP * 1.55)
        query_label = txt("257 queries: 256 positions + 1 scalar query", 12, GREEN, BOLD).next_to(queries, UP, buff=0.12)
        key_values = token_grid(3, 12, PURPLE, 0.075).move_to(RIGHT * 2.7 + DOWN * 1.75)
        kv_label = txt("key/value = all 273 fused tokens", 11, PURPLE).next_to(key_values, DOWN, buff=0.10)
        cphys = label_panel("c_physics", "384 → 192 FiLM", CYAN, 1.55, 0.60).move_to(RIGHT * 5.25 + UP * 0.75)
        predictor = gclct_stack().scale(0.88).move_to(RIGHT * 2.7 + DOWN * 0.12)
        self.play(FadeIn(queries), Write(query_label), FadeIn(key_values), Write(kv_label), FadeIn(cphys), FadeIn(predictor))
        self.play(GrowArrow(arrow_between(fused.get_right(), queries.get_left(), GREEN)))
        travel(self, fused.get_right(), queries.get_left(), GREEN)
        conditioning = arrow_between(cphys.get_left(), predictor.get_right(), CYAN)
        self.play(GrowArrow(conditioning), Indicate(cphys, color=CYAN), ApplyWave(predictor[0], direction=DOWN), run_time=0.9)

        output = VGroup(token_grid(2, 8, GREEN, 0.10), RoundedRectangle(width=0.34, height=0.34, corner_radius=0.06,
                      stroke_color=YELLOW, stroke_width=1.4, fill_color=YELLOW, fill_opacity=0.75))
        output.arrange(RIGHT, buff=0.18).next_to(predictor, RIGHT, buff=0.24)
        output_label = txt("z_hat\n256 occupancy + 1 scalar", 10, GREEN).next_to(output, DOWN, buff=0.08)
        self.play(FadeIn(output, shift=RIGHT * 0.12), Write(output_label))
        final = txt("Every masked location must infer its latent from visible geometry, the scalar context, and the spectrum goal.", 17, INK, BOLD)
        final.to_edge(DOWN, buff=0.28)
        self.play(Write(final))
        self.wait(1.0)


# ---------------------------------------------------------------------------
# 4. EMA teacher, decoders, MetaDiT assembly, and all loss routes
# ---------------------------------------------------------------------------


class TargetDecodeLossScene(Scene):
    def construct(self):
        h = heading("One forward step creates several honest learning signals", "Target, decoding, and physics")
        self.play(Write(h))

        # Teacher path (top) is separate and frozen from gradient.
        full_occ = occupancy_map(side=0.095, seed=15, hidden=False).move_to(LEFT * 5.45 + UP * 1.55)
        true_scalars = scalar_vector(("2.75", "0.75", "4.25"), (1, 1, 1), compact=True).next_to(full_occ, RIGHT, buff=0.30)
        teacher_scalar = dense_network(1.10, 0.75, (4, 5, 4), PURPLE, "scalar_mlp_ema")
        teacher_scalar.scale(0.65).move_to(LEFT * 1.95 + UP * 1.55)
        teacher_encoder = mini_transformer(6, PURPLE, "EMA occupancy encoder · 6 blocks", 1.85, 1.40).move_to(LEFT * 0.05 + UP * 1.55)
        cphysics = token_grid(1, 6, CYAN, 0.075).move_to(LEFT * 1.80 + UP * 0.50)
        cphysics_label = txt("c_physics from frozen SpectrumPath", 9, CYAN).next_to(cphysics, DOWN, buff=0.06)
        spectrum_film = label_panel("frozen SpectrumFiLM", "γ, β for target", PINK, 1.35, 0.50).move_to(LEFT * 0.05 + UP * 0.50)
        scalar_target = token_grid(1, 5, PURPLE, 0.075).move_to(LEFT * 1.95 + UP * 0.72)
        scalar_target_label = txt("z_y_scal", 9, PURPLE).next_to(scalar_target, UP, buff=0.05)
        raw_target = token_grid(2, 8, PURPLE, 0.09).move_to(RIGHT * 2.05 + UP * 1.95)
        spec_target = token_grid(2, 8, PINK, 0.09).move_to(RIGHT * 2.05 + UP * 0.93)
        teacher_title = txt("STOP-GRAD EMA TARGET PATH", 14, PURPLE, BOLD).next_to(full_occ, UP, buff=0.16)
        self.play(
            FadeIn(full_occ), FadeIn(true_scalars), FadeIn(teacher_scalar),
            FadeIn(teacher_encoder), FadeIn(cphysics), Write(cphysics_label),
            FadeIn(spectrum_film), FadeIn(scalar_target), Write(scalar_target_label),
            Write(teacher_title),
        )
        self.play(GrowArrow(arrow_between(full_occ.get_right(), teacher_encoder.get_left(), PURPLE)),
                  GrowArrow(arrow_between(true_scalars.get_right(), teacher_scalar.get_left(), PURPLE)))
        self.play(
            GrowArrow(arrow_between(teacher_scalar.get_right(), teacher_encoder.get_left(), PURPLE)),
            GrowArrow(arrow_between(cphysics.get_right(), spectrum_film.get_left(), PINK)),
            GrowArrow(arrow_between(spectrum_film.get_top(), teacher_encoder.get_bottom(), PINK)),
        )
        self.play(FadeIn(raw_target), FadeIn(spec_target))
        raw_label = txt("z_y_raw\nspectrum-free", 10, PURPLE).next_to(raw_target, RIGHT, buff=0.10)
        spec_label = txt("z_y_occ_spec\nfrozen SpectrumFiLM", 10, PINK).next_to(spec_target, RIGHT, buff=0.10)
        self.play(Write(raw_label), Write(spec_label))

        # Student outputs then enter two decoders.
        predicted = token_grid(2, 8, GREEN, 0.10).move_to(LEFT * 5.2 + DOWN * 1.15)
        scalar_q = RoundedRectangle(width=0.36, height=0.36, corner_radius=0.06, stroke_color=YELLOW,
                                    stroke_width=1.5, fill_color=YELLOW, fill_opacity=0.7).next_to(predicted, RIGHT, buff=0.18)
        scalar_q.add(txt("s", 15, PANEL, BOLD).move_to(scalar_q))
        pred_label = txt("z_hat", 14, GREEN, BOLD).next_to(predicted, UP, buff=0.10)
        occ_decoder = VGroup(
            token_grid(4, 4, GREEN, 0.085), convolution_stack(BLUE, "reshape 16×16 → upsample", 1.10, 0.80),
            occupancy_map(side=0.07, seed=29, hidden=False),
        ).arrange(RIGHT, buff=0.20).move_to(LEFT * 1.65 + DOWN * 1.10)
        scalar_heads = dense_network(1.25, 0.80, (3, 5, 3), YELLOW, "three scalar heads")
        scalar_heads.scale(0.85).move_to(RIGHT * 3.20 + DOWN * 0.90)
        scalars_hat = scalar_vector(("l̂", "ĥ", "r̂"), (0, 0, 0), compact=True).move_to(RIGHT * 4.75 + DOWN * 0.90)
        self.play(FadeIn(predicted), FadeIn(scalar_q), Write(pred_label), FadeIn(occ_decoder), FadeIn(scalar_heads), FadeIn(scalars_hat))
        self.play(GrowArrow(arrow_between(predicted.get_right(), occ_decoder.get_left(), GREEN)),
                  GrowArrow(arrow_between(scalar_q.get_right(), scalar_heads.get_left(), YELLOW)),
                  GrowArrow(arrow_between(scalar_heads.get_right(), scalars_hat.get_left(), YELLOW)))

        # Geometry is only broadcast here, at the frozen surrogate boundary.
        assembly = label_panel("assemble geometry", "[3, 64, 64] only here", ORANGE, 1.65, 0.70).move_to(RIGHT * 1.55 + DOWN * 2.45)
        surrogate = convolution_stack(CYAN, "frozen MetaDiT EM surrogate", 1.70, 1.00).move_to(RIGHT * 4.15 + DOWN * 2.45)
        s_hat = spectrum(1.85, 0.58, GREEN, 22).next_to(surrogate, RIGHT, buff=0.25)
        self.play(FadeIn(assembly), FadeIn(surrogate), Create(s_hat))
        self.play(GrowArrow(arrow_between(occ_decoder.get_bottom(), assembly.get_top(), BLUE)),
                  GrowArrow(arrow_between(scalars_hat.get_bottom(), assembly.get_top(), YELLOW)),
                  GrowArrow(arrow_between(assembly.get_right(), surrogate.get_left(), ORANGE)),
                  GrowArrow(arrow_between(surrogate.get_right(), s_hat.get_left(), CYAN)))

        # Objective chips make every supervised comparison visible.
        chips = VGroup(
            loss_chip("L_inv + L_var + L_cov", "masked z_hat ↔ z_y_raw", RED, 1.75),
            loss_chip("L_cond", "spectrum-conditioned target", PINK, 1.35),
            loss_chip("L_scal_t", "scalar summary ↔ EMA summary", PURPLE, 1.48),
            loss_chip("L_scalar", "unknown l, h, r only", YELLOW, 1.30),
            loss_chip("L_occ", "masked-pixel BCE", BLUE, 1.25),
            loss_chip("L_phys", "SmoothL1(S_hat, S)", CYAN, 1.35),
            loss_chip("L_summary", "known scalar readout", ORANGE, 1.45),
        ).arrange(RIGHT, buff=0.08).scale(0.72)
        chips.to_edge(DOWN, buff=0.16)
        self.play(FadeIn(chips, shift=UP * 0.12))
        self.play(Indicate(chips[0], color=RED), Indicate(chips[5], color=CYAN), run_time=0.8)
        note = txt("Physics is differentiable through the input geometry—but the released surrogate parameters never update.", 14, MUTED)
        note.next_to(chips, UP, buff=0.10)
        self.play(Write(note))
        self.wait(1.0)


# ---------------------------------------------------------------------------
# 5. The optimizer step: exact gradient boundaries and EMA update
# ---------------------------------------------------------------------------


class UpdateScene(Scene):
    def construct(self):
        h = heading("Only the student learns from this sample", "Steps 9–10: backward and EMA")
        self.play(Write(h))

        objective = label_panel("weighted total loss", "all active terms summed", RED, 2.15, 0.82).move_to(LEFT * 5.0 + UP * 0.5)
        student_blocks = VGroup(
            label_panel("occupancy encoder", "patch + 6 blocks", BLUE, 1.75, 0.70),
            label_panel("scalar MLP", "live FiLM + summary", YELLOW, 1.75, 0.70),
            label_panel("fusion", "2 attention blocks", PURPLE, 1.55, 0.70),
            label_panel("GCLCT", "8 completion blocks", GREEN, 1.55, 0.70),
            label_panel("decoders", "occupancy + scalars", ORANGE, 1.55, 0.70),
        ).arrange(DOWN, buff=0.13).move_to(LEFT * 1.55 + UP * 0.10)
        optimizer = label_panel("AdamW", "gradient clip → parameter step", GREEN, 2.05, 0.80).move_to(RIGHT * 2.15 + UP * 0.45)
        ema_occ = label_panel("EMA occupancy encoder", "τ old + (1−τ) student", PURPLE, 2.30, 0.80).move_to(RIGHT * 5.10 + UP * 1.20)
        ema_scalar = label_panel("EMA scalar MLP", "FiLM for target path only", PURPLE, 2.30, 0.80).move_to(RIGHT * 5.10 + DOWN * 0.25)
        frozen_spectrum = label_panel("released spectrum encoder", "frozen", CYAN, 2.10, 0.70).move_to(RIGHT * 4.95 + DOWN * 1.75)
        frozen_surrogate = label_panel("EM surrogate", "frozen", CYAN, 2.10, 0.70).move_to(RIGHT * 4.95 + DOWN * 2.70)
        self.play(FadeIn(objective), FadeIn(student_blocks), FadeIn(optimizer), FadeIn(ema_occ), FadeIn(ema_scalar), FadeIn(frozen_spectrum), FadeIn(frozen_surrogate))

        lines = VGroup(*[arrow_between(objective.get_right(), block.get_left(), RED, 2.5) for block in student_blocks])
        self.play(*[GrowArrow(line) for line in lines], run_time=0.9)
        for block in student_blocks:
            travel(self, objective.get_right(), block.get_left(), RED, 0.30)
        self.play(GrowArrow(arrow_between(student_blocks.get_right(), optimizer.get_left(), GREEN)), Indicate(optimizer, color=GREEN))

        # Show weights as connected layers; they actually change only on this side.
        weights = dense_network(1.55, 1.10, (4, 6, 4), GREEN, "trainable weights").move_to(RIGHT * 1.85 + DOWN * 1.85)
        self.play(FadeIn(weights))
        self.play(ApplyWave(weights[1], direction=RIGHT), Flash(weights.get_center(), color=GREEN), run_time=0.8)
        step_note = txt("optimizer.step(): parameters move", 16, GREEN, BOLD).next_to(weights, DOWN, buff=0.12)
        self.play(Write(step_note))

        ema_lines = VGroup(
            arrow_between(optimizer.get_right(), ema_occ.get_left(), PURPLE, 2.2),
            arrow_between(optimizer.get_right(), ema_scalar.get_left(), PURPLE, 2.2),
        )
        self.play(*[GrowArrow(line) for line in ema_lines])
        travel(self, optimizer.get_right(), ema_occ.get_left(), PURPLE, 0.50)
        travel(self, optimizer.get_right(), ema_scalar.get_left(), PURPLE, 0.50)
        tau = formula("EMA ← 0.996…0.999 · old + (1 − τ) · student", 18, PURPLE)
        tau.next_to(ema_scalar, DOWN, buff=0.15)
        self.play(Write(tau))

        locks = VGroup(
            txt("no gradients", 12, CYAN, BOLD).next_to(frozen_spectrum, RIGHT, buff=0.16),
            txt("no gradients", 12, CYAN, BOLD).next_to(frozen_surrogate, RIGHT, buff=0.16),
            txt("no gradients", 12, PURPLE, BOLD).next_to(ema_occ, RIGHT, buff=0.12),
            txt("no gradients", 12, PURPLE, BOLD).next_to(ema_scalar, RIGHT, buff=0.12),
        )
        self.play(FadeIn(locks), Indicate(frozen_spectrum, color=CYAN), Indicate(frozen_surrogate, color=CYAN))
        final = txt("The teacher lags the student; the physics references remain fixed; only the student is optimized.", 19, INK, BOLD)
        final.to_edge(DOWN, buff=0.28)
        self.play(Write(final))
        self.wait(1.0)


# ---------------------------------------------------------------------------
# 6. Deployment: same machinery, target/loss/optimizer removed
# ---------------------------------------------------------------------------


class InferenceScene(Scene):
    def construct(self):
        h = heading("Inference: condition on a spectrum, complete a design", "Pure inverse design")
        self.play(Write(h))

        goal = spectrum(3.1, 1.1, CYAN, 31).move_to(LEFT * 5.0 + UP * 1.15)
        unknown_occ = occupancy_map(side=0.12, hidden=True).move_to(LEFT * 5.0 + DOWN * 1.40)
        unknown_s = scalar_vector(("0", "0", "0"), (0, 0, 0), compact=True).next_to(unknown_occ, DOWN, buff=0.13)
        self.play(Create(goal), FadeIn(unknown_occ), FadeIn(unknown_s))
        self.play(Write(txt("target S", 18, CYAN, BOLD).next_to(goal, DOWN, buff=0.12)),
                  Write(txt("all design variables unknown", 14, RED).next_to(unknown_s, DOWN, buff=0.12)))

        spec_net = convolution_stack(CYAN, "frozen SpectrumPath", 1.35, 0.95).move_to(LEFT * 2.2 + UP * 1.15)
        core = VGroup(mini_transformer(2, PURPLE, "fusion", 1.55, 0.82), gclct_stack().scale(0.56)).arrange(RIGHT, buff=0.35).move_to(ORIGIN + UP * 0.75)
        dec = VGroup(convolution_stack(BLUE, "occupancy decoder", 1.20, 0.82), dense_network(0.85, 0.68, (3, 4, 3), YELLOW, "scalar heads")).arrange(DOWN, buff=0.35).move_to(RIGHT * 3.85 + UP * 0.55)
        design = occupancy_map(side=0.11, seed=42).move_to(RIGHT * 5.55 + UP * 1.30)
        design_s = scalar_vector(("2.71", "0.77", "4.18"), (0, 0, 0), compact=True).scale(0.78).next_to(design, DOWN, buff=0.14)
        verify = spectrum(2.0, 0.62, GREEN, 30).move_to(RIGHT * 5.45 + DOWN * 1.92)
        self.play(FadeIn(spec_net), FadeIn(core), FadeIn(dec), FadeIn(design), FadeIn(design_s), Create(verify))
        edges = VGroup(
            arrow_between(goal.get_right(), spec_net.get_left(), CYAN),
            arrow_between(spec_net.get_right(), core.get_left(), PURPLE),
            arrow_between(core.get_right(), dec.get_left(), GREEN),
            arrow_between(dec.get_right(), design.get_left(), GREEN),
            arrow_between(design.get_bottom(), verify.get_top(), CYAN),
        )
        self.play(*[GrowArrow(edge) for edge in edges])
        for start, end, color in [
            (goal.get_right(), spec_net.get_left(), CYAN),
            (spec_net.get_right(), core.get_left(), PURPLE),
            (core.get_right(), dec.get_left(), GREEN),
            (dec.get_right(), design.get_left(), GREEN),
        ]:
            travel(self, start, end, color, 0.42)
        conclusion = txt("Forward once → decode a complete candidate → evaluate its predicted spectrum. No backpropagation here.", 18, GREEN, BOLD)
        conclusion.to_edge(DOWN, buff=0.26)
        self.play(Write(conclusion))
        self.wait(1.2)


# ---------------------------------------------------------------------------
# Master scene
# ---------------------------------------------------------------------------


class JEPAFullPresentation(Scene):
    """The complete, implementation-faithful training-to-inference story."""

    def construct(self):
        scenes = [
            DataAndMaskScene,
            EncodersScene,
            FusionPredictorScene,
            TargetDecodeLossScene,
            UpdateScene,
            InferenceScene,
        ]
        for index, scene_type in enumerate(scenes):
            scene_type.construct(self)
            if index < len(scenes) - 1:
                fade_scene(self)
