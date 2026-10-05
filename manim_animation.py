"""A story-driven visual tour of the unified 192-D metasurface JEPA.

Preview quickly:
    python -m manim -pql manim_animation.py JEPAFullPresentation

Render the final film:
    python -m manim -pqh manim_animation.py JEPAFullPresentation
"""

from manim import *
import numpy as np


# ---------------------------------------------------------------------------
# Visual language
# ---------------------------------------------------------------------------

BG = "#0B1020"
INK = "#F5F7FF"
MUTED = "#AAB4D4"
GRID = "#25304E"
BLUE = "#58A6FF"       # occupancy / geometry
CYAN = "#55DFFC"       # spectrum / physical goal
GREEN = "#65E59B"      # prediction / successful flow
YELLOW = "#FFD166"     # scalar conditioning
PURPLE = "#B28DFF"     # latent space / target
RED = "#FF7B89"        # loss signal
PINK = "#FF9EE8"

FONT = "Segoe UI"
MATH_FONT = "Cambria Math"
config.background_color = BG


def text(label, size=28, color=INK, weight=NORMAL):
    return Text(label, font=FONT, font_size=size, color=color, weight=weight)


def math_text(label, size=28, color=INK):
    """Formula-like text that renders without requiring a LaTeX install."""
    return Text(label, font=MATH_FONT, font_size=size, color=color)


def title(label, eyebrow=None):
    group = VGroup()
    if eyebrow:
        e = text(eyebrow.upper(), 16, CYAN, BOLD)
        group.add(e)
    h = text(label, 42, INK, BOLD)
    group.add(h)
    group.arrange(DOWN, aligned_edge=LEFT, buff=0.12)
    group.to_edge(UP, buff=0.38)
    return group


def rounded_panel(label, detail, color, width=2.25, height=1.1):
    """A compact architecture block with a title and one explanatory line."""
    body = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.16,
        stroke_color=color,
        stroke_width=2.4,
        fill_color="#101A31",
        fill_opacity=0.94,
    )
    head = text(label, 18, INK, BOLD)
    sub = text(detail, 12, MUTED)
    words = VGroup(head, sub).arrange(DOWN, buff=0.08)
    words.move_to(body)
    return VGroup(body, words)


def connector(start, end, color=MUTED, width=2.2):
    return Arrow(
        start,
        end,
        buff=0.10,
        color=color,
        stroke_width=width,
        max_tip_length_to_length_ratio=0.13,
    )


def glow_dot(point, color, radius=0.10):
    halo = Dot(point, radius=radius * 2.4, color=color, fill_opacity=0.16)
    core = Dot(point, radius=radius, color=color)
    return VGroup(halo, core)


def occupancy_grid(rows=12, cols=12, cell=0.15, seed=7, masked=False):
    """A deterministic, readable stand-in for a 64 x 64 binary occupancy map."""
    rng = np.random.default_rng(seed)
    occupied = rng.random((rows, cols)) > 0.57
    squares = VGroup()
    for row in range(rows):
        for col in range(cols):
            is_hidden = masked and 3 <= row <= 8 and 4 <= col <= 9
            fill = "#19223A" if is_hidden else (BLUE if occupied[row, col] else "#121A2D")
            square = Square(
                side_length=cell,
                stroke_color="#33415F",
                stroke_width=0.45,
                fill_color=fill,
                fill_opacity=0.98,
            )
            square.move_to(
                [
                    (col - (cols - 1) / 2) * cell,
                    ((rows - 1) / 2 - row) * cell,
                    0,
                ]
            )
            squares.add(square)
    if not masked:
        return squares

    cover = RoundedRectangle(
        width=cell * 6.1,
        height=cell * 6.1,
        corner_radius=0.05,
        stroke_color=RED,
        stroke_width=1.8,
        fill_color="#15172D",
        fill_opacity=0.80,
    )
    cover.move_to(squares.get_center() + RIGHT * cell * 1.5 - DOWN * cell * 0.5)
    question = text("?", 38, RED, BOLD).move_to(cover)
    return VGroup(squares, cover, question)


def spectrum_plot(width=2.5, height=1.1, seed=4, color=CYAN):
    rng = np.random.default_rng(seed)
    axes = Axes(
        x_range=[0, 1, 0.25],
        y_range=[0, 1, 0.5],
        x_length=width,
        y_length=height,
        axis_config={"stroke_color": GRID, "stroke_width": 1.2},
        tips=False,
    )
    x = np.linspace(0, 1, 100)
    y = 0.26 + 0.08 * np.sin(13 * x) + 0.06 * np.cos(29 * x)
    for center, strength in [(0.19, 0.38), (0.49, 0.30), (0.76, 0.42)]:
        y += strength * np.exp(-((x - center) ** 2) / 0.0024)
    y += rng.normal(0, 0.008, len(x))
    y = (y - y.min()) / (y.max() - y.min())
    curve = VMobject(color=color, stroke_width=3.2)
    curve.set_points_smoothly([axes.c2p(xi, yi) for xi, yi in zip(x, y)])
    return VGroup(axes, curve)


def token_row(count=11, color=PURPLE, radius=0.075):
    dots = VGroup(*[Dot(radius=radius, color=color) for _ in range(count)])
    dots.arrange(RIGHT, buff=radius * 0.65)
    return dots


def scalar_chips(values=("l", "h", "r"), unknown=False):
    chips = VGroup()
    for value in values:
        shown = "?" if unknown else value
        chip = RoundedRectangle(
            width=0.53,
            height=0.36,
            corner_radius=0.08,
            stroke_color=YELLOW,
            stroke_width=1.5,
            fill_color="#2A2630",
            fill_opacity=0.9,
        )
        label = text(shown, 16, YELLOW, BOLD).move_to(chip)
        chips.add(VGroup(chip, label))
    chips.arrange(RIGHT, buff=0.10)
    return chips


def travel(scene, start, end, color=GREEN, run_time=0.75):
    """Move one bright sample-token through an already drawn architecture edge."""
    particle = glow_dot(start, color)
    scene.add(particle)
    scene.play(MoveAlongPath(particle, Line(start, end)), run_time=run_time, rate_func=linear)
    scene.play(Flash(end, color=color, flash_radius=0.25, line_length=0.11), run_time=0.25)
    scene.remove(particle)


def fade_everything(scene, run_time=0.55):
    if scene.mobjects:
        scene.play(*[FadeOut(mob) for mob in scene.mobjects], run_time=run_time)


def architecture_map():
    """The live unified 192-D architecture; returns a map and named landmarks."""
    occ_input = rounded_panel("Masked occupancy", "1 × 64 × 64", BLUE)
    scalar_input = rounded_panel("Scalar values + flags", "[value, known] × 3", YELLOW)
    spectrum_input = rounded_panel("Target spectrum", "4 × 301 complex response", CYAN)
    occ_encoder = rounded_panel("Occupancy CNN", "192-D patch tokens", BLUE)
    scalar_mlp = rounded_panel("Scalar MLP", "FiLM: γ and β", YELLOW)
    spectrum_encoder = rounded_panel("Frozen SpectrumPath", "goal tokens + cphysics", CYAN)
    fusion = rounded_panel("Fusion transformer", "occupancy + goal tokens", PURPLE, 2.35)
    predictor = rounded_panel("Masked-token predictor", "192-D latent completion", GREEN, 2.45)
    occ_decoder = rounded_panel("Occupancy decoder", "64 × 64 logits", BLUE)
    scalar_decoder = rounded_panel("Scalar heads", "l̂, ĥ, r̂", YELLOW)
    physics = rounded_panel("Frozen EM surrogate", "physics consistency", CYAN, 2.25)
    ema = rounded_panel("EMA target encoder", "full sample; stop-grad", PURPLE, 2.25)
    losses = rounded_panel("Learning signals", "latent + scalar + physics", RED, 2.30)

    occ_input.move_to([-5.7, 1.75, 0])
    scalar_input.move_to([-5.7, 0.20, 0])
    spectrum_input.move_to([-5.7, -1.50, 0])
    occ_encoder.move_to([-3.05, 1.75, 0])
    scalar_mlp.move_to([-3.05, 0.20, 0])
    spectrum_encoder.move_to([-3.05, -1.50, 0])
    fusion.move_to([-0.15, 0.95, 0])
    predictor.move_to([2.65, 0.95, 0])
    occ_decoder.move_to([5.60, 1.75, 0])
    scalar_decoder.move_to([5.60, 0.20, 0])
    physics.move_to([5.60, -1.50, 0])
    ema.move_to([0.15, -2.55, 0])
    losses.move_to([3.15, -2.55, 0])

    edges = VGroup(
        connector(occ_input.get_right(), occ_encoder.get_left(), BLUE),
        connector(scalar_input.get_right(), scalar_mlp.get_left(), YELLOW),
        connector(spectrum_input.get_right(), spectrum_encoder.get_left(), CYAN),
        connector(occ_encoder.get_right(), fusion.get_left(), BLUE),
        connector(scalar_mlp.get_right(), occ_encoder.get_bottom(), YELLOW),
        connector(spectrum_encoder.get_right(), fusion.get_bottom(), CYAN),
        connector(fusion.get_right(), predictor.get_left(), PURPLE),
        connector(predictor.get_right(), occ_decoder.get_left(), BLUE),
        connector(predictor.get_right(), scalar_decoder.get_left(), YELLOW),
        connector(occ_decoder.get_bottom(), physics.get_top(), BLUE),
        connector(scalar_decoder.get_bottom(), physics.get_top(), YELLOW),
        connector(occ_encoder.get_bottom(), ema.get_left(), PURPLE),
        connector(ema.get_right(), losses.get_left(), PURPLE),
        connector(predictor.get_bottom(), losses.get_top(), GREEN),
        connector(physics.get_bottom(), losses.get_right(), CYAN),
    )

    landmark = {
        "occ_input": occ_input,
        "scalar_input": scalar_input,
        "spectrum_input": spectrum_input,
        "occ_encoder": occ_encoder,
        "scalar_mlp": scalar_mlp,
        "spectrum_encoder": spectrum_encoder,
        "fusion": fusion,
        "predictor": predictor,
        "occ_decoder": occ_decoder,
        "scalar_decoder": scalar_decoder,
        "physics": physics,
        "ema": ema,
        "losses": losses,
    }
    return VGroup(edges, *landmark.values()), landmark


# ---------------------------------------------------------------------------
# Scene 1: architecture flight and one training sample
# ---------------------------------------------------------------------------

class ArchitectureFlightScene(MovingCameraScene):
    def construct(self):
        self.camera.frame.set(width=config.frame_width).move_to(ORIGIN)
        self.camera.frame.save_state()
        heading = title("A single sample learns to become a design", "Unified 192-D JEPA")
        self.play(Write(heading), run_time=1.2)

        diagram, nodes = architecture_map()
        diagram.shift(DOWN * 0.35)
        self.play(FadeIn(diagram, shift=UP * 0.2), run_time=1.5)
        overview = text("one conditional model · one physical target · one stable answer key", 18, MUTED)
        overview.next_to(heading, DOWN, buff=0.20)
        self.play(FadeIn(overview))
        self.wait(0.5)

        # The sample first appears in the independent, maskable input streams.
        self.play(self.camera.frame.animate.move_to(LEFT * 4.35 + UP * 0.30).scale(0.66), run_time=1.2)
        sample_tag = text("ONE TRAINING SAMPLE", 16, INK, BOLD)
        sample_tag.next_to(nodes["occ_input"], UP, buff=0.5)
        grid = occupancy_grid(masked=True).scale(0.82).next_to(nodes["occ_input"], LEFT, buff=0.65)
        scalars = scalar_chips(("0.73", "?", "0.28"), unknown=False).scale(0.82)
        scalars.next_to(nodes["scalar_input"], LEFT, buff=0.55)
        unknown_ring = SurroundingRectangle(scalars[1], color=RED, buff=0.06, stroke_width=2)
        spectrum = spectrum_plot(width=2.15, height=0.86).next_to(nodes["spectrum_input"], LEFT, buff=0.55)
        self.play(FadeIn(sample_tag), DrawBorderThenFill(grid), FadeIn(scalars), Create(spectrum))
        self.play(Create(unknown_ring), Indicate(grid[1], color=RED), run_time=0.8)
        mask_note = text("Mask space and scalar knowledge independently", 17, YELLOW)
        mask_note.next_to(spectrum, DOWN, buff=0.30)
        self.play(Write(mask_note))
        self.wait(0.6)

        # Occupancy and scalar encoders meet as FiLM-conditioned tokens.
        self.play(FadeOut(mask_note), FadeOut(unknown_ring), run_time=0.35)
        travel(self, nodes["occ_input"].get_right(), nodes["occ_encoder"].get_left(), BLUE)
        travel(self, nodes["scalar_input"].get_right(), nodes["scalar_mlp"].get_left(), YELLOW)
        self.play(self.camera.frame.animate.move_to(LEFT * 2.55 + UP * 0.85).scale(0.80), run_time=0.9)
        film = math_text("γ(s) ⊙ features + β(s)", 23, YELLOW)
        film.next_to(nodes["scalar_mlp"], DOWN, buff=0.38)
        feature_tokens = token_row(10, BLUE).next_to(nodes["occ_encoder"], UP, buff=0.32)
        self.play(Indicate(nodes["scalar_mlp"], color=YELLOW), Write(film), FadeIn(feature_tokens, shift=RIGHT * 0.2))
        self.play(ApplyWave(feature_tokens, direction=RIGHT), run_time=0.8)
        conditioning = text("the same three physical parameters steer every occupancy block", 15, MUTED)
        conditioning.next_to(film, DOWN, buff=0.18)
        self.play(FadeIn(conditioning))
        self.wait(0.5)

        # The goal carries the desired physics and is frozen.
        self.play(FadeOut(film), FadeOut(conditioning), FadeOut(feature_tokens), run_time=0.35)
        self.play(self.camera.frame.animate.move_to(LEFT * 2.85 + DOWN * 1.15).scale(0.86), run_time=0.8)
        travel(self, nodes["spectrum_input"].get_right(), nodes["spectrum_encoder"].get_left(), CYAN)
        goal_tokens = token_row(8, CYAN).next_to(nodes["spectrum_encoder"], DOWN, buff=0.34)
        lock = text("❄ frozen physical representation", 17, CYAN, BOLD)
        lock.next_to(goal_tokens, DOWN, buff=0.20)
        self.play(FadeIn(goal_tokens, shift=RIGHT * 0.2), Write(lock), Indicate(nodes["spectrum_encoder"], color=CYAN))
        self.wait(0.5)
        self.play(FadeOut(goal_tokens), FadeOut(lock), run_time=0.3)

        # Zoom to fusion and predictor: the core JEPA completion question.
        self.play(self.camera.frame.animate.move_to(RIGHT * 1.25 + UP * 0.95).scale(0.72), run_time=1.0)
        travel(self, nodes["occ_encoder"].get_right(), nodes["fusion"].get_left(), BLUE)
        travel(self, nodes["spectrum_encoder"].get_right(), nodes["fusion"].get_bottom(), CYAN)
        fusion_tokens = VGroup(token_row(7, BLUE), token_row(5, CYAN)).arrange(DOWN, buff=0.12)
        fusion_tokens.next_to(nodes["fusion"], UP, buff=0.32)
        self.play(FadeIn(fusion_tokens, shift=UP * 0.15), Indicate(nodes["fusion"], color=PURPLE))
        self.play(ApplyWave(fusion_tokens, direction=RIGHT), run_time=0.8)
        travel(self, nodes["fusion"].get_right(), nodes["predictor"].get_left(), PURPLE)
        queries = VGroup(*[text("?", 20, GREEN, BOLD) for _ in range(5)]).arrange(RIGHT, buff=0.18)
        queries.next_to(nodes["predictor"], UP, buff=0.33)
        answer = token_row(5, GREEN)
        answer.move_to(queries)
        prompt = text("masked positions ask the context for latent answers", 16, GREEN)
        prompt.next_to(queries, UP, buff=0.20)
        self.play(FadeIn(queries, shift=DOWN * 0.1), Write(prompt))
        self.play(Transform(queries, answer), Flash(nodes["predictor"].get_center(), color=GREEN), run_time=1.0)
        self.wait(0.5)

        # The target branch is shown as an answer key, not another trainable path.
        self.play(FadeOut(queries), FadeOut(prompt), FadeOut(fusion_tokens), run_time=0.35)
        self.play(self.camera.frame.animate.move_to(RIGHT * 0.65 + DOWN * 1.65).scale(0.78), run_time=0.9)
        full_grid = occupancy_grid(seed=7).scale(0.58).next_to(nodes["ema"], LEFT, buff=0.48)
        stable = text("full, unmasked sample", 15, PURPLE)
        stable.next_to(full_grid, DOWN, buff=0.15)
        self.play(FadeIn(full_grid), Write(stable), Indicate(nodes["ema"], color=PURPLE))
        travel(self, nodes["occ_encoder"].get_bottom(), nodes["ema"].get_left(), PURPLE, 0.55)
        target_tokens = token_row(5, PURPLE).next_to(nodes["ema"], UP, buff=0.32)
        predicted_tokens = token_row(5, GREEN).next_to(nodes["predictor"], DOWN, buff=0.31)
        self.play(FadeIn(target_tokens), FadeIn(predicted_tokens))
        self.play(Indicate(nodes["losses"], color=RED), run_time=0.7)
        comparison = text("compare predicted and stable latent structure", 17, RED, BOLD)
        comparison.next_to(nodes["losses"], DOWN, buff=0.28)
        self.play(Write(comparison))
        self.wait(0.5)

        # Physics loss gives the completion an electromagnetic reality check.
        self.play(
            FadeOut(full_grid), FadeOut(stable), FadeOut(target_tokens),
            FadeOut(predicted_tokens), FadeOut(comparison), run_time=0.35,
        )
        self.play(self.camera.frame.animate.move_to(RIGHT * 4.75 + DOWN * 0.10).scale(0.70), run_time=0.9)
        travel(self, nodes["predictor"].get_right(), nodes["occ_decoder"].get_left(), BLUE)
        travel(self, nodes["predictor"].get_right(), nodes["scalar_decoder"].get_left(), YELLOW)
        decoded_grid = occupancy_grid(seed=19).scale(0.52).next_to(nodes["occ_decoder"], UP, buff=0.32)
        decoded_scalars = scalar_chips(("l̂", "ĥ", "r̂")).scale(0.75).next_to(nodes["scalar_decoder"], UP, buff=0.32)
        decoded_spectrum = spectrum_plot(width=2.0, height=0.75, seed=11, color=GREEN)
        decoded_spectrum.next_to(nodes["physics"], DOWN, buff=0.33)
        self.play(FadeIn(decoded_grid), FadeIn(decoded_scalars), Create(decoded_spectrum))
        travel(self, nodes["occ_decoder"].get_bottom(), nodes["physics"].get_top(), BLUE, 0.45)
        physics_note = text("Does this design reproduce the requested response?", 16, CYAN, BOLD)
        physics_note.next_to(decoded_spectrum, DOWN, buff=0.22)
        self.play(Write(physics_note), Flash(nodes["physics"].get_center(), color=CYAN))
        self.wait(0.8)

        # The camera finally pulls back: it was one coherent training step.
        self.play(
            FadeOut(decoded_grid), FadeOut(decoded_scalars), FadeOut(decoded_spectrum), FadeOut(physics_note),
            FadeOut(grid), FadeOut(scalars), FadeOut(spectrum), FadeOut(sample_tag),
            run_time=0.45,
        )
        self.play(Restore(self.camera.frame), run_time=1.3)
        self.play(Indicate(diagram, color=GREEN, scale_factor=1.015), run_time=1.0)
        closing = text("One sample.  Three views of correctness.  A single update.", 23, GREEN, BOLD)
        closing.to_edge(DOWN, buff=0.35)
        self.play(Write(closing))
        self.wait(1.3)


# ---------------------------------------------------------------------------
# Scene 2: what actually changes during training
# ---------------------------------------------------------------------------

class WeightUpdateScene(Scene):
    def construct(self):
        heading = title("The instant learning happens", "Training dynamics")
        self.play(Write(heading))

        loss = rounded_panel("Combined loss", "latent + scalar + physics", RED, 2.55, 1.15)
        student = rounded_panel("Student network", "CNN · MLP · fusion · predictor", GREEN, 3.25, 1.25)
        ema = rounded_panel("EMA target", "a slowly moving answer key", PURPLE, 2.65, 1.15)
        frozen = rounded_panel("Frozen physics", "SpectrumPath + EM surrogate", CYAN, 2.75, 1.15)
        loss.move_to(LEFT * 4.8 + UP * 0.45)
        student.move_to(ORIGIN + UP * 0.45)
        ema.move_to(RIGHT * 4.25 + UP * 1.55)
        frozen.move_to(RIGHT * 4.25 + DOWN * 1.25)

        backward = connector(loss.get_right(), student.get_left(), RED, 3.0)
        ema_link = connector(student.get_right(), ema.get_left(), PURPLE, 2.5)
        self.play(FadeIn(loss), FadeIn(student), FadeIn(ema), FadeIn(frozen), GrowArrow(backward), GrowArrow(ema_link))
        self.play(Write(text("backpropagation", 17, RED, BOLD).next_to(backward, UP, buff=0.14)))

        # Weight knobs make the otherwise invisible update tangible.
        sliders = VGroup()
        tracker_specs = [
            ("occupancy filters", BLUE, 0.22, 0.73),
            ("FiLM conditioning", YELLOW, 0.70, 0.38),
            ("attention weights", PURPLE, 0.31, 0.82),
            ("decoder weights", GREEN, 0.65, 0.45),
        ]
        trackers = []
        for index, (label, color, start, end) in enumerate(tracker_specs):
            y = 1.55 - index * 0.75
            name = text(label, 16, MUTED).move_to(LEFT * 4.8 + RIGHT * 0.10 + UP * y)
            rail = Line(LEFT * 2.55 + UP * y, LEFT * 0.55 + UP * y, color=GRID, stroke_width=5)
            progress = Line(rail.get_start(), rail.get_start(), color=color, stroke_width=5)
            value = ValueTracker(start)
            knob = always_redraw(lambda rail=rail, value=value, color=color: glow_dot(rail.point_from_proportion(value.get_value()), color, 0.07))
            fill = always_redraw(
                lambda rail=rail, value=value, color=color: Line(
                    rail.get_start(), rail.point_from_proportion(value.get_value()), color=color, stroke_width=5
                )
            )
            sliders.add(name, rail, fill, knob)
            trackers.append((value, end))
        sliders.shift(DOWN * 1.1)
        self.play(FadeIn(sliders, shift=RIGHT * 0.2))

        signal = text("error signal", 16, RED, BOLD).next_to(loss, DOWN, buff=0.35)
        self.play(FadeIn(signal), Indicate(loss, color=RED), run_time=0.7)
        travel(self, loss.get_right(), student.get_left(), RED, 0.8)
        self.play(*[tracker.animate.set_value(end) for tracker, end in trackers], run_time=1.7, rate_func=there_and_back_with_pause)
        update = text("small, coordinated parameter changes", 20, GREEN, BOLD)
        update.next_to(sliders, DOWN, buff=0.36)
        self.play(Write(update), Indicate(student, color=GREEN))

        # EMA is intentionally a lagging copy, never a gradient destination.
        student_pulse = glow_dot(student.get_right(), GREEN)
        self.add(student_pulse)
        self.play(MoveAlongPath(student_pulse, Line(student.get_right(), ema.get_left())), run_time=1.0, rate_func=linear)
        self.remove(student_pulse)
        slow = text("τ · old target + (1 − τ) · student", 17, PURPLE)
        slow.next_to(ema, UP, buff=0.28)
        stop = text("no gradient", 15, PURPLE, BOLD).next_to(ema, DOWN, buff=0.18)
        lock = text("no gradient", 15, CYAN, BOLD).next_to(frozen, DOWN, buff=0.18)
        self.play(Write(slow), FadeIn(stop), FadeIn(lock), Indicate(ema, color=PURPLE), Indicate(frozen, color=CYAN))
        self.wait(1.0)

        takeaway = text("Learnable parts adapt.  Targets stabilize.  Physics stays fixed.", 25, INK, BOLD)
        takeaway.to_edge(DOWN, buff=0.35)
        self.play(Write(takeaway))
        self.wait(1.2)


# ---------------------------------------------------------------------------
# Scene 3: inference is a direct, spectrum-conditioned design journey
# ---------------------------------------------------------------------------

class InferenceJourneyScene(MovingCameraScene):
    def construct(self):
        self.camera.frame.set(width=config.frame_width).move_to(ORIGIN)
        heading = title("Now reverse the question", "Inference")
        self.play(Write(heading))
        subtitle = text("Desired electromagnetic response in → manufacturable design out", 20, MUTED)
        subtitle.next_to(heading, DOWN, buff=0.20)
        self.play(FadeIn(subtitle))

        desired = spectrum_plot(width=3.4, height=1.55, seed=23, color=CYAN)
        desired.move_to(LEFT * 5.0 + UP * 0.45)
        desired_label = text("desired spectrum", 22, CYAN, BOLD).next_to(desired, DOWN, buff=0.28)
        unknown_geometry = occupancy_grid(masked=True).scale(0.8).move_to(LEFT * 5.0 + DOWN * 2.05)
        unknown_label = text("occupancy unknown", 17, MUTED).next_to(unknown_geometry, DOWN, buff=0.20)
        unknown_scalars = scalar_chips(("l", "h", "r"), unknown=True).move_to(LEFT * 5.0 + DOWN * 3.05)
        self.play(Create(desired), Write(desired_label), FadeIn(unknown_geometry), Write(unknown_label), FadeIn(unknown_scalars))

        # A compact inference-only path, deliberately without loss or EMA branches.
        spectrum_encoder = rounded_panel("Frozen SpectrumPath", "encode the goal", CYAN, 2.65, 1.15)
        fusion_predictor = rounded_panel("Fusion + predictor", "complete masked 192-D tokens", PURPLE, 3.0, 1.15)
        decode = rounded_panel("Decoders", "occupancy + l̂, ĥ, r̂", GREEN, 2.65, 1.15)
        verify = rounded_panel("Frozen EM surrogate", "check the response", CYAN, 2.75, 1.15)
        spectrum_encoder.move_to(LEFT * 2.0 + UP * 0.45)
        fusion_predictor.move_to(RIGHT * 1.25 + UP * 0.45)
        decode.move_to(RIGHT * 4.55 + UP * 0.45)
        verify.move_to(RIGHT * 4.55 + DOWN * 2.20)
        edges = VGroup(
            connector(desired.get_right(), spectrum_encoder.get_left(), CYAN, 2.8),
            connector(spectrum_encoder.get_right(), fusion_predictor.get_left(), PURPLE, 2.8),
            connector(fusion_predictor.get_right(), decode.get_left(), GREEN, 2.8),
            connector(decode.get_bottom(), verify.get_top(), CYAN, 2.8),
        )
        self.play(FadeIn(spectrum_encoder), FadeIn(fusion_predictor), FadeIn(decode), FadeIn(verify), GrowArrow(edges[0]))
        travel(self, desired.get_right(), spectrum_encoder.get_left(), CYAN)
        self.play(GrowArrow(edges[1]))
        travel(self, spectrum_encoder.get_right(), fusion_predictor.get_left(), PURPLE)

        # The unknown queries turn into a design, rather than a numerical optimization loop.
        unknown_queries = VGroup(*[text("?", 25, GREEN, BOLD) for _ in range(7)]).arrange(RIGHT, buff=0.12)
        unknown_queries.next_to(fusion_predictor, DOWN, buff=0.32)
        latent = token_row(7, GREEN, radius=0.09).move_to(unknown_queries)
        self.play(FadeIn(unknown_queries, shift=DOWN * 0.15), Indicate(fusion_predictor, color=GREEN))
        self.play(Transform(unknown_queries, latent), Flash(fusion_predictor.get_center(), color=GREEN), run_time=1.0)
        self.play(GrowArrow(edges[2]))
        travel(self, fusion_predictor.get_right(), decode.get_left(), GREEN)

        output_grid = occupancy_grid(seed=43).scale(0.88).next_to(decode, DOWN, buff=0.48)
        output_scalars = scalar_chips(("0.71", "0.39", "0.30")).scale(0.85).next_to(output_grid, DOWN, buff=0.22)
        output_label = text("a proposed metasurface", 17, GREEN, BOLD).next_to(output_scalars, DOWN, buff=0.18)
        self.play(FadeIn(output_grid, shift=UP * 0.15), FadeIn(output_scalars), Write(output_label))
        self.play(GrowArrow(edges[3]))
        travel(self, decode.get_bottom(), verify.get_top(), CYAN, 0.6)
        returned = spectrum_plot(width=2.2, height=0.78, seed=24, color=GREEN).next_to(verify, DOWN, buff=0.30)
        match = text("compare against the requested spectrum", 16, CYAN).next_to(returned, DOWN, buff=0.18)
        self.play(Create(returned), Write(match), Flash(verify.get_center(), color=CYAN))
        self.wait(0.7)

        no_loop = text("One conditioned forward pass — no gradient descent at inference.", 25, GREEN, BOLD)
        no_loop.to_edge(DOWN, buff=0.30)
        self.play(Write(no_loop))
        self.wait(1.2)


# ---------------------------------------------------------------------------
# Scene 4: a visual closing frame for the finished film
# ---------------------------------------------------------------------------

class ClosingScene(Scene):
    def construct(self):
        heading = title("A representation that speaks physics", "The central idea")
        self.play(Write(heading))

        left = occupancy_grid(seed=12).scale(1.35).shift(LEFT * 4.6)
        middle = VGroup(
            token_row(12, PURPLE, 0.10),
            text("192-D conditional latent space", 20, PURPLE, BOLD),
        ).arrange(DOWN, buff=0.35)
        right = spectrum_plot(width=3.4, height=1.45, seed=12, color=CYAN).shift(RIGHT * 4.5)
        left_label = text("geometry", 20, BLUE, BOLD).next_to(left, DOWN, buff=0.28)
        right_label = text("physical response", 20, CYAN, BOLD).next_to(right, DOWN, buff=0.28)
        arrows = VGroup(
            connector(left.get_right(), middle.get_left(), PURPLE, 3.0),
            connector(middle.get_right(), right.get_left(), CYAN, 3.0),
        )
        self.play(FadeIn(left), FadeIn(middle), Create(right), Write(left_label), Write(right_label))
        self.play(GrowArrow(arrows[0]), GrowArrow(arrows[1]))
        for color, start, end in [(BLUE, left.get_right(), middle.get_left()), (CYAN, right.get_left(), middle.get_right())]:
            travel(self, start, end, color, 0.65)
        statement = text(
            "Learn a stable latent prediction problem.\nDecode a design that satisfies a physical goal.",
            29,
            INK,
            BOLD,
        )
        statement.to_edge(DOWN, buff=0.42)
        self.play(Write(statement))
        self.wait(2.0)


class JEPAFullPresentation(MovingCameraScene):
    """Render the whole story as one continuous video."""

    def construct(self):
        scenes = [
            ArchitectureFlightScene,
            WeightUpdateScene,
            InferenceJourneyScene,
            ClosingScene,
        ]
        for index, scene_class in enumerate(scenes):
            scene_class.construct(self)
            if index < len(scenes) - 1:
                fade_everything(self)
                self.camera.frame.set(width=config.frame_width).move_to(ORIGIN)
