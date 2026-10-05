from manim import *
import numpy as np

# ============================================================
# JEPA 192-D METASURFACE EXPLAINER
# 3Blue1Brown-inspired mathematical animation
#
# Render:
#
#   manim -pqh manim_animation.py JEPAFullPresentation
#
# Individual scene:
#
#   manim -pqh manim_animation.py ProblemScene
#
# High quality:
#
#   manim -p -qh manim_animation.py JEPAFullPresentation
#
# ============================================================


# ------------------------------------------------------------
# GLOBAL STYLE
# ------------------------------------------------------------

BG = "#111111"
WHITE = "#F5F5F5"
GREY = "#A0A0A0"

BLUE = "#4EA5FF"
CYAN = "#45D9FF"
GREEN = "#5BE37D"
YELLOW = "#FFD166"
RED = "#FF6B6B"
PURPLE = "#B388FF"
ORANGE = "#FF9F43"

config.background_color = BG

FONT = "Segoe UI"
MATH_FONT = "Cambria Math"


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def title(text, size=42):
    return Text(
        text,
        font=FONT,
        font_size=size,
        color=WHITE,
        weight=BOLD
    )


def subtitle(text, size=25):
    return Text(
        text,
        font=FONT,
        font_size=size,
        color=GREY
    )


def formula(text, size=36, color=WHITE):
    """Create a formula without requiring a separate LaTeX installation."""
    return Text(
        text,
        font=MATH_FONT,
        font_size=size,
        color=color,
    )


def box(text, width=2.8, height=1.0, color=BLUE, fs=23):
    r = RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.12,
        stroke_color=color,
        stroke_width=2
    )
    t = Text(text, font=FONT, font_size=fs, color=WHITE)
    t.move_to(r.get_center())
    return VGroup(r, t)


def arrow(a, b, color=WHITE, buff=0.1):
    return Arrow(
        a,
        b,
        buff=buff,
        stroke_width=2.5,
        color=color,
        max_tip_length_to_length_ratio=0.12
    )


def label_box(text, pos, color=BLUE, width=2.8):
    g = box(text, width=width, color=color)
    g.move_to(pos)
    return g


def matrix_grid(
    rows,
    cols,
    cell=0.12,
    colors=None,
    stroke=GREY
):
    cells = VGroup()

    for i in range(rows):
        for j in range(cols):

            sq = Square(
                side_length=cell,
                stroke_width=0.4,
                stroke_color=stroke
            )

            sq.move_to(
                np.array([
                    (j - (cols - 1) / 2) * cell,
                    ((rows - 1) / 2 - i) * cell,
                    0
                ])
            )

            if colors:
                sq.set_fill(
                    colors[i][j],
                    opacity=1
                )

            cells.add(sq)

    return cells


def random_binary_grid(
    rows,
    cols,
    p=0.5,
    seed=1,
    cell=0.11
):
    rng = np.random.default_rng(seed)

    data = rng.random((rows, cols)) < p

    colors = [
        [
            WHITE if data[i, j] else "#222222"
            for j in range(cols)
        ]
        for i in range(rows)
    ]

    return matrix_grid(
        rows,
        cols,
        cell=cell,
        colors=colors,
        stroke="#444444"
    )


def spectrum_curve(
    width=5,
    height=2,
    seed=2,
    color=CYAN
):
    rng = np.random.default_rng(seed)

    x = np.linspace(0, 1, 150)

    y = (
        0.25 * np.sin(8 * x)
        + 0.13 * np.sin(21 * x)
        + 0.08 * np.sin(47 * x)
    )

    # Add resonances
    for center in [0.23, 0.47, 0.71]:
        y += 0.45 * np.exp(
            -((x - center) ** 2) / 0.003
        )

    y += 0.02 * rng.normal(size=len(x))

    y = (y - y.min()) / (y.max() - y.min())

    pts = []

    for xi, yi in zip(x, y):
        pts.append(
            np.array([
                (xi - 0.5) * width,
                (yi - 0.5) * height,
                0
            ])
        )

    curve = VMobject()
    curve.set_points_smoothly(pts)
    curve.set_stroke(color, width=3)

    axes = Axes(
        x_range=[0, 1, 0.2],
        y_range=[0, 1, 0.25],
        x_length=width,
        y_length=height,
        axis_config={
            "stroke_color": "#555555",
            "stroke_width": 1
        },
        tips=False
    )

    curve.move_to(axes.get_center())

    return VGroup(axes, curve)


def latent_vector(n=192, width=6):
    cells = VGroup()

    cell_w = width / n

    rng = np.random.default_rng(5)

    for i in range(n):

        value = rng.random()

        rect = Rectangle(
            width=cell_w * 0.9,
            height=0.35,
            stroke_width=0,
            fill_color=interpolate_color(
                ManimColor(BLUE),
                ManimColor(PURPLE),
                value
            ),
            fill_opacity=0.9
        )

        rect.move_to(
            LEFT * width / 2
            + RIGHT * (i + 0.5) * cell_w
        )

        cells.add(rect)

    return cells


# ============================================================
# SCENE 1
# ============================================================

class ProblemScene(Scene):

    def construct(self):

        t = title(
            "The inverse electromagnetic-design problem",
            42
        )

        self.play(Write(t))

        self.wait(1)

        self.play(t.animate.to_edge(UP))

        # Geometry
        geom = random_binary_grid(
            12,
            12,
            p=0.42,
            seed=4,
            cell=0.13
        )

        geom.scale(1.2)
        geom.shift(LEFT * 3)

        geom_label = Text(
            "Geometry G",
            font=FONT,
            font_size=26,
            color=BLUE
        )

        geom_label.next_to(geom, DOWN)

        # Spectrum
        spec = spectrum_curve(
            width=4.5,
            height=2.5,
            seed=8
        )

        spec.shift(RIGHT * 2.5)

        spec_label = Text(
            "Electromagnetic spectrum S",
            font=FONT,
            font_size=24,
            color=CYAN
        )

        spec_label.next_to(spec, DOWN)

        self.play(
            FadeIn(geom),
            FadeIn(geom_label)
        )

        self.play(
            FadeIn(spec),
            FadeIn(spec_label)
        )

        self.play(
            GrowArrow(
                arrow(
                    geom.get_right(),
                    spec.get_left(),
                    color=YELLOW
                )
            )
        )

        forward = Text(
            "Forward physics",
            font=FONT,
            font_size=24,
            color=YELLOW
        )

        forward.move_to(
            (geom.get_right() + spec.get_left()) / 2
            + UP * 0.35
        )

        self.play(FadeIn(forward))

        self.wait(2)

        inverse = Text(
            "But we want the reverse:",
            font=FONT,
            font_size=30
        )

        inverse.to_edge(DOWN)

        self.play(Write(inverse))

        self.wait(1)

        self.play(
            geom.animate.set_opacity(0.3),
            spec.animate.set_color(GREEN)
        )

        target = Text(
            "Given a target spectrum → recover a geometry",
            font=FONT,
            font_size=32,
            color=GREEN
        )

        target.to_edge(DOWN)

        self.play(
            Transform(inverse, target)
        )

        self.wait(2)


# ============================================================
# SCENE 2
# ============================================================

class DatasetScene(Scene):

    def construct(self):

        t = title("What does one dataset sample contain?")

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        geom = random_binary_grid(
            16,
            16,
            p=0.4,
            seed=12,
            cell=0.085
        )

        geom.move_to(LEFT * 4)

        gl = Text(
            "64 × 64 occupancy",
            font=FONT,
            font_size=24,
            color=BLUE
        )

        gl.next_to(geom, DOWN)

        self.play(
            FadeIn(geom),
            Write(gl)
        )

        scalars = VGroup(
            Text("l", font=FONT, font_size=30),
            Text("h", font=FONT, font_size=30),
            Text("r", font=FONT, font_size=30)
        )

        scalar_values = VGroup(
            Text("l = 0.73", font=FONT, font_size=24),
            Text("h = 0.41", font=FONT, font_size=24),
            Text("r = 0.28", font=FONT, font_size=24)
        )

        scalar_group = VGroup(
            *[
                VGroup(a, b)
                for a, b in zip(
                    scalars,
                    scalar_values
                )
            ]
        )

        scalar_group.arrange(
            DOWN,
            buff=0.35
        )

        scalar_group.move_to(ORIGIN)

        self.play(
            LaggedStart(
                *[
                    FadeIn(x)
                    for x in scalar_group
                ],
                lag_ratio=0.2
            )
        )

        spec = spectrum_curve(
            width=4,
            height=2.2,
            seed=3
        )

        spec.move_to(RIGHT * 3.5)

        sl = Text(
            "S ∈ C^(4 × 301)",
            font=FONT,
            font_size=25,
            color=CYAN
        )

        sl.next_to(spec, DOWN)

        self.play(
            FadeIn(spec),
            Write(sl)
        )

        self.wait(2)

        eq = formula("(G, l, h, r, S)", size=40)

        eq.to_edge(DOWN)

        self.play(Write(eq))

        self.wait(2)


# ============================================================
# SCENE 3
# ============================================================

class FactorizationScene(Scene):

    def construct(self):

        t = title(
            "First hidden idea: factor the information"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        original = box(
            "3-channel tensor\n[occupancy + scalars]",
            width=3.6,
            height=1.4,
            color=RED,
            fs=22
        )

        original.shift(LEFT * 4)

        self.play(FadeIn(original))

        split1 = box(
            "Occupancy\nG",
            width=2.2,
            height=1.1,
            color=BLUE
        )

        split2 = box(
            "Scalars\nl,h,r",
            width=2.2,
            height=1.1,
            color=YELLOW
        )

        split1.move_to(RIGHT * 0.5 + UP * 1)
        split2.move_to(RIGHT * 0.5 + DOWN * 1)

        self.play(
            GrowArrow(
                arrow(
                    original.get_right(),
                    split1.get_left(),
                    BLUE
                )
            ),
            GrowArrow(
                arrow(
                    original.get_right(),
                    split2.get_left(),
                    YELLOW
                )
            )
        )

        self.play(
            FadeIn(split1),
            FadeIn(split2)
        )

        reason = Text(
            "Different information should get different inductive biases.",
            font=FONT,
            font_size=27,
            color=GREEN
        )

        reason.to_edge(DOWN)

        self.play(Write(reason))

        self.wait(2)


# ============================================================
# SCENE 4
# ============================================================

class LatentScene(Scene):

    def construct(self):

        t = title(
            "Everything eventually meets in a 192-D latent space"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        inputs = [
            ("Occupancy", BLUE, LEFT * 4 + UP * 1.5),
            ("Scalars", YELLOW, LEFT * 4),
            ("Spectrum", CYAN, LEFT * 4 + DOWN * 1.5)
        ]

        encoders = VGroup()

        for name, color, pos in inputs:

            b = box(
                name,
                width=2.3,
                color=color
            )

            b.move_to(pos)

            encoders.add(b)

        self.play(
            LaggedStart(
                *[FadeIn(x) for x in encoders],
                lag_ratio=0.2
            )
        )

        latent = latent_vector(
            192,
            width=5.5
        )

        latent.move_to(RIGHT * 3)

        latent_label = Text(
            "z ∈ R^192",
            font=FONT,
            font_size=34,
            color=PURPLE
        )

        latent_label.next_to(latent, DOWN)

        for e in encoders:

            self.play(
                GrowArrow(
                    arrow(
                        e.get_right(),
                        latent.get_left(),
                        color=WHITE
                    )
                ),
                run_time=0.5
            )

        self.play(
            FadeIn(latent),
            Write(latent_label)
        )

        self.wait(2)

        note = Text(
            "The network is not trying to predict pixels directly.",
            font=FONT,
            font_size=27,
            color=GREEN
        )

        note.to_edge(DOWN)

        self.play(Write(note))

        self.wait(2)


# ============================================================
# SCENE 5
# ============================================================

class ArchitectureScene(Scene):

    def construct(self):

        t = title("The Unified 192-D JEPA")

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        occ = label_box(
            "Occupancy\nEncoder",
            LEFT * 5 + UP * 1.8,
            BLUE
        )

        scal = label_box(
            "Scalar\nEncoder",
            LEFT * 5,
            YELLOW
        )

        spec = label_box(
            "Frozen Spectrum\nEncoder",
            LEFT * 5 + DOWN * 1.8,
            CYAN
        )

        fusion = label_box(
            "Fusion\nTransformer",
            LEFT * 1,
            color=PURPLE,
            width=3
        )

        pred = label_box(
            "192-D Predictor",
            RIGHT * 3,
            color=GREEN,
            width=3
        )

        decoder = label_box(
            "Geometry\nDecoder",
            RIGHT * 5.5 + UP * 1,
            color=BLUE
        )

        spectrum = label_box(
            "Frozen EM\nSurrogate",
            RIGHT * 5.5 + DOWN * 1,
            color=RED
        )

        self.play(
            FadeIn(occ),
            FadeIn(scal),
            FadeIn(spec)
        )

        for x in [occ, scal, spec]:
            self.play(
                GrowArrow(
                    arrow(
                        x.get_right(),
                        fusion.get_left(),
                        color=WHITE
                    )
                ),
                run_time=0.4
            )

        self.play(FadeIn(fusion))

        self.play(
            GrowArrow(
                arrow(
                    fusion.get_right(),
                    pred.get_left(),
                    color=WHITE
                )
            )
        )

        self.play(FadeIn(pred))

        self.play(
            GrowArrow(
                arrow(
                    pred.get_right(),
                    decoder.get_left(),
                    color=GREEN
                )
            )
        )

        self.play(
            GrowArrow(
                arrow(
                    decoder.get_right(),
                    spectrum.get_left(),
                    color=RED
                )
            )
        )

        self.play(
            FadeIn(decoder),
            FadeIn(spectrum)
        )

        self.wait(2)


# ============================================================
# SCENE 6
# ============================================================

class SpectrumEncoderScene(Scene):

    def construct(self):

        t = title(
            "The spectrum is already a physical object"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        spec = spectrum_curve(
            width=4.5,
            height=2.5,
            seed=7
        )

        spec.move_to(LEFT * 3.5)

        label = Text(
            "4 × 301 complex response",
            font=FONT,
            font_size=25,
            color=CYAN
        )

        label.next_to(spec, DOWN)

        self.play(
            FadeIn(spec),
            Write(label)
        )

        enc = box(
            "Released\nSpectrum Encoder",
            width=3.3,
            height=1.3,
            color=CYAN
        )

        enc.move_to(ORIGIN)

        self.play(
            GrowArrow(
                arrow(
                    spec.get_right(),
                    enc.get_left(),
                    CYAN
                )
            ),
            FadeIn(enc)
        )

        tokens = VGroup()

        for i in range(8):

            c = Circle(
                radius=0.13,
                stroke_width=0,
                fill_color=CYAN,
                fill_opacity=0.8
            )

            tokens.add(c)

        tokens.arrange(RIGHT, buff=0.08)
        tokens.move_to(RIGHT * 3.5)

        self.play(
            GrowArrow(
                arrow(
                    enc.get_right(),
                    tokens.get_left(),
                    CYAN
                )
            ),
            FadeIn(tokens)
        )

        txt = Text(
            "The physics representation is kept frozen.",
            font=FONT,
            font_size=26,
            color=YELLOW
        )

        txt.to_edge(DOWN)

        self.play(Write(txt))

        self.wait(2)


# ============================================================
# SCENE 7
# ============================================================

class OccupancyEncoderScene(Scene):

    def construct(self):

        t = title(
            "Occupancy is processed as geometry"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        grid = random_binary_grid(
            16,
            16,
            0.42,
            15,
            0.11
        )

        grid.move_to(LEFT * 4)

        self.play(FadeIn(grid))

        cnn = box(
            "2D CNN\nEncoder",
            width=2.8,
            height=1.3,
            color=BLUE
        )

        cnn.move_to(ORIGIN)

        self.play(
            GrowArrow(
                arrow(
                    grid.get_right(),
                    cnn.get_left(),
                    BLUE
                )
            ),
            FadeIn(cnn)
        )

        z = box(
            "z_G\n192-D",
            width=2.3,
            height=1.1,
            color=PURPLE
        )

        z.move_to(RIGHT * 3.5)

        self.play(
            GrowArrow(
                arrow(
                    cnn.get_right(),
                    z.get_left(),
                    PURPLE
                )
            ),
            FadeIn(z)
        )

        note = Text(
            "Local spatial structure → latent representation",
            font=FONT,
            font_size=27,
            color=GREEN
        )

        note.to_edge(DOWN)

        self.play(Write(note))

        self.wait(2)


# ============================================================
# SCENE 8
# ============================================================

class ScalarFiLMScene(Scene):

    def construct(self):

        t = title(
            "Scalars enter differently: through conditioning"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        scalars = VGroup(
            formula("l"),
            formula("h"),
            formula("r")
        )

        scalars.arrange(DOWN, buff=0.35)
        scalars.move_to(LEFT * 4)

        self.play(
            LaggedStart(
                *[Write(x) for x in scalars],
                lag_ratio=0.2
            )
        )

        mlp = box(
            "Scalar MLP",
            width=2.6,
            color=YELLOW
        )

        mlp.move_to(LEFT * 1.5)

        self.play(
            GrowArrow(
                arrow(
                    scalars.get_right(),
                    mlp.get_left(),
                    YELLOW
                )
            ),
            FadeIn(mlp)
        )

        film = box(
            "FiLM\nγ, β",
            width=2.3,
            height=1.3,
            color=ORANGE
        )

        film.move_to(RIGHT * 1.5)

        self.play(
            GrowArrow(
                arrow(
                    mlp.get_right(),
                    film.get_left(),
                    ORANGE
                )
            ),
            FadeIn(film)
        )

        latent = box(
            "Occupancy\nfeatures",
            width=2.8,
            color=BLUE
        )

        latent.move_to(RIGHT * 4)

        self.play(
            GrowArrow(
                arrow(
                    film.get_right(),
                    latent.get_left(),
                    GREEN
                )
            ),
            FadeIn(latent)
        )

        equation = formula("FiLM(x) = γ(s) ⊙ x + β(s)", size=35)

        equation.to_edge(DOWN)

        self.play(Write(equation))

        self.wait(2)


# ============================================================
# SCENE 9
# ============================================================

class FusionScene(Scene):

    def construct(self):

        t = title(
            "Fusion: information becomes tokens"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        labels = [
            ("geometry", BLUE),
            ("scalar", YELLOW),
            ("spectrum", CYAN),
            ("goal", GREEN)
        ]

        tokens = VGroup()

        for i, (name, color) in enumerate(labels):

            c = RoundedRectangle(
                width=2,
                height=0.7,
                corner_radius=0.1,
                stroke_color=color,
                stroke_width=2
            )

            txt = Text(
                name,
                font=FONT,
                font_size=21
            )

            txt.move_to(c.get_center())

            tokens.add(
                VGroup(c, txt)
            )

        tokens.arrange(
            RIGHT,
            buff=0.25
        )

        tokens.move_to(ORIGIN)

        self.play(
            LaggedStart(
                *[
                    FadeIn(x, shift=DOWN)
                    for x in tokens
                ],
                lag_ratio=0.15
            )
        )

        self.wait(1)

        transformer = box(
            "Fusion Transformer",
            width=4,
            height=1.4,
            color=PURPLE,
            fs=28
        )

        transformer.move_to(DOWN * 1.7)

        self.play(
            GrowArrow(
                arrow(
                    tokens.get_bottom(),
                    transformer.get_top(),
                    PURPLE
                )
            )
        )

        self.play(FadeIn(transformer))

        note = Text(
            "Self-attention lets the representations interact.",
            font=FONT,
            font_size=28,
            color=GREEN
        )

        note.to_edge(DOWN)

        self.play(Write(note))

        self.wait(2)


# ============================================================
# SCENE 10
# ============================================================

class JEPAScene(Scene):

    def construct(self):

        t = title(
            "JEPA changes what we ask the network to predict"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        context = box(
            "Context\nx",
            width=2.4,
            height=1.2,
            color=BLUE
        )

        target = box(
            "Target\n y",
            width=2.4,
            height=1.2,
            color=CYAN
        )

        predictor = box(
            "Predictor\nPθ",
            width=2.7,
            height=1.2,
            color=GREEN
        )

        context.move_to(LEFT * 4)
        predictor.move_to(ORIGIN)
        target.move_to(RIGHT * 4)

        self.play(
            FadeIn(context),
            FadeIn(target)
        )

        self.play(
            GrowArrow(
                arrow(
                    context.get_right(),
                    predictor.get_left(),
                    GREEN
                )
            )
        )

        self.play(FadeIn(predictor))

        prediction = formula("ẑᵧ = Pθ(zₓ)", size=35, color=GREEN)

        prediction.move_to(
            ORIGIN + DOWN * 1.5
        )

        self.play(Write(prediction))

        self.play(
            GrowArrow(
                arrow(
                    predictor.get_right(),
                    target.get_left(),
                    GREEN
                )
            )
        )

        self.wait(2)

        loss = formula("LJEPA = d(ẑᵧ, zᵧ)", size=38)

        loss.to_edge(DOWN)

        self.play(Write(loss))

        self.wait(2)


# ============================================================
# SCENE 11
# ============================================================

class EMAScene(Scene):

    def construct(self):

        t = title(
            "But the target network does not chase the student"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        student = box(
            "Student\nθ",
            width=2.5,
            color=GREEN
        )

        target = box(
            "EMA Target\nξ",
            width=2.5,
            color=CYAN
        )

        student.move_to(LEFT * 3)
        target.move_to(RIGHT * 3)

        self.play(
            FadeIn(student),
            FadeIn(target)
        )

        eq = formula("ξ ← τξ + (1 − τ)θ", size=38)

        eq.move_to(DOWN * 1.5)

        self.play(Write(eq))

        arrow1 = Arrow(
            student.get_right(),
            target.get_left(),
            color=YELLOW,
            stroke_width=2.5
        )

        self.play(GrowArrow(arrow1))

        label = Text(
            "slow moving target",
            font=FONT,
            font_size=26,
            color=YELLOW
        )

        label.next_to(arrow1, UP)

        self.play(Write(label))

        self.wait(2)


# ============================================================
# SCENE 12
# ============================================================

class VICRegScene(Scene):

    def construct(self):

        t = title(
            "Why doesn't the latent space collapse?"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        terms = [
            (
                "Linv",
                "same information\n→ same representation",
                BLUE
            ),
            (
                "Lvar",
                "keep dimensions\nalive",
                GREEN
            ),
            (
                "Lcov",
                "avoid redundant\ndimensions",
                PURPLE
            )
        ]

        groups = VGroup()

        for formula_text, text, color in terms:

            f = formula(formula_text, size=38, color=color)

            tx = Text(
                text,
                font=FONT,
                font_size=21
            )

            g = VGroup(f, tx)
            g.arrange(DOWN, buff=0.3)

            groups.add(g)

        groups.arrange(
            RIGHT,
            buff=0.8
        )

        groups.move_to(ORIGIN)

        self.play(
            LaggedStart(
                *[FadeIn(x) for x in groups],
                lag_ratio=0.25
            )
        )

        equation = formula("L = λᵢLinv + λᵥLvar + λ𝚌Lcov", size=30)

        equation.to_edge(DOWN)

        self.play(Write(equation))

        self.wait(2)


# ============================================================
# SCENE 13
# ============================================================

class MaskingScene(Scene):

    def construct(self):

        t = title(
            "Masking creates the prediction problem"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        full = random_binary_grid(
            16,
            16,
            0.4,
            20,
            0.11
        )

        full.move_to(LEFT * 3)

        self.play(FadeIn(full))

        hole = Rectangle(
            width=0.85,
            height=1.3,
            stroke_color=RED,
            stroke_width=3,
            fill_color=RED,
            fill_opacity=0.25
        )

        hole.move_to(
            full.get_center() + RIGHT * 0.2
        )

        self.play(Create(hole))

        label = Text(
            "masked region",
            font=FONT,
            font_size=24,
            color=RED
        )

        label.next_to(hole, DOWN)

        self.play(Write(label))

        predict = box(
            "Predict what\nis hidden",
            width=3,
            height=1.3,
            color=GREEN
        )

        predict.move_to(RIGHT * 3)

        self.play(
            GrowArrow(
                arrow(
                    full.get_right(),
                    predict.get_left(),
                    GREEN
                )
            ),
            FadeIn(predict)
        )

        note = Text(
            "The model cannot simply copy the answer.",
            font=FONT,
            font_size=28,
            color=YELLOW
        )

        note.to_edge(DOWN)

        self.play(Write(note))

        self.wait(2)


# ============================================================
# SCENE 14
# ============================================================

class TrainingStepScene(Scene):

    def construct(self):

        t = title(
            "One training iteration"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        steps = [
            ("1", "sample data"),
            ("2", "mask / condition"),
            ("3", "student forward"),
            ("4", "target forward"),
            ("5", "latent losses"),
            ("6", "physics loss"),
            ("7", "backprop"),
            ("8", "EMA update")
        ]

        groups = VGroup()

        for i, (num, txt) in enumerate(steps):

            n = Circle(
                radius=0.28,
                stroke_color=BLUE,
                stroke_width=2
            )

            nt = Text(
                num,
                font=FONT,
                font_size=20
            )

            nt.move_to(n.get_center())

            label = Text(
                txt,
                font=FONT,
                font_size=22
            )

            g = VGroup(n, nt, label)
            label.next_to(n, RIGHT, buff=0.25)

            groups.add(g)

        groups.arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.22
        )

        groups.scale(0.9)

        self.play(
            LaggedStart(
                *[
                    FadeIn(x, shift=RIGHT)
                    for x in groups
                ],
                lag_ratio=0.12
            )
        )

        self.wait(2)


# ============================================================
# SCENE 15
# ============================================================

class PhysicsScene(Scene):

    def construct(self):

        t = title(
            "The physics loop asks a harder question"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        latent = box(
            "predicted\nlatent",
            width=2.5,
            color=PURPLE
        )

        latent.move_to(LEFT * 5)

        decoder = box(
            "Geometry\nDecoder",
            width=2.5,
            color=BLUE
        )

        decoder.move_to(LEFT * 2)

        geom = box(
            "Predicted\nGeometry",
            width=2.8,
            color=GREEN
        )

        geom.move_to(RIGHT * 1)

        surrogate = box(
            "Frozen EM\nSurrogate",
            width=3,
            color=RED
        )

        surrogate.move_to(RIGHT * 4.5)

        target = spectrum_curve(
            width=3,
            height=1.7,
            seed=5
        )

        target.move_to(
            RIGHT * 4.5 + DOWN * 2
        )

        tl = Text(
            "target spectrum",
            font=FONT,
            font_size=21,
            color=CYAN
        )

        tl.next_to(target, DOWN)

        for a, b, c in [
            (latent, decoder, BLUE),
            (decoder, geom, GREEN),
            (geom, surrogate, RED)
        ]:

            self.play(
                GrowArrow(
                    arrow(
                        a.get_right(),
                        b.get_left(),
                        c
                    )
                ),
                run_time=0.5
            )

        self.play(
            FadeIn(latent),
            FadeIn(decoder),
            FadeIn(geom),
            FadeIn(surrogate)
        )

        self.play(
            FadeIn(target),
            Write(tl)
        )

        loss = formula("Lphys = SmoothL1(Ŝ, S)", size=30)

        loss.to_edge(DOWN)

        self.play(Write(loss))

        self.wait(2)


# ============================================================
# SCENE 16
# ============================================================

class STEScene(Scene):

    def construct(self):

        t = title(
            "Why do we need a Straight-Through Estimator?"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        soft = box(
            "soft occupancy\n0.37, 0.62, ...",
            width=3.2,
            color=YELLOW,
            fs=21
        )

        hard = box(
            "hard occupancy\n0 or 1",
            width=3,
            color=GREEN,
            fs=23
        )

        surrogate = box(
            "EM surrogate",
            width=2.8,
            color=RED
        )

        soft.move_to(LEFT * 4)
        hard.move_to(LEFT * 0.5)
        surrogate.move_to(RIGHT * 3.5)

        self.play(
            FadeIn(soft),
            FadeIn(hard),
            FadeIn(surrogate)
        )

        self.play(
            GrowArrow(
                arrow(
                    soft.get_right(),
                    hard.get_left(),
                    YELLOW
                )
            )
        )

        self.play(
            GrowArrow(
                arrow(
                    hard.get_right(),
                    surrogate.get_left(),
                    RED
                )
            )
        )

        equation = formula("∂yhard / ∂x ≈ ∂ysoft / ∂x", size=38, color=GREEN)

        equation.to_edge(DOWN)

        self.play(Write(equation))

        note = Text(
            "Forward: physical binary geometry",
            font=FONT,
            font_size=23,
            color=WHITE
        )

        note.next_to(hard, DOWN)

        self.play(Write(note))

        self.wait(2)


# ============================================================
# SCENE 17
# ============================================================

class InferenceScene(Scene):

    def construct(self):

        t = title(
            "Inference: spectrum → geometry"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        spectrum = spectrum_curve(
            width=3.5,
            height=2,
            seed=22
        )

        spectrum.move_to(LEFT * 4)

        encoder = box(
            "Spectrum\nEncoder",
            width=2.5,
            color=CYAN
        )

        encoder.move_to(LEFT * 1)

        predictor = box(
            "JEPA\nPredictor",
            width=2.7,
            color=GREEN
        )

        predictor.move_to(RIGHT * 2)

        geom = random_binary_grid(
            16,
            16,
            0.4,
            31,
            0.09
        )

        geom.move_to(RIGHT * 5)

        label = Text(
            "generated geometry",
            font=FONT,
            font_size=23,
            color=GREEN
        )

        label.next_to(geom, DOWN)

        self.play(
            FadeIn(spectrum),
            FadeIn(encoder),
            FadeIn(predictor),
            FadeIn(geom),
            Write(label)
        )

        self.play(
            GrowArrow(
                arrow(
                    spectrum.get_right(),
                    encoder.get_left(),
                    CYAN
                )
            )
        )

        self.play(
            GrowArrow(
                arrow(
                    encoder.get_right(),
                    predictor.get_left(),
                    GREEN
                )
            )
        )

        self.play(
            GrowArrow(
                arrow(
                    predictor.get_right(),
                    geom.get_left(),
                    GREEN
                )
            )
        )

        self.wait(2)

        note = Text(
            "No gradient descent is required at inference.",
            font=FONT,
            font_size=28,
            color=YELLOW
        )

        note.to_edge(DOWN)

        self.play(Write(note))

        self.wait(2)


# ============================================================
# SCENE 18
# ============================================================

class ShuffledTestScene(Scene):

    def construct(self):

        t = title(
            "How do we prove the model actually uses the spectrum?"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        real = box(
            "real spectrum",
            width=2.8,
            color=GREEN
        )

        shuffled = box(
            "shuffled spectrum",
            width=2.8,
            color=RED
        )

        real.move_to(LEFT * 3)
        shuffled.move_to(RIGHT * 3)

        self.play(
            FadeIn(real),
            FadeIn(shuffled)
        )

        model = box(
            "same model",
            width=2.8,
            color=BLUE
        )

        model.move_to(DOWN * 1.7)

        self.play(
            GrowArrow(
                arrow(
                    real.get_bottom(),
                    model.get_top(),
                    GREEN
                )
            ),
            GrowArrow(
                arrow(
                    shuffled.get_bottom(),
                    model.get_top(),
                    RED
                )
            )
        )

        self.play(FadeIn(model))

        result = formula("Ereal < Eshuffled", size=45, color=GREEN)

        result.to_edge(DOWN)

        self.play(Write(result))

        self.wait(2)

        explanation = Text(
            "If scrambling the spectrum hurts performance,\n"
            "the spectrum must contain information the model uses.",
            font=FONT,
            font_size=24,
            color=WHITE
        )

        explanation.move_to(
            DOWN * 1.3
        )

        self.play(
            Write(explanation)
        )

        self.wait(2)


# ============================================================
# SCENE 19
# ============================================================

class ResultsScene(Scene):

    def construct(self):

        t = title(
            "70k-step × 3-seed result"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        numbers = [
            ("MAE", "0.0490 ± 0.0013", BLUE),
            ("AAE", "29.51 ± 0.74", CYAN),
            ("Scenario A", "0.9967", GREEN),
            ("Scenario B", "0.9772", GREEN),
            ("Scenario C", "≈ 0.930", GREEN)
        ]

        cards = VGroup()

        for name, value, color in numbers:

            b = RoundedRectangle(
                width=3.1,
                height=1.15,
                corner_radius=0.12,
                stroke_color=color,
                stroke_width=2
            )

            n = Text(
                name,
                font=FONT,
                font_size=21,
                color=GREY
            )

            v = Text(
                value,
                font=FONT,
                font_size=29,
                color=WHITE,
                weight=BOLD
            )

            g = VGroup(n, v)
            g.arrange(DOWN, buff=0.15)

            g.move_to(b.get_center())

            cards.add(
                VGroup(b, g)
            )

        cards.arrange(
            RIGHT,
            buff=0.25
        )

        cards.scale(0.82)

        self.play(
            LaggedStart(
                *[
                    FadeIn(x, shift=UP)
                    for x in cards
                ],
                lag_ratio=0.15
            )
        )

        self.wait(2)

        note = Text(
            "Spectrum-conditioned geometry recovery is strong.",
            font=FONT,
            font_size=29,
            color=GREEN
        )

        note.to_edge(DOWN)

        self.play(Write(note))

        self.wait(2)


# ============================================================
# SCENE 20
# ============================================================

class AblationScene(Scene):

    def construct(self):

        t = title(
            "Is JEPA actually doing something?"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        jepa = box(
            "JEPA objective\nMAE = 0.049",
            width=3.5,
            height=1.5,
            color=GREEN,
            fs=25
        )

        conventional = box(
            "Conventional objective\nMAE = 0.092",
            width=4,
            height=1.5,
            color=RED,
            fs=24
        )

        jepa.move_to(LEFT * 2.7)
        conventional.move_to(RIGHT * 2.7)

        self.play(
            FadeIn(jepa),
            FadeIn(conventional)
        )

        ratio = formula("0.092 / 0.049 ≈ 1.9×", size=45, color=YELLOW)

        ratio.to_edge(DOWN)

        self.play(Write(ratio))

        self.wait(2)

        note = Text(
            "Under the controlled ablation, the JEPA objective wins.",
            font=FONT,
            font_size=28,
            color=GREEN
        )

        note.next_to(ratio, UP)

        self.play(Write(note))

        self.wait(2)


# ============================================================
# SCENE 21
# ============================================================

class ScalarFailureScene(Scene):

    def construct(self):

        t = title(
            "But one part of the architecture is still broken"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        scalar = box(
            "known scalar\nl,h,r",
            width=3,
            color=YELLOW
        )

        scalar.move_to(LEFT * 4)

        model = box(
            "model",
            width=2.5,
            color=BLUE
        )

        model.move_to(ORIGIN)

        prediction1 = box(
            "prediction A",
            width=2.8,
            color=GREEN
        )

        prediction2 = box(
            "prediction B",
            width=2.8,
            color=GREEN
        )

        prediction1.move_to(RIGHT * 3.5 + UP * 1)
        prediction2.move_to(RIGHT * 3.5 + DOWN * 1)

        self.play(
            FadeIn(scalar),
            FadeIn(model)
        )

        self.play(
            GrowArrow(
                arrow(
                    scalar.get_right(),
                    model.get_left(),
                    YELLOW
                )
            )
        )

        self.play(
            GrowArrow(
                arrow(
                    model.get_right(),
                    prediction1.get_left(),
                    GREEN
                )
            ),
            GrowArrow(
                arrow(
                    model.get_right(),
                    prediction2.get_left(),
                    GREEN
                )
            )
        )

        self.play(
            FadeIn(prediction1),
            FadeIn(prediction2)
        )

        chance = formula("scalar dependence ≈ 0.5", size=38, color=RED)

        chance.to_edge(DOWN)

        self.play(Write(chance))

        self.wait(2)

        note = Text(
            "More training did not solve this.",
            font=FONT,
            font_size=29,
            color=RED
        )

        note.next_to(chance, UP)

        self.play(Write(note))

        self.wait(2)


# ============================================================
# SCENE 22
# ============================================================

class RetrievalScene(Scene):

    def construct(self):

        t = title(
            "And there is a surprisingly strong baseline"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        target = spectrum_curve(
            width=3.2,
            height=1.8,
            seed=42
        )

        target.move_to(LEFT * 4)

        label = Text(
            "target spectrum",
            font=FONT,
            font_size=23
        )

        label.next_to(target, DOWN)

        self.play(
            FadeIn(target),
            Write(label)
        )

        database = VGroup()

        for i in range(8):

            c = Circle(
                radius=0.18,
                stroke_color=CYAN,
                fill_color=CYAN,
                fill_opacity=0.5
            )

            database.add(c)

        database.arrange(
            DOWN,
            buff=0.18
        )

        database.move_to(RIGHT * 0.5)

        self.play(FadeIn(database))

        self.play(
            GrowArrow(
                arrow(
                    target.get_right(),
                    database.get_left(),
                    CYAN
                )
            )
        )

        best = box(
            "nearest training\nspectrum",
            width=3.2,
            height=1.3,
            color=GREEN
        )

        best.move_to(RIGHT * 4)

        self.play(
            GrowArrow(
                arrow(
                    database.get_right(),
                    best.get_left(),
                    GREEN
                )
            ),
            FadeIn(best)
        )

        result = formula(
            "MAENN ≈ 0.0296 < MAEJEPA ≈ 0.0490",
            size=33,
            color=YELLOW,
        )

        result.to_edge(DOWN)

        self.play(Write(result))

        self.wait(2)


# ============================================================
# SCENE 23
# ============================================================

class ConclusionScene(Scene):

    def construct(self):

        t = title(
            "So what did we actually learn?"
        )

        self.play(Write(t))
        self.play(t.animate.to_edge(UP))

        statements = [
            (
                "1",
                "JEPA learns a useful latent prediction problem.",
                GREEN
            ),
            (
                "2",
                "The spectrum genuinely influences geometry.",
                CYAN
            ),
            (
                "3",
                "Physics gives a differentiable consistency constraint.",
                PURPLE
            ),
            (
                "4",
                "The scalar pathway remains unresolved.",
                RED
            ),
            (
                "5",
                "Nearest-neighbour retrieval remains a serious baseline.",
                YELLOW
            )
        ]

        groups = VGroup()

        for n, text, color in statements:

            circ = Circle(
                radius=0.27,
                stroke_color=color,
                stroke_width=2
            )

            num = Text(
                n,
                font=FONT,
                font_size=18
            )

            num.move_to(circ.get_center())

            tx = Text(
                text,
                font=FONT,
                font_size=25,
                color=WHITE
            )

            tx.next_to(
                circ,
                RIGHT,
                buff=0.25
            )

            groups.add(
                VGroup(circ, num, tx)
            )

        groups.arrange(
            DOWN,
            aligned_edge=LEFT,
            buff=0.32
        )

        groups.move_to(ORIGIN)

        self.play(
            LaggedStart(
                *[
                    FadeIn(x, shift=RIGHT)
                    for x in groups
                ],
                lag_ratio=0.2
            )
        )

        self.wait(3)

        final = Text(
            "The interesting question is no longer\n"
            "\"Can JEPA reconstruct geometry?\"\n\n"
            "It is:\n"
            "\"Can we make the learned representation\n"
            "faithfully encode every physical degree of freedom?\"",
            font=FONT,
            font_size=28,
            line_spacing=1.2,
            color=WHITE
        )

        final.move_to(ORIGIN)

        self.play(
            FadeOut(groups),
            FadeOut(t),
            FadeIn(final)
        )

        self.wait(4)


# ============================================================
# MASTER SCENE
# ============================================================

class JEPAFullPresentation(Scene):

    def construct(self):

        scenes = [
            ProblemScene,
            DatasetScene,
            FactorizationScene,
            LatentScene,
            ArchitectureScene,
            SpectrumEncoderScene,
            OccupancyEncoderScene,
            ScalarFiLMScene,
            FusionScene,
            JEPAScene,
            EMAScene,
            VICRegScene,
            MaskingScene,
            TrainingStepScene,
            PhysicsScene,
            STEScene,
            InferenceScene,
            ShuffledTestScene,
            ResultsScene,
            AblationScene,
            ScalarFailureScene,
            RetrievalScene,
            ConclusionScene,
        ]

        # Reuse each standalone scene's construction logic so the documented
        # full-presentation command produces one continuous video.  Clearing
        # between scenes prevents the previous scene's mobjects from leaking
        # into the next one.
        for index, scene_class in enumerate(scenes):
            scene_class.construct(self)
            if index < len(scenes) - 1:
                self.clear()
