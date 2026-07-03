from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

from manim import *


ROOT = Path(__file__).resolve().parent
AUDIO_MANIFEST = ROOT / "audio" / "manifest.json"
RAG_SIM_PATH = ROOT / "rag_similarity.json"
FONT = "Arial"

BG = "#0B0F14"
PANEL = "#121A22"
PANEL_2 = "#17212B"
TEXT = "#EEF4F8"
MUTED = "#91A2B2"
BLUE = "#58A6FF"
CYAN = "#39D4D8"
GREEN = "#69E28D"
YELLOW = "#FFD166"
ORANGE = "#FF9F4A"
RED = "#FF6B6B"
PINK = "#F778BA"
PURPLE = "#B392F0"


def load_manifest() -> dict[str, dict]:
    if not AUDIO_MANIFEST.exists():
        return {}
    data = json.loads(AUDIO_MANIFEST.read_text())
    return {row["key"]: row for row in data}


def load_rag_similarity() -> dict[str, float]:
    if not RAG_SIM_PATH.exists():
        return {"milk_tense": 0.8917, "unrelated": 0.1621}
    rows = json.loads(RAG_SIM_PATH.read_text())
    return {row["name"]: float(row["cosine"]) for row in rows}


def wrapped(text: str, width: int = 44) -> str:
    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        lines.extend(textwrap.wrap(para, width=width))
    return "\n".join(lines)


def label(text: str, size: int = 28, color: str = TEXT, weight: str = NORMAL) -> Text:
    return Text(text, font=FONT, font_size=size, color=color, weight=weight, disable_ligatures=True)


def paragraph(text: str, size: int = 24, color: str = TEXT, width: int = 46, line_spacing: float = 0.75) -> Text:
    return Text(
        wrapped(text, width),
        font=FONT,
        font_size=size,
        color=color,
        line_spacing=line_spacing,
        disable_ligatures=True,
    )


def words(items: list[str], size: int = 18, color: str = TEXT, weight: str = NORMAL, buff: float = 0.08) -> VGroup:
    """Render short phrases with explicit word spacing for small on-screen text."""
    return VGroup(*[label(item, size, color, weight) for item in items]).arrange(RIGHT, buff=buff)


def panel(width: float, height: float, color: str = PANEL, stroke: str = "#2A3846") -> RoundedRectangle:
    return RoundedRectangle(
        width=width,
        height=height,
        corner_radius=0.18,
        fill_color=color,
        fill_opacity=1,
        stroke_color=stroke,
        stroke_width=1.5,
    )


def memory_card(title: str, subtitle: str, color: str, source: str = "") -> VGroup:
    box = panel(3.35, 1.62, PANEL, color)
    title_m = label(title, 24, TEXT, BOLD).move_to(box.get_top() + DOWN * 0.35)
    sub_m = paragraph(subtitle, 14, MUTED, 25).next_to(title_m, DOWN, buff=0.12)
    sub_m.set_width(2.85)
    src = label(source, 10, color).move_to(box.get_bottom() + UP * 0.18)
    return VGroup(box, title_m, sub_m, src)


def tiny_doc(text: str, color: str = "#DDE7EF") -> VGroup:
    rect = RoundedRectangle(width=0.9, height=1.12, corner_radius=0.04, fill_color=color, fill_opacity=1, stroke_width=0)
    corner = Polygon(
        rect.get_corner(UR),
        rect.get_corner(UR) + LEFT * 0.22,
        rect.get_corner(UR) + DOWN * 0.22,
        fill_color="#B7C6D4",
        fill_opacity=1,
        stroke_width=0,
    )
    lines = VGroup(*[Line(LEFT * 0.22, RIGHT * 0.22, color="#687887", stroke_width=2) for _ in range(4)]).arrange(DOWN, buff=0.10)
    lines.move_to(rect.get_center() + DOWN * 0.08)
    cap = label(text, 10, BG).next_to(rect, DOWN, buff=0.05)
    return VGroup(rect, corner, lines, cap)


def arrow_between(a: Mobject, b: Mobject, color: str = MUTED, buff: float = 0.15) -> Arrow:
    return Arrow(a.get_right(), b.get_left(), buff=buff, color=color, stroke_width=3, max_tip_length_to_length_ratio=0.12)


class MemoryStrategyCards(Scene):
    def setup(self):
        self.manifest = load_manifest()
        self.camera.background_color = BG

    def audio(self, key: str, fallback: float = 12.0) -> float:
        row = self.manifest.get(key)
        if not row:
            return fallback
        self.add_sound(str(ROOT / row["file"]))
        return float(row["duration"])

    def end_segment(self, total_duration: float, used: float) -> None:
        self.wait(max(total_duration - used, 0.25))

    def title_bar(self, section: str, source: str | None = None) -> VGroup:
        top = VGroup()
        title = label(section, 25, TEXT, BOLD).to_corner(UL).shift(DOWN * 0.08)
        rule = Line(LEFT * 6.8, RIGHT * 6.8, color="#22313F", stroke_width=1).to_edge(UP).shift(DOWN * 0.62)
        top.add(title, rule)
        if source:
            src = label(source, 13, MUTED).to_corner(DL).shift(UP * 0.08)
            top.add(src)
        return top

    def construct(self):
        self.intro()
        self.rag()
        self.temporal_kg_and_logs()
        self.observation_logs()
        self.timeline_reconstruction()
        self.offline_consolidation()
        self.recursive_read_time()
        self.learned_policy()
        self.learned_chunking()
        self.latent_cache_memory()
        self.synthesis()
        self.references()

    def intro(self):
        dur = self.audio("00_hook", 25)
        title = label("Memory Strategy Cards", 54, TEXT, BOLD)
        subtitle = paragraph(
            "Nine ways to give an AI agent memory, and why they are not all solving the same problem.",
            25,
            MUTED,
            58,
        ).next_to(title, DOWN, buff=0.25)
        cards = VGroup(
            memory_card("RAG", "retrieve text", BLUE, "Lewis 2020"),
            memory_card("KG", "structured state", GREEN, "Zep"),
            memory_card("Timeline", "replay events", YELLOW, "event sourcing"),
            memory_card("Consolidation", "rewrite later", PURPLE, "Auto-Dreamer"),
            memory_card("RLM", "inspect at read time", CYAN, "RLM"),
            memory_card("RL Policy", "learn actions", RED, "Memory-R1"),
            memory_card("Chunking", "learn boundaries", ORANGE, "H-Net"),
            memory_card("KV Cache", "store model state", PINK, "Cartridges"),
        ).arrange_in_grid(rows=2, cols=4, buff=0.18)
        cards.scale(0.80).next_to(subtitle, DOWN, buff=0.45)

        self.play(FadeIn(title, shift=DOWN), run_time=1.1)
        self.play(FadeIn(subtitle, shift=DOWN), run_time=0.8)
        self.play(LaggedStart(*(FadeIn(c, shift=UP) for c in cards), lag_ratio=0.08), run_time=2.5)

        axes = VGroup(
            Arrow(LEFT * 5.4, RIGHT * 5.4, color="#2E4053", stroke_width=3),
            Arrow(DOWN * 2.2, UP * 2.2, color="#2E4053", stroke_width=3),
        )
        x1 = label("compute at write time", 15, MUTED).move_to(LEFT * 4.0 + DOWN * 2.9)
        x2 = label("compute at read time", 15, MUTED).move_to(RIGHT * 4.0 + DOWN * 2.9)
        y1 = label("text / symbols", 15, MUTED).move_to(UP * 2.65 + LEFT * 1.0)
        y2 = label("learned / latent", 15, MUTED).move_to(DOWN * 2.65 + LEFT * 1.0)
        map_group = VGroup(axes, x1, x2, y1, y2).shift(DOWN * 0.15)
        self.play(FadeOut(title), FadeOut(subtitle), cards.animate.scale(0.50).shift(UP * 2.55), run_time=1.2)
        self.play(Create(axes), FadeIn(x1), FadeIn(x2), FadeIn(y1), FadeIn(y2), run_time=1.2)
        self.play(
            cards[0].animate.move_to(LEFT * 4.45 + UP * 0.65),
            cards[1].animate.move_to(LEFT * 2.45 + UP * 1.45),
            cards[2].animate.move_to(LEFT * 0.35 + UP * 1.35),
            cards[3].animate.move_to(RIGHT * 0.45 + DOWN * 0.72),
            cards[4].animate.move_to(RIGHT * 2.05 + UP * 1.05),
            cards[5].animate.move_to(RIGHT * 2.55 + DOWN * 1.25),
            cards[6].animate.move_to(LEFT * 3.65 + DOWN * 1.35),
            cards[7].animate.move_to(RIGHT * 4.45 + DOWN * 1.35),
            run_time=2.2,
        )
        self.end_segment(dur, 9.0)
        self.play(FadeOut(cards), FadeOut(map_group), run_time=0.7)

    def rag(self):
        dur = self.audio("01_rag", 31)
        sims = load_rag_similarity()
        top = self.title_bar("1. RAG: retrieve text", "RAG: Lewis et al. 2020")
        docs = VGroup(tiny_doc("doc A"), tiny_doc("doc B"), tiny_doc("doc C")).arrange(DOWN, buff=0.15).move_to(LEFT * 5.1)
        chunks = VGroup(*[RoundedRectangle(width=1.1, height=0.34, corner_radius=0.05, fill_color=BLUE, fill_opacity=0.75, stroke_width=0) for _ in range(7)]).arrange(DOWN, buff=0.08).move_to(LEFT * 2.8)
        vec = Circle(radius=1.05, color=BLUE, fill_color="#102A43", fill_opacity=1).move_to(LEFT * 0.55)
        dots = VGroup(*[Dot(vec.get_center() + np.array([math.cos(i) * 0.65, math.sin(i * 1.7) * 0.45, 0]), radius=0.035, color=CYAN) for i in np.linspace(0, TAU, 18)])
        topk = VGroup(*[RoundedRectangle(width=1.35, height=0.44, corner_radius=0.05, fill_color=CYAN, fill_opacity=0.8, stroke_width=0) for _ in range(3)]).arrange(DOWN, buff=0.13).move_to(RIGHT * 1.65)
        llm = panel(1.9, 1.35, "#171D2B", PINK).move_to(RIGHT * 4.5)
        llm_text = label("LLM", 34, TEXT, BOLD).move_to(llm)
        q = panel(2.15, 0.55, "#24161A", RED).move_to(DOWN * 2.65 + LEFT * 2.2)
        q_text = words(["query:", "current", "status?"], 17, TEXT, buff=0.09).move_to(q)

        self.play(FadeIn(top), run_time=0.5)
        self.play(LaggedStart(FadeIn(docs), FadeIn(chunks), Create(vec), FadeIn(dots), FadeIn(topk), FadeIn(llm), FadeIn(llm_text), lag_ratio=0.14), run_time=2.4)
        arrows = VGroup(arrow_between(docs, chunks, BLUE), arrow_between(chunks, vec, BLUE), arrow_between(vec, topk, CYAN), arrow_between(topk, llm, PINK))
        self.play(Create(arrows), FadeIn(q), FadeIn(q_text), run_time=1.4)

        sim_box = panel(4.9, 1.65, "#151E2A", YELLOW).move_to(UP * 2.0 + RIGHT * 1.05)
        sim_title = label("Embedding cosine check", 20, TEXT, BOLD).move_to(sim_box.get_top() + DOWN * 0.28)
        row1 = VGroup(
            words(["will", "buy", "milk"], 16, TEXT, buff=0.16),
            label("vs", 16, MUTED),
            words(["bought", "milk"], 16, TEXT, buff=0.16),
            label("=", 16, MUTED),
            label(f"{sims.get('milk_tense', 0.8917):.3f}", 16, TEXT),
        ).arrange(RIGHT, buff=0.16).move_to(sim_box.get_center() + UP * 0.12)
        row2 = VGroup(
            words(["buy", "milk"], 16, MUTED, buff=0.16),
            label("vs", 16, MUTED),
            words(["broken", "laptop"], 16, MUTED, buff=0.16),
            label("=", 16, MUTED),
            label(f"{sims.get('unrelated', 0.1621):.3f}", 16, MUTED),
        ).arrange(RIGHT, buff=0.16).move_to(sim_box.get_center() + DOWN * 0.38)
        bar1_bg = Rectangle(width=2.0, height=0.08, fill_color="#2A3846", fill_opacity=1, stroke_width=0).next_to(row1, DOWN, buff=0.08).align_to(row1, LEFT)
        bar1 = Rectangle(width=2.0 * sims.get("milk_tense", 0.8917), height=0.08, fill_color=YELLOW, fill_opacity=1, stroke_width=0).align_to(bar1_bg, LEFT).move_to(bar1_bg.get_center(), aligned_edge=LEFT)
        bar2_bg = Rectangle(width=2.0, height=0.08, fill_color="#2A3846", fill_opacity=1, stroke_width=0).next_to(row2, DOWN, buff=0.08).align_to(row2, LEFT)
        bar2 = Rectangle(width=2.0 * sims.get("unrelated", 0.1621), height=0.08, fill_color=MUTED, fill_opacity=1, stroke_width=0).align_to(bar2_bg, LEFT).move_to(bar2_bg.get_center(), aligned_edge=LEFT)
        warn = paragraph("High similarity can still hide different world states.", 23, YELLOW, 48).to_edge(DOWN).shift(UP * 1.25)
        self.play(FadeIn(sim_box), FadeIn(sim_title), FadeIn(row1), FadeIn(row2), FadeIn(bar1_bg), FadeIn(bar2_bg), run_time=1.2)
        self.play(GrowFromEdge(bar1, LEFT), GrowFromEdge(bar2, LEFT), FadeIn(warn), run_time=2.0)
        self.end_segment(dur, 7.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def temporal_kg_and_logs(self):
        dur = self.audio("02_temporal_knowledge_graphs", 31)
        top = self.title_bar("2. Structured memory: temporal KGs and observation logs", "Zep / Graphiti, Mastra OM")
        self.play(FadeIn(top), run_time=0.5)
        nodes = {
            "User": LEFT * 4.2 + UP * 1.2,
            "Boston": LEFT * 2.1 + UP * 1.9,
            "NYC": LEFT * 2.1 + UP * 0.5,
            "Job": LEFT * 2.1 + DOWN * 0.9,
            "Preference": LEFT * 4.2 + DOWN * 1.0,
        }
        graph_nodes = VGroup()
        for name, pos in nodes.items():
            c = Circle(radius=0.42, fill_color="#172A3A", fill_opacity=1, color=GREEN).move_to(pos)
            t = label(name, 13, TEXT).move_to(c)
            graph_nodes.add(VGroup(c, t))
        edges = VGroup(
            Line(nodes["User"], nodes["Boston"], color=GREEN),
            Line(nodes["User"], nodes["NYC"], color=GREEN),
            Line(nodes["User"], nodes["Job"], color=GREEN),
            Line(nodes["User"], nodes["Preference"], color=GREEN),
        )
        edge_labels = VGroup(
            label("lived_in until Aug", 12, MUTED).move_to(LEFT * 3.1 + UP * 1.65),
            label("lived_in from Aug", 12, MUTED).move_to(LEFT * 3.0 + UP * 0.68),
            label("works_at now", 12, MUTED).move_to(LEFT * 3.0 + DOWN * 0.55),
        )
        log_box = panel(4.8, 3.6, PANEL_2, CYAN).move_to(RIGHT * 3.2)
        log_title = label("Observation log", 26, TEXT, BOLD).move_to(log_box.get_top() + DOWN * 0.35)
        obs = VGroup(
            paragraph("• user lived in Boston before Aug", 18, TEXT, 34),
            paragraph("• user moved to NYC in Aug", 18, TEXT, 34),
            paragraph("• newer facts supersede older facts", 18, TEXT, 34),
            paragraph("• compact view replaces raw history", 18, TEXT, 34),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18).next_to(log_title, DOWN, buff=0.35)
        caveat = paragraph("Clean structure is only useful if extraction is right.", 24, YELLOW, 44).to_edge(DOWN).shift(UP * 0.65)
        self.play(Create(edges), FadeIn(graph_nodes), FadeIn(edge_labels), run_time=2.0)
        self.play(FadeIn(log_box), FadeIn(log_title), LaggedStart(*(FadeIn(o, shift=RIGHT) for o in obs), lag_ratio=0.18), run_time=2.5)
        self.play(FadeIn(caveat), Circumscribe(graph_nodes[0], color=YELLOW), run_time=1.8)
        self.end_segment(dur, 6.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def timeline_reconstruction(self):
        dur = self.audio("04_timeline_reconstruction", 28)
        top = self.title_bar("4. Timeline reconstruction: answer from state at time T", "Event sourcing, temporal databases, Zep")
        self.play(FadeIn(top), run_time=0.5)
        axis = Line(LEFT * 5.7, RIGHT * 5.7, color="#34495E", stroke_width=4).shift(DOWN * 0.2)
        months = ["Jan", "May", "Aug", "Now"]
        xs = [-5.1, -2.0, 1.2, 5.1]
        ticks = VGroup()
        for m, x in zip(months, xs):
            tick = Line(UP * 0.18, DOWN * 0.18, color=MUTED).move_to(np.array([x, -0.2, 0]))
            txt = label(m, 18, MUTED).next_to(tick, DOWN, buff=0.18)
            ticks.add(VGroup(tick, txt))
        e1 = panel(2.1, 0.68, "#13231A", GREEN).move_to(np.array([-4.2, 1.25, 0]))
        e1t = label("lives in Boston", 18, TEXT).move_to(e1)
        e2 = panel(2.1, 0.68, "#1C2435", BLUE).move_to(np.array([1.2, 1.25, 0]))
        e2t = label("moves to NYC", 18, TEXT).move_to(e2)
        question = panel(3.4, 0.75, "#251B12", ORANGE).move_to(DOWN * 2.2)
        qt = label("Where did I live in May?", 22, TEXT, BOLD).move_to(question)
        cursor = DashedLine(UP * 2.0, DOWN * 1.25, color=YELLOW, stroke_width=3).move_to(np.array([-2.0, 0.25, 0]))
        answer = panel(2.8, 0.85, "#14251A", GREEN).move_to(RIGHT * 3.5 + DOWN * 2.2)
        ans_t = label("Boston", 32, TEXT, BOLD).move_to(answer)
        replay = VGroup(label("sort", 17, MUTED), Arrow(LEFT * 0.35, RIGHT * 0.35, color=MUTED), label("replay to T", 17, MUTED), Arrow(LEFT * 0.35, RIGHT * 0.35, color=MUTED), label("answer", 17, MUTED)).arrange(RIGHT, buff=0.15).move_to(UP * 2.45)
        self.play(Create(axis), FadeIn(ticks), run_time=1.0)
        self.play(FadeIn(e1), FadeIn(e1t), run_time=0.8)
        self.play(FadeIn(e2), FadeIn(e2t), run_time=0.8)
        self.play(FadeIn(question), FadeIn(qt), Create(cursor), run_time=1.1)
        self.play(LaggedStart(*(FadeIn(x, shift=DOWN) for x in replay), lag_ratio=0.12), run_time=1.4)
        self.play(FadeIn(answer, shift=LEFT), FadeIn(ans_t, shift=LEFT), Circumscribe(e1, color=GREEN), run_time=1.5)
        self.end_segment(dur, 6.6)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def observation_logs(self):
        dur = self.audio("03_observation_logs", 28)
        top = self.title_bar("3. Observation logs: compact the running history", "Mastra Observational Memory")
        self.play(FadeIn(top), run_time=0.5)
        raw = VGroup(*[tiny_doc(f"chat {i}") for i in range(1, 7)]).arrange(RIGHT, buff=0.10).scale(0.62).move_to(LEFT * 4.25 + UP * 1.55)
        observer = panel(2.15, 1.1, "#132A35", CYAN).move_to(LEFT * 1.25 + UP * 1.55)
        observer_t = label("Observer", 26, TEXT, BOLD).move_to(observer)
        reflector = panel(2.15, 1.1, "#171629", PURPLE).move_to(RIGHT * 1.45 + UP * 1.55)
        reflector_t = label("Reflector", 26, TEXT, BOLD).move_to(reflector)
        log = panel(5.4, 2.5, PANEL_2, GREEN).move_to(DOWN * 1.05)
        log_title = label("Dense observation log", 25, TEXT, BOLD).move_to(log.get_top() + DOWN * 0.35)
        bullets = VGroup(
            paragraph("• stable facts and recurring context", 18, TEXT, 44),
            paragraph("• compact enough to fit in the prompt", 18, TEXT, 44),
            paragraph("• powerful when the background pass is right", 18, TEXT, 44),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.14).next_to(log_title, DOWN, buff=0.28)
        warning = paragraph(
            "The compact log replaces detail. If the observer misses it, the future agent may never see it.",
            23,
            YELLOW,
            58,
        ).to_edge(DOWN).shift(UP * 0.55)
        self.play(FadeIn(raw), run_time=0.9)
        self.play(Create(Arrow(raw.get_right(), observer.get_left(), color=CYAN, stroke_width=4)), FadeIn(observer), FadeIn(observer_t), run_time=1.2)
        self.play(Create(Arrow(observer.get_right(), reflector.get_left(), color=PURPLE, stroke_width=4)), FadeIn(reflector), FadeIn(reflector_t), run_time=1.2)
        self.play(
            Create(Arrow(reflector.get_bottom(), log.get_top(), color=GREEN, stroke_width=4)),
            FadeIn(log),
            FadeIn(log_title),
            LaggedStart(*(FadeIn(b, shift=RIGHT) for b in bullets), lag_ratio=0.15),
            run_time=2.1,
        )
        self.play(FadeIn(warning), run_time=1.0)
        self.end_segment(dur, 6.4)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def offline_consolidation(self):
        dur = self.audio("05_offline_consolidation", 35)
        top = self.title_bar("5. Offline consolidation: rewrite with hindsight", "Mastra Reflector, Auto-Dreamer")
        self.play(FadeIn(top), run_time=0.5)
        raw_cards = []
        for i in range(12, 18):
            b = panel(1.18, 0.54, PANEL, "#405064")
            t = label(f"turn {i}", 13, TEXT, BOLD).move_to(b.get_top() + DOWN * 0.17)
            s = label("raw event", 11, MUTED).move_to(b.get_bottom() + UP * 0.17)
            raw_cards.append(VGroup(b, t, s))
        raw = VGroup(*raw_cards).arrange_in_grid(rows=2, cols=3, buff=0.13).move_to(LEFT * 4.25)
        sleep = panel(2.45, 2.2, "#171629", PURPLE).move_to(ORIGIN)
        moon = Circle(radius=0.36, fill_color=YELLOW, fill_opacity=1, stroke_width=0).move_to(sleep.get_top() + DOWN * 0.58 + LEFT * 0.35)
        cut = Circle(radius=0.34, fill_color="#171629", fill_opacity=1, stroke_width=0).move_to(moon.get_center() + RIGHT * 0.18 + UP * 0.08)
        sleep_text = paragraph("offline\nconsolidator", 27, TEXT, 20).move_to(sleep.get_center() + DOWN * 0.25)
        compact = VGroup(
            memory_card("fact", "what stayed true", GREEN, ""),
            memory_card("procedure", "what to do next", BLUE, ""),
            memory_card("archive", "what no longer matters", ORANGE, ""),
        ).arrange(DOWN, buff=0.18).scale(0.72).move_to(RIGHT * 4.2)
        a1 = Arrow(raw.get_right(), sleep.get_left(), color=PURPLE, stroke_width=4)
        a2 = Arrow(sleep.get_right(), compact.get_left(), color=PURPLE, stroke_width=4)
        mastra = label("prompted reflector", 18, MUTED).next_to(sleep, DOWN, buff=0.18).shift(LEFT * 0.8)
        autod = label("learned consolidator + reward", 18, PURPLE).next_to(mastra, DOWN, buff=0.12).shift(RIGHT * 0.35)
        self.play(LaggedStart(*(FadeIn(x, shift=RIGHT) for x in raw), lag_ratio=0.08), run_time=1.6)
        self.play(Create(a1), FadeIn(sleep), FadeIn(moon), FadeIn(cut), FadeIn(sleep_text), run_time=1.3)
        self.play(Create(a2), LaggedStart(*(FadeIn(x, shift=LEFT) for x in compact), lag_ratio=0.12), run_time=1.7)
        self.play(FadeIn(mastra), FadeIn(autod), Indicate(autod, color=PURPLE), run_time=1.8)
        formula = paragraph("reward = task success + compactness + utility", 23, YELLOW, 52).to_edge(DOWN).shift(UP * 0.65)
        self.play(FadeIn(formula), run_time=0.8)
        self.end_segment(dur, 7.7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def recursive_read_time(self):
        dur = self.audio("06_recursive_read_time_memory", 29)
        top = self.title_bar("6. Recursive read-time memory", "Recursive Language Models")
        self.play(FadeIn(top), run_time=0.5)
        corpus = panel(3.1, 5.1, "#111B24", BLUE).move_to(LEFT * 4.4)
        lines = VGroup(*[Line(LEFT * 1.15, RIGHT * 1.15, color="#3A5064", stroke_width=2) for _ in range(18)]).arrange(DOWN, buff=0.14).move_to(corpus)
        q = panel(2.3, 0.68, "#251B12", ORANGE).move_to(UP * 2.45)
        qt = label("hard question", 21, TEXT, BOLD).move_to(q)
        slices = VGroup(*[panel(1.55, 0.62, "#132A35", CYAN) for _ in range(4)]).arrange(DOWN, buff=0.18).move_to(LEFT * 0.6)
        slice_text = VGroup(*[label(f"slice {i}", 15, TEXT).move_to(slices[i - 1]) for i in range(1, 5)])
        agg = panel(2.3, 1.6, "#171D2B", PURPLE).move_to(RIGHT * 3.5)
        agg_t = paragraph("aggregate\npartial answers", 24, TEXT, 24).move_to(agg)
        self.play(FadeIn(corpus), FadeIn(lines), FadeIn(q), FadeIn(qt), run_time=1.2)
        self.play(LaggedStart(*(TransformFromCopy(lines[i * 4], slices[i]) for i in range(4)), lag_ratio=0.18), FadeIn(slice_text), run_time=2.0)
        self.play(
            LaggedStart(*(Create(Arrow(s.get_right(), agg.get_left(), color=CYAN, stroke_width=3)) for s in slices), lag_ratio=0.08),
            FadeIn(agg),
            FadeIn(agg_t),
            run_time=1.7,
        )
        note = paragraph("Spend compute when the question is hard.", 27, YELLOW, 45).to_edge(DOWN).shift(UP * 0.65)
        self.play(FadeIn(note), Circumscribe(agg, color=PURPLE), run_time=1.4)
        self.end_segment(dur, 6.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def learned_policy(self):
        dur = self.audio("07_learned_memory_policies", 31)
        top = self.title_bar("7. Learned memory policies", "Memory-R1, Search-R1 style environments")
        self.play(FadeIn(top), run_time=0.5)
        state = panel(2.4, 1.05, "#13231A", GREEN).move_to(LEFT * 4.2 + UP * 1.4)
        st = paragraph("state:\ntrace + task + budget", 20, TEXT, 26).move_to(state)
        policy = panel(2.5, 1.25, "#171629", PURPLE).move_to(ORIGIN + UP * 1.4)
        pt = label("policy", 30, TEXT, BOLD).move_to(policy)
        actions = VGroup(
            memory_card("search", "raw traces", BLUE, ""),
            memory_card("write", "memory state", GREEN, ""),
            memory_card("update", "new evidence", YELLOW, ""),
            memory_card("answer", "final response", RED, ""),
        ).arrange_in_grid(rows=2, cols=2, buff=0.18).scale(0.62).move_to(RIGHT * 4.2 + UP * 1.15)
        reward = panel(3.4, 1.2, "#251B12", ORANGE).move_to(DOWN * 1.9)
        rt = paragraph("reward:\ncorrect + supported + current - cost", 21, TEXT, 40).move_to(reward)
        arrows = VGroup(
            Arrow(state.get_right(), policy.get_left(), color=GREEN, stroke_width=4),
            Arrow(policy.get_right(), actions.get_left(), color=PURPLE, stroke_width=4),
            Arrow(actions.get_bottom(), reward.get_right(), color=ORANGE, stroke_width=4),
            CurvedArrow(reward.get_left(), state.get_bottom(), color=ORANGE, stroke_width=4, angle=-TAU / 3),
        )
        hard = paragraph("Hard part: which memory action deserved credit?", 26, YELLOW, 50).to_edge(DOWN).shift(UP * 0.55)
        self.play(FadeIn(state), FadeIn(st), FadeIn(policy), FadeIn(pt), run_time=1.2)
        self.play(Create(arrows[0]), Create(arrows[1]), FadeIn(actions), run_time=1.7)
        self.play(Create(arrows[2]), Create(arrows[3]), FadeIn(reward), FadeIn(rt), run_time=1.5)
        self.play(FadeIn(hard), Circumscribe(actions[1], color=YELLOW), Circumscribe(actions[2], color=YELLOW), run_time=1.8)
        self.end_segment(dur, 6.7)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def learned_chunking(self):
        dur = self.audio("08_learned_chunking", 30)
        top = self.title_bar("8. Learned chunking: learn the unit", "H-Net")
        self.play(FadeIn(top), run_time=0.5)
        text_line = Line(LEFT * 5.7, RIGHT * 5.7, color="#2E4053", stroke_width=8).move_to(UP * 1.6)
        fixed = VGroup(*[Line(UP * 0.45, DOWN * 0.45, color=RED, stroke_width=3).move_to(LEFT * 5.7 + RIGHT * (i * 1.15)) for i in range(11)])
        fixed_lbl = label("fixed windows", 24, RED, BOLD).next_to(text_line, UP, buff=0.35)
        learned_line = Line(LEFT * 5.7, RIGHT * 5.7, color="#2E4053", stroke_width=8).move_to(DOWN * 0.85)
        positions = [-5.4, -4.1, -2.8, -0.6, 0.15, 2.6, 4.7, 5.7]
        learned = VGroup(*[Line(UP * 0.45, DOWN * 0.45, color=GREEN, stroke_width=3).move_to(np.array([x, -0.85, 0])) for x in positions])
        learned_lbl = label("learned boundaries", 24, GREEN, BOLD).next_to(learned_line, UP, buff=0.35)
        tags = VGroup(
            label("topic", 14, MUTED).move_to(np.array([-4.75, -1.55, 0])),
            label("function", 14, MUTED).move_to(np.array([-1.55, -1.55, 0])),
            label("contradiction", 14, MUTED).move_to(np.array([1.4, -1.55, 0])),
            label("episode", 14, MUTED).move_to(np.array([4.0, -1.55, 0])),
        )
        self.play(Create(text_line), FadeIn(fixed_lbl), Create(fixed), run_time=1.5)
        self.play(Create(learned_line), FadeIn(learned_lbl), Create(learned), FadeIn(tags), run_time=1.8)
        note = paragraph("Chunking is not just preprocessing. It defines what memory can remember.", 26, YELLOW, 58).to_edge(DOWN).shift(UP * 0.6)
        self.play(FadeIn(note), run_time=1.0)
        self.end_segment(dur, 4.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def latent_cache_memory(self):
        dur = self.audio("09_latent_and_kv_cache_memory", 34)
        top = self.title_bar("9. Latent and KV cache memory", "Cartridges, Titans, KV cache systems")
        self.play(FadeIn(top), run_time=0.5)
        corpus = VGroup(*[tiny_doc(f"{i}") for i in range(1, 6)]).arrange(RIGHT, buff=0.12).move_to(LEFT * 4.4 + UP * 1.25)
        trainer = panel(2.3, 1.1, "#171629", PURPLE).move_to(LEFT * 1.1 + UP * 1.25)
        trainer_t = paragraph("offline\nself-study", 22, TEXT, 20).move_to(trainer)
        cache = VGroup(*[Rectangle(width=0.25, height=1.6, fill_color=c, fill_opacity=0.9, stroke_width=0) for c in [BLUE, CYAN, GREEN, YELLOW, PINK]]).arrange(RIGHT, buff=0.06).move_to(RIGHT * 1.5 + UP * 1.25)
        cache_box = SurroundingRectangle(cache, color=PINK, buff=0.18, corner_radius=0.08)
        cache_t = label("compact KV cache", 18, PINK).next_to(cache_box, DOWN, buff=0.12)
        model = panel(2.2, 1.25, "#1B1A2A", BLUE).move_to(RIGHT * 4.7 + UP * 1.25)
        model_t = label("model", 30, TEXT, BOLD).move_to(model)
        text_side = panel(4.6, 1.55, "#13231A", GREEN).move_to(LEFT * 2.75 + DOWN * 1.45)
        text_t = paragraph("Text memory:\nauditable, citeable, editable", 23, TEXT, 40).move_to(text_side)
        latent_side = panel(4.6, 1.55, "#241629", PINK).move_to(RIGHT * 2.75 + DOWN * 1.45)
        latent_t = paragraph("Latent memory:\nfast, compact, hard to inspect", 23, TEXT, 40).move_to(latent_side)
        self.play(FadeIn(corpus), run_time=1.0)
        self.play(Create(Arrow(corpus.get_right(), trainer.get_left(), color=PURPLE)), FadeIn(trainer), FadeIn(trainer_t), run_time=1.2)
        self.play(Create(Arrow(trainer.get_right(), cache_box.get_left(), color=PINK)), FadeIn(cache), FadeIn(cache_box), FadeIn(cache_t), run_time=1.4)
        self.play(Create(Arrow(cache_box.get_right(), model.get_left(), color=BLUE)), FadeIn(model), FadeIn(model_t), run_time=1.1)
        self.play(FadeIn(text_side), FadeIn(text_t), FadeIn(latent_side), FadeIn(latent_t), run_time=1.6)
        self.end_segment(dur, 6.3)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def synthesis(self):
        dur = self.audio("10_synthesis", 45)
        top = self.title_bar("10. Synthesis: the likely winning stack")
        self.play(FadeIn(top), run_time=0.5)
        layers = [
            ("raw traces", "messages, files, commits, tools", BLUE),
            ("evidence index", "spans, metadata, time, code", CYAN),
            ("memory views", "facts, procedures, project state", GREEN),
            ("temporal layer", "validity, updates, active processes", YELLOW),
            ("offline consolidation", "compact with hindsight", PURPLE),
            ("learned controller", "retrieve, refresh, answer, wait", RED),
        ]
        stack = VGroup()
        for title, sub, color in layers:
            b = panel(6.75, 0.58, "#121A22", color)
            title_gap = 0.14 if title == "temporal layer" else 0.08
            t = words(title.split(), 18, TEXT, BOLD, buff=title_gap).move_to(b.get_left() + RIGHT * 1.28)
            s = label(sub, 12, MUTED).move_to(b.get_right() + LEFT * 1.75)
            stack.add(VGroup(b, t, s))
        stack.arrange(DOWN, buff=0.11).move_to(LEFT * 1.35 + UP * 0.60)
        question = panel(4.95, 0.95, "#251B12", ORANGE).to_edge(DOWN).shift(UP * 0.30 + LEFT * 0.2)
        qt = VGroup(
            words(["Task", "+", "time", "+", "history", "+", "budget"], 20, TEXT, buff=0.08),
            words(["->", "what", "should", "the", "agent", "see", "next?"], 19, TEXT, buff=0.07),
        ).arrange(DOWN, buff=0.12).move_to(question)
        self.play(LaggedStart(*(FadeIn(layer, shift=UP) for layer in stack), lag_ratio=0.12), run_time=3.3)
        self.play(FadeIn(question), FadeIn(qt), run_time=1.2)
        cards = VGroup(
            words(["not", "a", "database"], 17, MUTED, buff=0.11),
            words(["not", "a", "prompt"], 17, MUTED, buff=0.11),
            words(["learned", "continuity", "layer"], 20, GREEN, BOLD, buff=0.09),
        ).arrange(DOWN, buff=0.22).move_to(RIGHT * 4.55 + UP * 1.12)
        self.play(LaggedStart(*(FadeIn(c, shift=LEFT) for c in cards), lag_ratio=0.18), run_time=1.8)
        self.end_segment(dur, 6.8)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.6)

    def references(self):
        top = label("Sources to read next", 40, TEXT, BOLD).to_edge(UP).shift(DOWN * 0.35)
        refs = [
            "RAG — Lewis et al. 2020",
            "Zep / Graphiti — Temporal KG for agent memory",
            "Mastra Observational Memory — Observer + Reflector",
            "Event Sourcing — Martin Fowler",
            "Auto-Dreamer — learned offline consolidation",
            "RLM — recursive read-time context inspection",
            "Memory-R1 — RL memory manager",
            "H-Net — learned chunk boundaries",
            "Cartridges + Titans — KV / neural memory",
        ]
        left = VGroup(*[paragraph(r, 20, TEXT, 56) for r in refs[:5]]).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(LEFT * 3.35)
        right = VGroup(*[paragraph(r, 20, TEXT, 56) for r in refs[5:]]).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to(RIGHT * 3.25)
        footer = label("Links live in sources.json", 17, MUTED).to_edge(DOWN).shift(UP * 0.35)
        self.play(FadeIn(top), LaggedStart(*(FadeIn(x, shift=UP) for x in left), lag_ratio=0.08), LaggedStart(*(FadeIn(x, shift=UP) for x in right), lag_ratio=0.08), FadeIn(footer), run_time=3.5)
        self.wait(5.5)
        self.play(FadeOut(top), FadeOut(left), FadeOut(right), FadeOut(footer), run_time=0.8)
