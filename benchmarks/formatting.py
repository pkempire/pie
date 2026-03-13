"""Clean formatting utilities for benchmark output."""

def format_qa_result(
    idx: int,
    total: int,
    question: str,
    expected: str,
    predicted: str,
    score: float,
    qtype: str = "",
    latency_ms: float = 0,
) -> str:
    """Format a Q&A result in a clean, readable box."""
    
    emoji = "✅" if score == 1.0 else "🟡" if score == 0.5 else "❌"
    score_pct = int(score * 100)
    
    # Wrap long text
    def wrap(text, width: int = 70) -> str:
        text = str(text).replace("\n", " ").strip()
        if len(text) <= width:
            return text
        lines = []
        while text:
            if len(text) <= width:
                lines.append(text)
                break
            # Find last space before width
            idx = text.rfind(" ", 0, width)
            if idx == -1:
                idx = width
            lines.append(text[:idx])
            text = text[idx:].lstrip()
        return "\n│   ".join(lines)
    
    q_wrapped = wrap(question)
    e_wrapped = wrap(expected)
    p_wrapped = wrap(predicted)
    
    type_str = f" [{qtype}]" if qtype else ""
    latency_str = f" • {latency_ms:.0f}ms" if latency_ms else ""
    
    return f"""
┌─────────────────────────────────────────────────────────────────────────────
│ {emoji} [{idx}/{total}]{type_str}{latency_str}
├─────────────────────────────────────────────────────────────────────────────
│ Q: {q_wrapped}
│
│ Expected: {e_wrapped}
│
│ Got: {p_wrapped}
│
│ Score: {score_pct}%
└─────────────────────────────────────────────────────────────────────────────"""


def format_summary_header(benchmark: str, baseline: str, total: int) -> str:
    """Format a summary header."""
    return f"""
╔═════════════════════════════════════════════════════════════════════════════
║  {benchmark} — {baseline}
║  {total} questions
╚═════════════════════════════════════════════════════════════════════════════"""


def format_running_score(correct: int, total: int) -> str:
    """Format running score."""
    pct = (correct / total * 100) if total > 0 else 0
    return f"Running: {correct}/{total} ({pct:.1f}%)"
