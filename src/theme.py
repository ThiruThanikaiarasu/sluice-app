"""Presentation-only styling for the Streamlit UI.

Holds the parts of the Agent Orchestrator design language (see
`knowledge-base/DESIGN.md`) that Streamlit's `[theme]` config cannot express:
the global negative tracking, the inverted pill buttons, and the focus ring.
Everything expressible as a theme token lives in `.streamlit/config.toml`
instead, so this file stays small.

Nothing here reads or computes application state -- `apply()` emits a single
stylesheet and returns.
"""

from __future__ import annotations

import streamlit as st

# Design tokens, mirrored from .streamlit/config.toml so the CSS below reads as
# token names rather than loose hex literals.
_TOKENS = """
  --background: #0c0c09;
  --foreground: #fbfbf9;
  --card: #1d1d16;
  --muted: #2b2b22;
  --muted-foreground: #abab9c;
  --primary: #fbfbf9;
  --primary-foreground: #0c0c09;
  --brand: #d25611;
  --destructive: #ff6467;

  /* Borders are translucent white, never solid grey -- edges read as light
     rather than paint, so one token works over any surface depth. Decorative
     edges sit at the design's 10%; controls use a stronger alpha (see below). */
  --border: rgb(255 255 255 / 0.10);
  --border-control: rgb(255 255 255 / 0.35);
  --ring: #7c7c67;
"""

_CSS = """
/* ---- Signature: global negative tracking ------------------------------
   -0.5px on the body, inherited everywhere. Monospace resets to normal, as
   do data tables: tight tracking hurts character disambiguation exactly
   where figures are being read off a grid. */
html, body, [data-testid="stAppViewContainer"] {
  letter-spacing: -0.5px;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
code, pre, kbd, samp,
[data-testid="stDataFrame"],
[data-testid="stMetricValue"] {
  letter-spacing: normal;
}

/* Inline code inherits a relative size from its paragraph and lands at 12px,
   which is below the 14px the type scale specifies for code. Pin it. */
[data-testid="stMarkdownContainer"] code { font-size: 0.875rem; }

/* ---- Headings: line height tightens as size grows -------------------- */
[data-testid="stHeading"] h1 { line-height: 0.98; margin-bottom: 0.5rem; }
[data-testid="stHeading"] h2 { line-height: 1.0; }
[data-testid="stHeading"] h3 { line-height: 1.11; }

/* Lead paragraph treatment for the caption directly under a heading. */
[data-testid="stCaptionContainer"] { color: var(--muted-foreground); }

/* ---- Buttons ---------------------------------------------------------
   Full pills. The primary is inverted -- near-white fill, near-black text --
   and carries the hierarchy through weight (600) as well as through the
   inversion; the secondary stays at 400. */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
  border-radius: 999px;
  min-height: 48px;
  padding: 12px 24px;
  transition: background-color 120ms ease, border-color 120ms ease, color 120ms ease;
}
[data-testid="stBaseButton-primary"] {
  background: var(--primary);
  color: var(--primary-foreground);
  border: 1px solid transparent;
  font-weight: 600;
}
[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
  background: #ffffff;
  color: var(--primary-foreground);
}
[data-testid="stBaseButton-secondary"] {
  background: var(--background);
  color: var(--foreground);
  border: 1px solid var(--border-control);
  font-weight: 400;
}
/* Hover lifts to the foreground rather than to the brand -- the accent is
   reserved for focus moments. */
[data-testid="stBaseButton-secondary"]:hover:not(:disabled) {
  background: var(--muted);
  border-color: var(--foreground);
  color: var(--foreground);
}

/* ---- Focus ------------------------------------------------------------
   A 2px ring with an offset, on every focusable element. #7c7c67 clears 3:1
   against the canvas, the card and the muted surface, so the ring stays
   perceivable whatever it lands on. Keyboard-only, via :focus-visible. */
:where(a, button, input, select, textarea, summary, [tabindex]):focus-visible,
[data-testid="stAppViewContainer"] *:focus-visible {
  outline: 2px solid var(--ring);
  outline-offset: 2px;
  border-radius: 4px;
}

/* ---- Decorative edges -------------------------------------------------
   `borderColor` is set to the control alpha in config.toml so that every
   widget boundary is identifiable without relying on a selector matching.
   These are the edges that carry no such duty -- card outlines, dividers and
   table rules -- softened back to the design's 10%. */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"] details {
  border-color: var(--border) !important;
}

/* ---- Cards ------------------------------------------------------------
   st.container(border=True) -- the remedy cards in the escalation panel. */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card);
  border-radius: 10px;
}

/* ---- Metrics ----------------------------------------------------------
   Big and light, per the heading rule; the label is the secondary text. */
[data-testid="stMetricLabel"] {
  color: var(--muted-foreground);
  font-size: 0.875rem;
  font-weight: 500;
}
[data-testid="stMetricValue"] { line-height: 1.1; }

/* ---- Dividers and dataframes ----------------------------------------- */
hr, [data-testid="stDivider"] hr { border-color: var(--border) !important; }
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ---- Motion ----------------------------------------------------------- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* ---- Forced colors ----------------------------------------------------
   Windows High Contrast strips backgrounds; keep the pill outlines drawn so
   buttons do not collapse into the page. */
@media (forced-colors: active) {
  [data-testid="stBaseButton-secondary"],
  [data-testid="stBaseButton-primary"] { border: 1px solid ButtonText; }
}
"""


def apply() -> None:
    """Inject the stylesheet. Call once, before anything else is rendered."""
    st.markdown(
        f"<style>:root {{{_TOKENS}}}\n{_CSS}</style>",
        unsafe_allow_html=True,
    )
