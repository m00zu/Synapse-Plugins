"""
report_node.py — ReportNode plugin for Synapse.

Generates an HTML scientific report from upstream tables and figures,
using an LLM to write narrative text (Methods, Results, Conclusion).
Reuses the LLM provider/key configuration from the AI Assistant panel.
"""

from __future__ import annotations

import base64
import io
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from nodes.base import BaseExecutionNode, PORT_COLORS, NodeFileSaver
from NodeGraphQt.widgets.node_widgets import NodeBaseWidget
import NodeGraphQt
from data_models import TableData, FigureData, ImageData, HtmlData

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LLM_CONFIG_PATH = Path.home() / ".synapse_llm_config.json"


def _read_port(node, port_name):
    """Return the data value from a connected input port, or None."""
    port = node.inputs().get(port_name)
    if not port or not port.connected_ports():
        return None
    cp = port.connected_ports()[0]
    return cp.node().output_values.get(cp.name())


def _read_all_ports(node, port_name):
    """Return list of (data, source_node) for all connections to a port."""
    port = node.inputs().get(port_name)
    if not port or not port.connected_ports():
        return []
    results = []
    for cp in port.connected_ports():
        data = cp.node().output_values.get(cp.name())
        if data is not None:
            results.append((data, cp.node()))
    return results


def _walk_upstream(node, visited=None):
    """Walk upstream from *node* and return an ordered list of step dicts."""
    if visited is None:
        visited = set()
    if node.id in visited:
        return []
    visited.add(node.id)

    steps = []
    for port in node.inputs().values():
        for cp in port.connected_ports():
            steps.extend(_walk_upstream(cp.node(), visited))

    # Collect non-framework, non-UI properties from the node.
    # Skip properties that are purely cosmetic or at default values
    # to avoid confusing the LLM with irrelevant parameters.
    _SKIP_PROPS = frozenset({
        'name', 'color', 'border_color', 'text_color', 'type', 'id', 'pos',
        'layout_direction', 'selected', 'visible', 'custom', 'progress',
        'table_view', 'image_view', 'show_preview', 'live_preview',
        # UI-only cosmetic properties
        'fig_width', 'fig_height', 'tick_rotation', 'palette',
        'bar_value_fmt', 'bar_value_fontsize', 'bar_value_color',
        'bar_value_fontweight', 'bar_value_offset', 'bar_width',
        'capsize', 'err_linewidth', 'err_color', 'point_style',
        'point_size', 'point_color', 'stat_line_color', 'stat_line_width',
        'stat_text_color', 'stat_text_size', 'stat_y_offset',
        'stat_show_ns', 'stat_label_mode',
        'show_bar_values', 'show_points',
        'eq_x', 'eq_y', 'eq_size', 'eq_spacing',
        'n_inputs', 'n_outputs', 'script_code',
    })
    name = getattr(node, 'NODE_NAME', node.__class__.__name__)
    key_props = {}
    try:
        all_props = node.model.custom_properties if hasattr(node, 'model') else {}
        for k, v in all_props.items():
            if k in _SKIP_PROPS or k.startswith('_'):
                continue
            if v is not None and v != '' and v != [] and v != {}:
                key_props[k] = v
    except Exception:
        pass

    # Extract first paragraph of docstring (scientific description)
    raw_doc = getattr(node.__class__, '__doc__', '') or ''
    doc_lines = []
    for line in raw_doc.strip().splitlines():
        line = line.strip()
        if not line:
            break  # stop at first blank line (end of first paragraph)
        if line.lower().startswith('keywords'):
            break
        doc_lines.append(line)
    brief_doc = ' '.join(doc_lines)

    steps.append({
        'name': name,
        'props': key_props,
        'doc': brief_doc,
    })
    return steps


def _format_pipeline(steps: list[dict]) -> str:
    """Format pipeline steps as a short chain string (for HTML header)."""
    return " → ".join(
        f"{s['name']}({s['props']})" if s['props'] else s['name']
        for s in steps
    )


def _format_pipeline_detailed(steps: list[dict]) -> str:
    """Format pipeline steps with docstrings (for LLM prompt)."""
    lines = []
    for i, s in enumerate(steps, 1):
        props_str = f", parameters: {s['props']}" if s['props'] else ""
        doc_str = f"\n   Description: {s['doc']}" if s['doc'] else ""
        lines.append(f"{i}. **{s['name']}**{props_str}{doc_str}")
    return "\n".join(lines)


def _fig_to_base64_png(fig, dpi=150) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def _build_llm_prompt(pipeline_desc: list[dict], table_summaries: list[str],
                      n_figures: int, user_context: str,
                      table_names: list[str] | None = None,
                      fig_names: list[str] | None = None) -> str:
    """Build the prompt sent to the LLM for report generation."""
    detailed = _format_pipeline_detailed(pipeline_desc) if pipeline_desc else "(no upstream nodes)"
    parts = [
        "You are a scientific report writer. Write a concise analysis report "
        "based on the following workflow and data.\n",
        "## Analysis Pipeline\n",
        detailed,
        "",
    ]

    n_tables = len(table_summaries)
    if table_summaries:
        parts.append("## Data Summary\n")
        for i, summary in enumerate(table_summaries, 1):
            tname = table_names[i - 1] if table_names and i <= len(table_names) else f"Table {i}"
            parts.append(f"### Table {i} (from '{tname}')\n{summary}\n")

    if n_figures:
        parts.append(f"\n{n_figures} figure(s) are included in the report.")
        if fig_names:
            for i, name in enumerate(fig_names, 1):
                parts.append(f"  - Figure {i}: from '{name}' node")
        parts.append("")

    if user_context:
        parts.append(f"\n## Additional Context\n{user_context}\n")

    section_num = 3
    parts.append(
        "\n## Instructions\n"
        "Write a report with these sections:\n"
        "1. **Methods** — Describe the analysis pipeline in scientific language. "
        "Be specific about parameters used. ONLY mention parameters that are "
        "relevant to the chosen method — for example, if the method is "
        "Student's T-test, do NOT mention permutation count; if p-value "
        "adjustment is 'none', do NOT say any correction was applied.\n"
        "2. **Results** — Summarize key findings from the data tables. "
        "Reference specific numbers.\n"
        "3. **Conclusion** — Brief interpretation of the results.\n"
    )
    headings = "'## Methods', '## Results', '## Conclusion'"
    if n_tables:
        section_num += 1
        parts.append(
            f"{section_num}. **Table Legends** — Write a publication-style caption "
            f"for each of the {n_tables} table(s). Format as 'Table 1. Description...' "
            "for each. Describe the PURPOSE of the table and what each column "
            "represents. Do NOT describe summary statistics (count, mean, std, "
            "quartiles) — those are provided to you only as context for writing "
            "the Results section. The table legend should describe the table's "
            "content and role in the analysis.\n"
        )
        headings += ", '## Table Legends'"
    if n_figures:
        section_num += 1
        parts.append(
            f"{section_num}. **Figure Legends** — Write a publication-style caption "
            f"for each of the {n_figures} figure(s). Format as 'Figure 1. Description...' "
            "for each. Describe what the figure shows, axes, any statistical "
            "annotations visible (e.g. significance brackets, R² values), and key "
            "trends. Be definitive — do NOT use hedging words like 'likely', "
            "'possibly', 'may'. If significance brackets are present, state it as "
            "fact. Describe exactly what is shown based on the pipeline and data.\n"
        )
        headings += ", '## Figure Legends'"
    parts.append(
        "Keep it under 600 words. Use scientific language but be concise.\n\n"
        "IMPORTANT: Output the report as PLAIN TEXT inside a single code block "
        "(triple backticks). Use " + headings + " as section headings. "
        "No bold, no bullet points, no markdown rendering. "
        "The text will be parsed by a program, so it must be inside a code block."
    )
    return "\n".join(parts)


def _render_html(title: str, report_text: str, tables: list[pd.DataFrame],
                 figures_b64: list[str], pipeline_desc: list[dict], *,
                 table_names: list[str] | None = None,
                 fig_names: list[str] | None = None) -> str:
    """Render the final self-contained HTML report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build table HTML
    table_html_parts = []
    for i, df in enumerate(tables, 1):
        label = (table_names[i - 1] if table_names and i <= len(table_names)
                 else f"Table {i}")
        tbl = df.to_html(index=False, classes='data-table',
                         float_format=lambda x: f'{x:.4g}',
                         border=0)
        extra = f"<p class='table-note'>{len(df)} rows</p>"
        table_html_parts.append(
            f"<div class='table-container'>"
            f"<h3>Table {i} — {label}</h3>{tbl}{extra}</div>"
        )

    # Build figure HTML
    figure_html_parts = []
    for i, b64 in enumerate(figures_b64, 1):
        label = (fig_names[i - 1] if fig_names and i <= len(fig_names)
                 else f"Figure {i}")
        figure_html_parts.append(
            f"<div class='figure-container'>"
            f"<img src='data:image/png;base64,{b64}' alt='{label}'/>"
            f"<p class='figure-caption'>Figure {i} — {label}</p></div>"
        )

    # Convert report markdown-ish text to simple HTML paragraphs
    report_html = ""
    for line in report_text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('## ') or line.startswith('**') and line.endswith('**'):
            heading = line.strip('#* ')
            report_html += f"<h2>{heading}</h2>\n"
        elif line.startswith('### '):
            report_html += f"<h3>{line[4:]}</h3>\n"
        else:
            report_html += f"<p>{line}</p>\n"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  @media print {{ @page {{ margin: 2cm; }} }}
  body {{
    font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
    max-width: 900px; margin: 0 auto; padding: 2em;
    color: #222; line-height: 1.6;
  }}
  h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 0.3em; color: #2c3e50; }}
  h2 {{ color: #2c3e50; margin-top: 1.5em; }}
  h3 {{ color: #34495e; }}
  .meta {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 2em; }}
  .pipeline {{ background: #f8f9fa; padding: 1em; border-left: 3px solid #3498db;
               font-family: monospace; font-size: 0.85em; margin: 1em 0;
               white-space: pre-wrap; }}
  .data-table {{
    border-collapse: collapse; width: 100%; margin: 0.5em 0; font-size: 0.75em;
  }}
  .data-table th {{
    background: #2c3e50; color: white; padding: 4px 8px; text-align: left;
    position: sticky; top: 0; z-index: 1;
  }}
  .data-table td {{ padding: 3px 8px; border-bottom: 1px solid #ecf0f1; }}
  .data-table tr:nth-child(even) {{ background: #f8f9fa; }}
  .table-container {{
    margin: 1.5em 0; overflow-x: auto; overflow-y: auto;
    max-height: 400px; border: 1px solid #ecf0f1; border-radius: 4px;
  }}
  .table-note {{ color: #7f8c8d; font-size: 0.8em; font-style: italic; }}
  .figure-container {{ text-align: center; margin: 1.5em 0; }}
  .figure-container img {{ max-width: 100%; border: 1px solid #ecf0f1; }}
  .figure-caption {{ color: #7f8c8d; font-size: 0.9em; font-style: italic; }}
  .footer {{ margin-top: 3em; padding-top: 1em; border-top: 1px solid #ecf0f1;
             color: #95a5a6; font-size: 0.8em; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">Generated {now} &mdash; Synapse Scientific Workflow Editor</div>

<h2>Analysis Pipeline</h2>
<div class="pipeline">{_format_pipeline(pipeline_desc) if pipeline_desc else "(no upstream nodes)"}</div>

{report_html}

{"<h2>Data</h2>" if table_html_parts else ""}
{"".join(table_html_parts)}

{"<h2>Figures</h2>" if figure_html_parts else ""}
{"".join(figure_html_parts)}

<div class="footer">
  Report generated by Synapse ReportNode.
  All analysis steps are reproducible via the saved workflow.
</div>
</body>
</html>"""


def _create_llm_client():
    """Create an LLM client from the saved AI Assistant configuration.

    Returns (client, provider_name) or (None, error_message).
    """
    try:
        cfg = json.loads(_LLM_CONFIG_PATH.read_text())
    except Exception:
        return None, "No LLM configuration found. Set up a provider in the AI Assistant panel first."

    provider = cfg.get("provider", "Ollama")
    model = cfg.get("last_models", {}).get(provider, "")

    # Import client classes lazily to avoid circular imports
    from synapse.llm_assistant import (
        OllamaClient, OpenAIClient, ClaudeClient,
        GroqClient, GeminiClient, _retrieve_api_key,
    )

    api_key = _retrieve_api_key(provider, cfg.get("api_keys", {}).get(provider, ""))

    if provider == "Ollama":
        client = OllamaClient(model=model or OllamaClient.DEFAULT_MODEL)
    elif provider == "Ollama Cloud":
        client = OllamaClient(
            base_url=OllamaClient.CLOUD_BASE_URL,
            model=model or OllamaClient.DEFAULT_MODEL,
            api_key=api_key,
        )
    elif provider == "OpenAI":
        if not api_key:
            return None, "OpenAI API key not set."
        client = OpenAIClient(api_key=api_key, model=model or OpenAIClient.DEFAULT_MODEL)
    elif provider == "Claude":
        if not api_key:
            return None, "Claude API key not set."
        client = ClaudeClient(api_key=api_key, model=model or ClaudeClient.DEFAULT_MODEL)
    elif provider == "Groq":
        if not api_key:
            return None, "Groq API key not set."
        client = GroqClient(api_key=api_key, model=model or GroqClient.DEFAULT_MODEL)
    elif provider == "Gemini":
        if not api_key:
            return None, "Gemini API key not set."
        client = GeminiClient(api_key=api_key, model=model or GeminiClient.DEFAULT_MODEL)
    else:
        return None, f"Provider '{provider}' is not supported for report generation."

    return client, provider


# ---------------------------------------------------------------------------
# ReportNode
# ---------------------------------------------------------------------------

class _MultiLineTextWidget(NodeBaseWidget):
    """A node widget with a multi-line QPlainTextEdit."""
    def __init__(self, parent=None, name='', label='', text='', height=80):
        super().__init__(parent, name, label)
        from PySide6 import QtWidgets, QtGui
        self._edit = QtWidgets.QPlainTextEdit()
        self._edit.setPlainText(text)
        self._edit.setMinimumHeight(height)
        self._edit.setMaximumHeight(height)
        self._edit.setFont(QtGui.QFont("Helvetica Neue", 9))
        self._edit.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #ddd; "
            "border: 1px solid #555; border-radius: 3px; }"
        )
        self._edit.textChanged.connect(
            lambda: self.value_changed.emit(self.get_name(), self.get_value())
        )
        self.set_custom_widget(self._edit)

    def get_value(self):
        return self._edit.toPlainText()

    def set_value(self, value):
        self._edit.blockSignals(True)
        self._edit.setPlainText(str(value or ''))
        self._edit.blockSignals(False)


class _ButtonWidget(NodeBaseWidget):
    """A node widget wrapping a QPushButton."""
    def __init__(self, parent=None, name='', label='', button_text='',
                 tooltip=''):
        super().__init__(parent, name, label)
        from PySide6 import QtWidgets
        self.btn = QtWidgets.QPushButton(button_text)
        if tooltip:
            self.btn.setToolTip(tooltip)
        self.set_custom_widget(self.btn)

    def get_value(self): return ''
    def set_value(self, _v): pass


class ReportNode(BaseExecutionNode):
    """
    Generates an HTML scientific report from upstream tables and figures.

    **Workflow:**

    - Connect table and/or figure inputs, then run the graph. The node collects all upstream data and prepares the report prompt.
    - Click **Generate with API** to use the configured LLM, or click **Copy for Web AI** to paste into ChatGPT / Claude.ai / Gemini.
    - If using web AI, paste the response into the **AI Response** box, then click **Build Report** to render the HTML.

    **title** — report title (default: "Analysis Report").

    **context** — optional text giving the LLM additional context (e.g. "This is a cell viability assay comparing drug A vs control").

    Keywords: report, export, html, pdf, summary, write, document, 報告, 匯出, 摘要, 文件
    """

    __identifier__ = 'plugins.Plugins.Report'
    NODE_NAME = 'Report'
    PORT_SPEC = {'inputs': ['table', 'figure'], 'outputs': ['html']}

    _EXT_FILTER = 'HTML Files (*.html);;All Files (*)'
    _UI_PROPS = frozenset({
        'context', 'web_ai_response', 'title', '_copy_btn',
        '_gen_btn', '_build_btn', 'file_path',
    })

    def __init__(self):
        super().__init__()
        self.add_input('table', color=PORT_COLORS['table'], multi_input=True)
        self.add_input('figure', color=PORT_COLORS['figure'], multi_input=True)
        self.add_output('report', color=PORT_COLORS['html'])

        # --- Widgets ---
        self.add_text_input('title', 'Title', text='Analysis Report')

        ctx_w = _MultiLineTextWidget(
            self.view, name='context', label='Context',
            text='', height=60,
        )
        self.add_custom_widget(ctx_w)

        saver = NodeFileSaver(self.view, name='file_path', label='Save Path',
                              ext_filter=self._EXT_FILTER)
        self.add_custom_widget(
            saver,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
        )

        # --- Action buttons ---
        copy_w = _ButtonWidget(
            self.view, name='_copy_btn', button_text='Copy for Web AI',
            tooltip="Copy the report prompt to clipboard.\n"
                    "Paste into ChatGPT / Claude.ai / Gemini.",
        )
        copy_w.btn.clicked.connect(self._on_copy_for_web)
        self.add_custom_widget(copy_w)

        gen_w = _ButtonWidget(
            self.view, name='_gen_btn', button_text='Generate with API',
            tooltip="Call the LLM configured in the AI Assistant panel\n"
                    "to generate the report narrative.",
        )
        gen_w.btn.clicked.connect(self._on_generate_api)
        self.add_custom_widget(gen_w)

        # Multi-line text area for pasting web AI response
        resp_w = _MultiLineTextWidget(
            self.view, name='web_ai_response', label='AI Response',
            text='', height=120,
        )
        self.add_custom_widget(resp_w)

        # Build Report + Save PDF in one row
        class _DualButtonWidget(NodeBaseWidget):
            def __init__(self, parent=None):
                super().__init__(parent, name='_build_pdf_row')
                from PySide6 import QtWidgets as _Qw
                container = _Qw.QWidget()
                row = _Qw.QHBoxLayout(container)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(4)
                self.build_btn = _Qw.QPushButton('Build Report')
                self.build_btn.setToolTip(
                    "Render the HTML report and open in browser")
                self.pdf_btn = _Qw.QPushButton('Save as PDF')
                self.pdf_btn.setToolTip("Export the report as a PDF file")
                row.addWidget(self.build_btn)
                row.addWidget(self.pdf_btn)
                self.set_custom_widget(container)
            def get_value(self): return ''
            def set_value(self, _v): pass

        dual_w = _DualButtonWidget(self.view)
        dual_w.build_btn.clicked.connect(self._on_build_report)
        dual_w.pdf_btn.clicked.connect(self._on_save_pdf)
        self.add_custom_widget(dual_w)

        # Internal state set by evaluate()
        self._tables = []
        self._figures_b64 = []
        self._pipeline_desc = []
        self._table_summaries = []
        self._ready = False
        self._tmp_dirs: list[Path] = []  # temp dirs to clean up

    # ------------------------------------------------------------------
    def _cleanup_tmp(self):
        """Remove any previously created temp figure directories."""
        import shutil
        for d in self._tmp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        self._tmp_dirs.clear()

    # ------------------------------------------------------------------
    def _gather_context(self):
        """Collect inputs, build pipeline description, and table summaries.

        Returns (tables, figures_b64, pipeline_desc, table_summaries)
        or raises ValueError if nothing is connected.
        """
        tables: list[tuple[pd.DataFrame, str]] = []
        table_sources = _read_all_ports(self, 'table')
        for data, src_node in table_sources:
            if isinstance(data, TableData) and data.payload is not None:
                src_name = getattr(src_node, 'NODE_NAME', src_node.__class__.__name__)
                tables.append((data.payload, src_name))

        figures_b64: list[tuple[str, str]] = []
        figure_sources = _read_all_ports(self, 'figure')
        for data, src_node in figure_sources:
            if isinstance(data, FigureData) and data.payload is not None:
                src_name = getattr(src_node, 'NODE_NAME', src_node.__class__.__name__)
                figures_b64.append((_fig_to_base64_png(data.payload), src_name))

        if not tables and not figures_b64:
            raise ValueError("No tables or figures connected.")

        # Build pipeline description
        all_upstream_steps = []
        for port in self.inputs().values():
            for cp in port.connected_ports():
                all_upstream_steps.extend(_walk_upstream(cp.node()))
        # Deduplicate by node name while preserving order
        seen = set()
        unique_steps = []
        for s in all_upstream_steps:
            if s['name'] not in seen:
                seen.add(s['name'])
                unique_steps.append(s)
        pipeline_desc = unique_steps

        # Summarize tables
        table_summaries = []
        for i, (df, src_name) in enumerate(tables):
            summary_parts = [f"Source: {src_name}"]
            summary_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            summary_parts.append(f"Columns: {', '.join(df.columns.tolist()[:30])}")
            try:
                desc = df.describe().to_string()
                summary_parts.append(f"Statistics:\n{desc}")
            except Exception:
                pass
            try:
                head = df.head(5).to_string()
                summary_parts.append(f"First rows:\n{head}")
            except Exception:
                pass
            table_summaries.append("\n".join(summary_parts))

        return tables, figures_b64, pipeline_desc, table_summaries

    # ------------------------------------------------------------------
    def evaluate(self):
        """Collect upstream data and prepare the prompt. Does NOT call LLM."""
        self.reset_progress()

        try:
            (self._tables, self._figures_b64,
             self._pipeline_desc, self._table_summaries) = self._gather_context()
        except ValueError as exc:
            self._ready = False
            self.mark_error()
            return False, str(exc)

        self._ready = True
        n_tables = len(self._tables)
        n_figs = len(self._figures_b64)

        self.set_progress(100)
        self.mark_clean()
        return True, (
            f"Ready — {n_tables} table(s), {n_figs} figure(s) collected.\n"
            "Use 'Generate with API' or 'Copy for Web AI' to create the report."
        )

    # ------------------------------------------------------------------
    def _on_copy_for_web(self):
        """Copy the report prompt to clipboard and save figures as temp PNGs."""
        from PySide6 import QtWidgets as _Qw

        if not self._ready:
            print("[ReportNode] Run the graph first to collect upstream data.")
            return

        user_context = str(self.get_property('context') or '').strip()

        # Clean up previous temp dirs, then save figures as temp PNGs
        self._cleanup_tmp()
        fig_paths = []
        if self._figures_b64:
            import tempfile
            tmp_dir = Path(tempfile.mkdtemp(prefix='synapse_report_figs_'))
            self._tmp_dirs.append(tmp_dir)
            for i, (b64, name) in enumerate(self._figures_b64, 1):
                fig_path = tmp_dir / f"figure_{i}_{name.replace(' ', '_')}.png"
                fig_path.write_bytes(base64.b64decode(b64))
                fig_paths.append(fig_path)

        prompt = (
            "You are a scientific report writer.\n"
            "Write clear, concise analysis reports.\n"
            "Use **bold** for section headings.\n\n"
            + _build_llm_prompt(self._pipeline_desc, self._table_summaries,
                                len(self._figures_b64), user_context,
                                table_names=[n for _, n in self._tables],
                                fig_names=[n for _, n in self._figures_b64])
        )

        _Qw.QApplication.clipboard().setText(prompt)

        if fig_paths:
            # Open the folder so user can drag images into the chat
            import subprocess, sys
            folder = str(fig_paths[0].parent)
            if sys.platform == 'darwin':
                subprocess.Popen(['open', folder])
            elif sys.platform == 'win32':
                subprocess.Popen(['explorer', folder])
            else:
                subprocess.Popen(['xdg-open', folder])
            print(f"[ReportNode] Prompt copied. {len(fig_paths)} figure(s) saved to: {folder}")
            print("[ReportNode] Drag the images into the chat alongside the pasted prompt.")
        else:
            print("[ReportNode] Prompt copied to clipboard.")

    # ------------------------------------------------------------------
    def _on_generate_api(self):
        """Call the configured LLM to generate the report narrative (background thread)."""
        from PySide6 import QtCore as _Qc

        if not self._ready:
            print("[ReportNode] Run the graph first to collect upstream data.")
            return

        # Get client from graph (shared with AI Assistant / AI Chat panels)
        client = getattr(self.graph, '_llm_client', None)
        if client is None:
            # Fallback: try loading from config file
            client, info = _create_llm_client()
            if client is None:
                print(f"[ReportNode] {info}")
                return

        user_context = str(self.get_property('context') or '').strip()
        prompt = _build_llm_prompt(self._pipeline_desc, self._table_summaries,
                                   len(self._figures_b64), user_context,
                                   table_names=[n for _, n in self._tables],
                                   fig_names=[n for _, n in self._figures_b64])

        system = (
            "You are a scientific report writer. "
            "Write clear, concise analysis reports in plain text. "
            "Use **bold** for section headings."
        )

        # Collect figure images for vision-capable models
        fig_images = [b64 for b64, _ in self._figures_b64] if self._figures_b64 else None

        # Run the LLM call on a background thread to avoid freezing the UI
        class _ReportWorker(_Qc.QObject):
            result = _Qc.Signal(str)
            error = _Qc.Signal(str)
            def __init__(self, client, system, prompt, images):
                super().__init__()
                self._client = client
                self._system = system
                self._prompt = prompt
                self._images = images
            def run(self):
                try:
                    text = self._client.chat(self._system, self._prompt,
                                             images=self._images)
                    self.result.emit(text)
                except Exception as exc:
                    self.error.emit(str(exc))

        # Disable button and show status
        self._gen_worker = _ReportWorker(client, system, prompt, fig_images)
        self._gen_thread = _Qc.QThread()
        self._gen_worker.moveToThread(self._gen_thread)

        self._gen_thread.started.connect(self._gen_worker.run)
        self._gen_worker.result.connect(self._on_api_result)
        self._gen_worker.error.connect(self._on_api_error)
        self._gen_worker.result.connect(self._gen_thread.quit)
        self._gen_worker.error.connect(self._gen_thread.quit)
        self._gen_thread.finished.connect(self._gen_worker.deleteLater)
        self._gen_thread.finished.connect(self._gen_thread.deleteLater)

        # Find and disable the Generate button
        for w_name, w_obj in self.widgets().items():
            if w_name == '_gen_btn' and hasattr(w_obj, 'btn'):
                w_obj.btn.setEnabled(False)
                w_obj.btn.setText("Generating…")

        self._gen_thread.start()

    def _on_api_result(self, report_text: str):
        """Called on main thread when the LLM worker finishes successfully."""
        # Re-enable button
        for w_name, w_obj in self.widgets().items():
            if w_name == '_gen_btn' and hasattr(w_obj, 'btn'):
                w_obj.btn.setEnabled(True)
                w_obj.btn.setText("Generate with API")

        self.set_property('web_ai_response', report_text)
        self._on_build_report()

    def _on_api_error(self, error_msg: str):
        """Called on main thread when the LLM worker fails."""
        for w_name, w_obj in self.widgets().items():
            if w_name == '_gen_btn' and hasattr(w_obj, 'btn'):
                w_obj.btn.setEnabled(True)
                w_obj.btn.setText("Generate with API")
        self.mark_error()
        print(f"[ReportNode] LLM error: {error_msg}")

    # ------------------------------------------------------------------
    @staticmethod
    def _clean_report_text(raw: str) -> str:
        """Clean up LLM response: strip code fences, parse JSON if needed."""
        import re as _re
        text = raw.strip()
        if not text:
            return text
        # Strip code block markers
        text = _re.sub(r'^```\w*\n?', '', text)
        text = _re.sub(r'\n?```$', '', text)
        text = text.strip()
        # Handle JSON responses (e.g. Gemini returns structured JSON)
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed:
                parsed = parsed[0]
            if isinstance(parsed, dict):
                section_order = ['Methods', 'Results', 'Conclusion',
                                 'Table Legends', 'Figure Legends']
                parts = []
                for section in section_order:
                    val = parsed.get(section, '')
                    if val:
                        parts.append(f'## {section}')
                        parts.append(val)
                        parts.append('')
                for key, val in parsed.items():
                    if key not in section_order and val:
                        parts.append(f'## {key}')
                        parts.append(str(val))
                        parts.append('')
                if parts:
                    text = '\n'.join(parts)
        except (json.JSONDecodeError, TypeError):
            pass
        return text

    # ------------------------------------------------------------------
    def _on_build_report(self):
        """Render HTML from the AI response and open in browser."""

        if not self._ready:
            print("[ReportNode] Run the graph first to collect upstream data.")
            return

        report_text = self._clean_report_text(
            str(self.get_property('web_ai_response') or '')
        )
        if not report_text:
            print("[ReportNode] No AI response. Use 'Generate with API' or paste a response first.")
            return

        title = str(self.get_property('title') or 'Analysis Report').strip()

        table_dfs = [df for df, _ in self._tables]
        table_names = [name for _, name in self._tables]
        fig_b64s = [b64 for b64, _ in self._figures_b64]
        fig_names = [name for _, name in self._figures_b64]

        html = _render_html(title, report_text, table_dfs, fig_b64s,
                            self._pipeline_desc,
                            table_names=table_names, fig_names=fig_names)

        # Output for downstream nodes
        self.output_values['report'] = HtmlData(payload=html, title=title)

        # Save to disk only if user specified a path
        file_path = str(self.get_property('file_path') or '').strip()
        if file_path:
            os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
            Path(file_path).write_text(html, encoding='utf-8')

        # Open in browser via a temp file if not saved
        import webbrowser, tempfile
        if file_path:
            webbrowser.open(Path(file_path).as_uri())
        else:
            tmp = tempfile.NamedTemporaryFile(
                suffix='.html', prefix='synapse_report_',
                delete=False, mode='w', encoding='utf-8',
            )
            tmp.write(html)
            tmp.close()
            webbrowser.open(Path(tmp.name).as_uri())
            # Clean up after browser has time to load
            from PySide6.QtCore import QTimer
            QTimer.singleShot(10_000, lambda p=tmp.name: Path(p).unlink(missing_ok=True))

    # ------------------------------------------------------------------
    def _on_save_pdf(self):
        """Export the current report as a PDF using fpdf2."""
        from PySide6 import QtWidgets as _Qw

        if not self._ready:
            print("[ReportNode] Run the graph first.")
            return

        # Need the built report text
        report_text = self._clean_report_text(
            str(self.get_property('web_ai_response') or '')
        )
        if not report_text:
            print("[ReportNode] Build the report first before saving as PDF.")
            return

        # Ask for save path
        file_path, _ = _Qw.QFileDialog.getSaveFileName(
            None, "Save Report as PDF", "",
            "PDF Files (*.pdf);;All Files (*)",
        )
        if not file_path:
            return
        if not file_path.lower().endswith('.pdf'):
            file_path += '.pdf'

        title = str(self.get_property('title') or 'Analysis Report').strip()

        try:
            self._build_pdf(file_path, title, report_text)
            print(f"[ReportNode] PDF saved to {file_path}")
            import webbrowser
            webbrowser.open(Path(file_path).as_uri())
        except Exception as exc:
            print(f"[ReportNode] PDF export failed: {exc}")
            import traceback
            traceback.print_exc()

    @staticmethod
    def _find_unicode_font() -> str | None:
        """Find a Unicode TTF font on the system."""
        import sys
        candidates = []
        if sys.platform == 'darwin':
            candidates = [
                '/System/Library/Fonts/Supplemental/Arial.ttf',
                '/Library/Fonts/Arial Unicode.ttf',
                '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
            ]
        elif sys.platform == 'win32':
            windir = os.environ.get('WINDIR', r'C:\Windows')
            candidates = [
                os.path.join(windir, 'Fonts', 'arial.ttf'),
                os.path.join(windir, 'Fonts', 'segoeui.ttf'),
            ]
        else:
            candidates = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _build_pdf(self, file_path: str, title: str, report_text: str):
        """Build a PDF report using fpdf2's write_html for full Unicode support."""
        from fpdf import FPDF
        import html as _html_mod
        import tempfile

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)

        # Register a Unicode TTF font
        font_path = self._find_unicode_font()
        if font_path:
            pdf.add_font('report', '', font_path)
            pdf.add_font('report', 'B', font_path)
            pdf.add_font('report', 'I', font_path)
            pdf.set_font('report', '', 10)
            _font = 'report'
        else:
            pdf.set_font('Helvetica', '', 10)
            _font = 'Helvetica'

        pdf.add_page()

        # --- Build HTML content for write_html ---
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        pipeline_str = _html_mod.escape(
            _format_pipeline(self._pipeline_desc)
        ) if self._pipeline_desc else ''

        # Convert report text (## headings + paragraphs) to HTML
        report_html = ''
        for line in report_text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('## '):
                report_html += f'<h2 color="#2c3e50">{_html_mod.escape(stripped[3:])}</h2>'
            elif stripped.startswith('### '):
                report_html += f'<h3>{_html_mod.escape(stripped[4:])}</h3>'
            else:
                report_html += f'<p>{_html_mod.escape(stripped)}</p>'

        # Build table HTML
        table_html = ''
        if self._tables:
            table_html += '<h2 color="#2c3e50">Data</h2>'
            for i, (df, src_name) in enumerate(self._tables, 1):
                display_df = df.head(50)
                cols = display_df.columns.tolist()
                if len(cols) > 10:
                    cols = cols[:10]

                table_html += f'<h3>Table {i} - {_html_mod.escape(src_name)}</h3>'
                table_html += '<font size="7"><table border="1" cellpadding="1"><thead><tr>'
                for col in cols:
                    table_html += f'<th bgcolor="#2c3e50"><font color="#ffffff">{_html_mod.escape(str(col)[:20])}</font></th>'
                table_html += '</tr></thead><tbody>'
                for _, row in display_df.iterrows():
                    table_html += '<tr>'
                    for col in cols:
                        val = row[col]
                        txt = f'{val:.4g}' if isinstance(val, float) else str(val)[:20]
                        table_html += f'<td>{_html_mod.escape(txt)}</td>'
                    table_html += '</tr>'
                table_html += '</tbody></table></font>'
                if len(df) > 50:
                    table_html += f'<p><i><font color="#888888">Showing first 50 of {len(df)} rows</font></i></p>'

        # Figures are added directly via pdf.image() after write_html
        # for proper sizing control
        tmp_paths = []

        # Build a compact pipeline string (just node names, no props)
        if self._pipeline_desc:
            short_pipeline = ' -> '.join(s['name'] for s in self._pipeline_desc)
        else:
            short_pipeline = ''

        # Compose full HTML (text + tables, no figures)
        full_html = f"""
        <h1>{_html_mod.escape(title)}</h1>
        <p><font color="#888888" size="8">Generated {now} - Synapse Scientific Workflow Editor</font></p>
        <h2 color="#2c3e50">Analysis Pipeline</h2>
        <p><font size="8">{_html_mod.escape(short_pipeline)}</font></p>
        {report_html}
        {table_html}
        """

        pdf.write_html(full_html)

        # Add figures directly via pdf.image() for proper sizing
        if self._figures_b64:
            pdf.add_page()
            pdf.set_font(_font, 'B', 16)
            pdf.cell(0, 10, 'Figures', new_x='LMARGIN', new_y='NEXT')
            pdf.ln(4)

            page_w = pdf.w - pdf.l_margin - pdf.r_margin

            for i, (b64_str, src_name) in enumerate(self._figures_b64, 1):
                img_bytes = base64.b64decode(b64_str)
                tmp = tempfile.NamedTemporaryFile(
                    suffix='.png', delete=False)
                tmp.write(img_bytes)
                tmp.close()
                tmp_paths.append(tmp.name)

                # Check if we need a new page
                if pdf.get_y() > pdf.h - 120:
                    pdf.add_page()

                try:
                    pdf.image(tmp.name, x=pdf.l_margin, w=page_w)
                except Exception:
                    pass

                pdf.ln(2)
                pdf.set_font(_font, 'I', 9)
                pdf.set_text_color(120, 120, 120)
                pdf.cell(0, 5, f'Figure {i} - {src_name}',
                         new_x='LMARGIN', new_y='NEXT')
                pdf.set_text_color(0, 0, 0)
                pdf.ln(8)

        # Footer
        pdf.ln(4)
        pdf.set_font(_font, 'I', 8)
        pdf.set_text_color(150, 150, 150)
        pdf.set_x(pdf.l_margin)
        w = pdf.w - pdf.l_margin - pdf.r_margin
        try:
            pdf.multi_cell(w, 4,
                'Report generated by Synapse ReportNode. '
                'All analysis steps are reproducible via the saved workflow.')
        except Exception:
            pass

        pdf.output(file_path)

        # Cleanup temp figure files
        for p in tmp_paths:
            Path(p).unlink(missing_ok=True)


# ===========================================================================
# SaveHtmlNode
# ===========================================================================

class SaveHtmlNode(BaseExecutionNode):
    """
    Saves HtmlData to an HTML file on disk.

    Connect the **html** output from a Report node. Choose a save path
    and the node writes the self-contained HTML file.

    Keywords: save, html, report, export, write, 儲存, 匯出, 報告, HTML
    """
    __identifier__ = 'plugins.Plugins.Report'
    NODE_NAME = 'Save HTML'
    PORT_SPEC = {'inputs': ['html'], 'outputs': []}

    _EXT_FILTER = 'HTML Files (*.html);;All Files (*)'

    def __init__(self):
        super().__init__()
        self.add_input('html', color=PORT_COLORS['html'])

        saver = NodeFileSaver(self.view, name='file_path', label='Save Path',
                              ext_filter=self._EXT_FILTER)
        self.add_custom_widget(
            saver,
            widget_type=NodeGraphQt.constants.NodePropWidgetEnum.QLINE_EDIT.value,
        )

    def evaluate(self):
        self.reset_progress()

        port = self.inputs().get('html')
        if not port or not port.connected_ports():
            self.mark_error()
            return False, "No input connected"
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())

        if not isinstance(data, HtmlData) or not data.payload:
            self.mark_error()
            return False, "Input must be HtmlData"

        file_path = str(self.get_property('file_path') or '').strip()
        if not file_path:
            self.mark_error()
            return False, "No file path specified"

        self.set_progress(50)
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        Path(file_path).write_text(data.payload, encoding='utf-8')

        self.set_progress(100)
        self.mark_clean()
        return True, None


# ===========================================================================
# DisplayHtmlNode
# ===========================================================================

class _NodeHtmlWidget(NodeBaseWidget):
    """Embeds a QTextBrowser on a node to display HTML content."""

    _DEFAULT_W = 650
    _DEFAULT_H = 550

    def __init__(self, parent=None):
        super().__init__(parent, name='html_view', label='')
        from PySide6 import QtWidgets as _Qw

        container = _Qw.QWidget()
        container.setFixedSize(self._DEFAULT_W, self._DEFAULT_H)
        layout = _Qw.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._browser = _Qw.QTextBrowser()
        self._browser.setOpenExternalLinks(True)
        self._browser.setReadOnly(True)
        # Force white background with dark text regardless of app theme
        self._browser.setStyleSheet(
            "QTextBrowser { background-color: #ffffff; color: #222222; "
            "border: 1px solid #ccc; }"
        )
        layout.addWidget(self._browser)
        self.set_custom_widget(container)

    def get_value(self):
        return self._browser.toHtml()

    def set_value(self, html):
        if html:
            # Ensure text is readable by wrapping in a body with forced colors,
            # in case the report HTML doesn't specify them explicitly
            wrapped = (
                '<div style="background-color:#ffffff; color:#222222; '
                'padding:8px;">'
                + str(html)
                + '</div>'
            )
            self._browser.setHtml(wrapped)
        else:
            self._browser.clear()


class DisplayHtmlNode(BaseExecutionNode):
    """
    Displays HtmlData content directly on the node surface.

    Connect the **html** output from a Report node to preview the
    generated report inline without opening a browser.

    Keywords: display, html, report, preview, viewer, 顯示, 預覽, 報告, HTML
    """
    __identifier__ = 'plugins.Plugins.Report'
    NODE_NAME = 'Display HTML'
    PORT_SPEC = {'inputs': ['html'], 'outputs': []}

    def __init__(self):
        super().__init__(use_progress=False)
        self.add_input('html', color=PORT_COLORS['html'])
        self._html_widget = _NodeHtmlWidget(self.view)
        self.add_custom_widget(self._html_widget, tab='View')

    def evaluate(self):
        self.reset_progress()

        port = self.inputs().get('html')
        if not port or not port.connected_ports():
            return False, "No input connected"
        cp = port.connected_ports()[0]
        data = cp.node().output_values.get(cp.name())

        if not isinstance(data, HtmlData) or not data.payload:
            return False, "Input must be HtmlData"

        self.set_display(data.payload)
        self.mark_clean()
        return True, None

    def _display_ui(self, data):
        """Update the embedded HTML widget (main thread only)."""
        self._html_widget.set_value(data)
        self.view.draw_node()
