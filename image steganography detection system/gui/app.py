"""
Main GUI application for the Steganography Detector.
Supports single and multi-image analysis, message extraction, batch PDF reports.
"""
from utils.report import generate_batch_report
from utils.visualizer import (
    make_histogram_figure, make_lsb_figure,
    make_risk_figure, fig_to_pil,
)
from analysis.detector import StegoAnalyzer
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import sys
import json
from datetime import datetime
from io import BytesIO

from PIL import Image, ImageTk
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ── colours ─────────────────────────────────────────────────────────────
BG = "#0d1117"
PANEL = "#161b22"
PANEL2 = "#1f2937"
BORDER = "#30363d"
ACCENT = "#00d4aa"
ACCENT2 = "#ff6b6b"
ACCENT3 = "#ffd93d"
TEXT = "#e6edf3"
SUBTEXT = "#8b949e"
DETECTED = "#ff4444"
CLEAN = "#00d4aa"

FONT_MONO = ("Courier New", 9)
FONT_BODY = ("Segoe UI", 9)
FONT_TITLE = ("Segoe UI", 14, "bold")
FONT_H2 = ("Segoe UI", 11, "bold")
FONT_SMALL = ("Segoe UI", 8)


def pil_to_tk(img, max_w=600, max_h=400):
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return ImageTk.PhotoImage(img)


# ════════════════════════════════════════════════════════════════════════
class StegoDetectorApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Steganography Detector  ·  Forensic Image Analyzer")
        self.geometry("1280x820")
        self.minsize(1000, 680)
        self.configure(bg=BG)
        self._configure_styles()

        # State — list of dicts {path, analyzer, results, overall, extraction}
        self.images = []
        self.current_idx = None   # which image is shown in detail tabs
        self._tk_images = []

        self._build_ui()

    # ── TTK styles ───────────────────────────────────────────────────

    def _configure_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".",
                    background=BG, foreground=TEXT,
                    fieldbackground=PANEL, bordercolor=BORDER,
                    troughcolor=PANEL, selectbackground=ACCENT,
                    selectforeground=BG, font=FONT_BODY)
        s.configure("TFrame",  background=BG)
        s.configure("TLabel",  background=BG, foreground=TEXT)
        s.configure("TButton", background=PANEL, foreground=TEXT,
                    bordercolor=BORDER, relief="flat", padding=(12, 6))
        s.map("TButton",
              background=[("active", ACCENT), ("pressed", "#00b894")],
              foreground=[("active", BG)])
        s.configure("TNotebook", background=PANEL, bordercolor=BORDER)
        s.configure("TNotebook.Tab", background=PANEL, foreground=SUBTEXT,
                    padding=(14, 7), bordercolor=BORDER)
        s.map("TNotebook.Tab",
              background=[("selected", BG)],
              foreground=[("selected", ACCENT)])
        s.configure("TProgressbar", background=ACCENT,
                    troughcolor=PANEL, bordercolor=BORDER, thickness=5)
        s.configure("TScrollbar", background=PANEL,
                    troughcolor=BG, bordercolor=BORDER, arrowcolor=SUBTEXT)
        s.configure("Treeview", background=PANEL, foreground=TEXT,
                    fieldbackground=PANEL, bordercolor=BORDER, rowheight=28)
        s.configure("Treeview.Heading", background=BG, foreground=ACCENT,
                    bordercolor=BORDER, font=("Segoe UI", 8, "bold"))
        s.map("Treeview",
              background=[("selected", PANEL2)],
              foreground=[("selected", ACCENT)])

    # ── Top-level layout ─────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=PANEL, height=56)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⬡", bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 22)).pack(side="left", padx=(16, 5), pady=6)
        tk.Label(hdr, text="STEGO DETECTOR", bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 14, "bold")).pack(side="left", pady=6)
        tk.Label(hdr, text="Multi-Image Forensic Steganography Analyzer",
                 bg=PANEL, fg=SUBTEXT, font=FONT_BODY).pack(side="left", padx=12)

        # Status bar
        self.status_var = tk.StringVar(value="Ready — add images to begin")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="bottom")
        tk.Label(self, textvariable=self.status_var, bg=PANEL, fg=SUBTEXT,
                 font=FONT_SMALL, anchor="w", padx=12, pady=4).pack(fill="x", side="bottom")

        # Paned layout
        paned = tk.PanedWindow(self, orient="horizontal",
                               bg=BORDER, sashwidth=4, bd=0)
        paned.pack(fill="both", expand=True)

        left = tk.Frame(paned, bg=BG, width=310)
        paned.add(left, minsize=260)
        self._build_left(left)

        right = tk.Frame(paned, bg=BG)
        paned.add(right, minsize=640)
        self._build_right(right)

    # ── Left panel ───────────────────────────────────────────────────

    def _build_left(self, parent):
        self._section(parent, "IMAGES")

        btn_row = tk.Frame(parent, bg=BG)
        btn_row.pack(fill="x", padx=10, pady=(0, 6))
        self._btn(btn_row, "＋ Add Images", ACCENT, BG,
                  self._add_images).pack(side="left", fill="x", expand=True, padx=(0, 3))
        self._btn(btn_row, "✕ Remove", PANEL, SUBTEXT,
                  self._remove_selected).pack(side="left")

        # Image list
        list_frame = tk.Frame(parent, bg=PANEL,
                              highlightthickness=1, highlightbackground=BORDER)
        list_frame.pack(fill="both", expand=True, padx=10, pady=(0, 6))
        self.img_list = ttk.Treeview(list_frame,
                                     columns=("risk", "payload"), show="tree headings",
                                     selectmode="browse", height=8)
        self.img_list.heading("#0",      text="Filename", anchor="w")
        self.img_list.heading("risk",    text="Risk",     anchor="center")
        self.img_list.heading("payload", text="Payload",  anchor="center")
        self.img_list.column("#0",      width=140, anchor="w")
        self.img_list.column("risk",    width=60,  anchor="center")
        self.img_list.column("payload", width=60,  anchor="center")
        sb = ttk.Scrollbar(list_frame, orient="vertical",
                           command=self.img_list.yview)
        self.img_list.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.img_list.pack(fill="both", expand=True)
        self.img_list.bind("<<TreeviewSelect>>", self._on_list_select)

        # Progress
        self._section(parent, "ANALYSIS")
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(fill="x", padx=10, pady=(0, 5))

        self.run_btn = self._btn(parent, "▶  Analyse All Images", ACCENT, BG,
                                 self._run_all, font=("Segoe UI", 10, "bold"),
                                 pady=8, state="disabled")
        self.run_btn.pack(fill="x", padx=10, pady=(0, 5))

        self.export_btn = self._btn(parent, "⬇  Export Batch PDF Report", PANEL, TEXT,
                                    self._export_pdf, state="disabled")
        self.export_btn.pack(fill="x", padx=10, pady=(0, 4))

        self._btn(parent, "✕  Clear All", PANEL, SUBTEXT,
                  self._clear_all).pack(fill="x", padx=10, pady=(0, 6))

        # Verdict
        self._section(parent, "SELECTED IMAGE VERDICT")
        vf = tk.Frame(parent, bg=PANEL,
                      highlightthickness=1, highlightbackground=BORDER)
        vf.pack(fill="x", padx=10, pady=(0, 6))
        self.verdict_lbl = tk.Label(vf, text="—", bg=PANEL, fg=SUBTEXT,
                                    font=("Segoe UI", 12, "bold"), wraplength=230,
                                    justify="center", pady=10)
        self.verdict_lbl.pack(fill="x")
        self.risk_lbl = tk.Label(vf, text="", bg=PANEL, fg=SUBTEXT,
                                 font=FONT_SMALL, pady=4)
        self.risk_lbl.pack(fill="x")

    def _btn(self, parent, text, bg, fg, cmd, font=None, pady=6, state="normal"):
        return tk.Button(parent, text=text, bg=bg, fg=fg,
                         relief="flat", activebackground=ACCENT, activeforeground=BG,
                         font=font or FONT_BODY, cursor="hand2",
                         pady=pady, state=state, command=cmd)

    def _section(self, parent, text):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", padx=10, pady=(10, 3))
        tk.Label(f, text=text, bg=BG, fg=ACCENT,
                 font=("Segoe UI", 7, "bold")).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(
            side="left", fill="x", expand=True, padx=(6, 0))

    # ── Right panel ──────────────────────────────────────────────────

    def _build_right(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        self.tab_summary = tk.Frame(self.notebook, bg=BG)
        self.tab_lsb = tk.Frame(self.notebook, bg=BG)
        self.tab_hist = tk.Frame(self.notebook, bg=BG)
        self.tab_extract = tk.Frame(self.notebook, bg=BG)
        self.tab_raw = tk.Frame(self.notebook, bg=BG)

        self.notebook.add(self.tab_summary, text="  Summary  ")
        self.notebook.add(self.tab_lsb,     text="  LSB Planes  ")
        self.notebook.add(self.tab_hist,    text="  Histograms  ")
        self.notebook.add(self.tab_extract, text="  Message Extraction  ")
        self.notebook.add(self.tab_raw,     text="  Raw JSON  ")

        self._build_welcome()
        self._build_raw_tab()
        self._placeholder(
            self.tab_lsb,     "LSB planes will appear after analysis")
        self._placeholder(
            self.tab_hist,    "Histograms will appear after analysis")
        self._placeholder(self.tab_extract,
                          "Message extraction will appear after analysis")

    def _placeholder(self, tab, msg):
        for w in tab.winfo_children():
            w.destroy()
        tk.Label(tab, text=msg, bg=BG, fg=SUBTEXT,
                 font=FONT_BODY).pack(expand=True)

    def _build_welcome(self):
        for w in self.tab_summary.winfo_children():
            w.destroy()
        wf = tk.Frame(self.tab_summary, bg=BG)
        wf.pack(fill="both", expand=True)
        tk.Label(wf, text="⬡", bg=BG, fg=ACCENT,
                 font=("Segoe UI", 48)).pack(pady=(50, 8))
        tk.Label(wf, text="Add images then press  ▶ Analyse All Images",
                 bg=BG, fg=TEXT, font=("Segoe UI", 13)).pack()
        tk.Label(wf, text="PNG · JPEG · BMP · TIFF · WEBP  |  Multiple images supported",
                 bg=BG, fg=SUBTEXT, font=FONT_BODY).pack(pady=4)

        features = [
            ("LSB Analysis",       "Bit-level randomness across colour channels"),
            ("Chi-Square Test",    "Statistical pixel-pair equalisation"),
            ("RS Analysis",        "Payload fraction estimation"),
            ("Histogram Analysis", "Comb pattern detection"),
            ("Message Extraction", "Attempts to decode hidden ASCII text"),
            ("File/Video Detection", "Detects embedded files in LSB data"),
            ("Batch PDF Report",   "Synopsis, risk factors, aggregate summary"),
        ]
        ff = tk.Frame(wf, bg=PANEL, highlightthickness=1,
                      highlightbackground=BORDER)
        ff.pack(padx=40, pady=16, fill="x")
        for name, desc in features:
            row = tk.Frame(ff, bg=PANEL)
            row.pack(fill="x", padx=14, pady=3)
            tk.Label(row, text=f"◈  {name}", bg=PANEL, fg=ACCENT,
                     font=("Segoe UI", 9, "bold"), width=22, anchor="w").pack(side="left")
            tk.Label(row, text=desc, bg=PANEL, fg=SUBTEXT,
                     font=FONT_SMALL, anchor="w").pack(side="left")

    def _build_raw_tab(self):
        f = tk.Frame(self.tab_raw, bg=BG)
        f.pack(fill="both", expand=True)
        tk.Label(f, text="Raw Analysis Output (selected image)",
                 bg=BG, fg=ACCENT, font=FONT_H2).pack(anchor="w", padx=16, pady=(10, 4))
        tf = tk.Frame(f, bg=PANEL, highlightthickness=1,
                      highlightbackground=BORDER)
        tf.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        self.raw_text = tk.Text(tf, bg=PANEL, fg=TEXT, font=FONT_MONO,
                                relief="flat", bd=0, state="disabled", wrap="none")
        sby = ttk.Scrollbar(tf, orient="vertical",
                            command=self.raw_text.yview)
        sbx = ttk.Scrollbar(tf, orient="horizontal",
                            command=self.raw_text.xview)
        self.raw_text.configure(yscrollcommand=sby.set, xscrollcommand=sbx.set)
        sby.pack(side="right",  fill="y")
        sbx.pack(side="bottom", fill="x")
        self.raw_text.pack(fill="both", expand=True)

    # ── Actions ──────────────────────────────────────────────────────

    def _add_images(self):
        paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image Files",
                        "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
                       ("All Files", "*.*")])
        if not paths:
            return
        added = 0
        existing = {img["path"] for img in self.images}
        for p in paths:
            if p not in existing:
                self.images.append({
                    "path": p, "analyzer": None,
                    "results": {}, "overall": {}, "extraction": None,
                })
                iid = self.img_list.insert("", "end",
                                           text=os.path.basename(p),
                                           values=("—", "—"))
                self.images[-1]["iid"] = iid
                added += 1
        if added:
            self.run_btn.configure(state="normal")
            self._set_status(
                f"{len(self.images)} image(s) loaded — ready to analyse")

    def _remove_selected(self):
        sel = self.img_list.selection()
        if not sel:
            return
        iid = sel[0]
        self.images = [img for img in self.images if img.get("iid") != iid]
        self.img_list.delete(iid)
        if not self.images:
            self.run_btn.configure(state="disabled")
            self.export_btn.configure(state="disabled")
            self._build_welcome()
        self._set_status(f"{len(self.images)} image(s) remaining")

    def _clear_all(self):
        self.images = []
        self.current_idx = None
        for iid in self.img_list.get_children():
            self.img_list.delete(iid)
        self.run_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.verdict_lbl.configure(text="—", fg=SUBTEXT)
        self.risk_lbl.configure(text="")
        self._build_welcome()
        self._placeholder(
            self.tab_lsb,     "LSB planes will appear after analysis")
        self._placeholder(
            self.tab_hist,    "Histograms will appear after analysis")
        self._placeholder(self.tab_extract,
                          "Message extraction will appear after analysis")
        self.raw_text.configure(state="normal")
        self.raw_text.delete("1.0", "end")
        self.raw_text.configure(state="disabled")
        self._set_status("Ready — add images to begin")

    def _on_list_select(self, event):
        sel = self.img_list.selection()
        if not sel:
            return
        iid = sel[0]
        for i, img in enumerate(self.images):
            if img.get("iid") == iid:
                self.current_idx = i
                if img["overall"]:
                    self._show_image_result(i)
                break

    def _set_status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    # ── Analysis ─────────────────────────────────────────────────────

    def _run_all(self):
        if not self.images:
            return
        self.run_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")
        self.progress.start(10)
        self._set_status("Analysing…")
        threading.Thread(target=self._analysis_worker, daemon=True).start()

    def _analysis_worker(self):
        total = len(self.images)
        for i, img in enumerate(self.images):
            try:
                self.after(0, lambda i=i: self._set_status(
                    f"Analysing image {i+1}/{total}: "
                    f"{os.path.basename(self.images[i]['path'])}"))
                az = StegoAnalyzer(img["path"])
                overall = az.run_all()
                # Use cached extraction from overall_assessment if available
                extraction = getattr(
                    az, '_extraction_cache', None) or az.extract_lsb_message(max_chars=600)
                img["analyzer"] = az
                img["results"] = az.results
                img["overall"] = overall
                img["extraction"] = extraction
                self.after(0, lambda i=i: self._update_list_row(i))
            except Exception as e:
                err = str(e)
                self.after(0, lambda e=err, i=i: self._set_status(
                    f"Error on image {i+1}: {e}"))
        self.after(0, self._on_all_done)

    def _update_list_row(self, idx):
        img = self.images[idx]
        ov = img["overall"]
        level = ov.get("risk_level", "?")
        payload = ov.get("estimated_payload_pct", 0)
        col = {"HIGH": DETECTED, "MEDIUM": ACCENT3,
               "LOW": CLEAN}.get(level, SUBTEXT)
        iid = img.get("iid")
        if iid:
            self.img_list.item(iid, values=(level, f"{payload:.1f}%"))
            self.img_list.tag_configure(level, foreground=col)
            # select first completed image
            if self.current_idx is None:
                self.current_idx = idx
                self.img_list.selection_set(iid)
                self._show_image_result(idx)

    def _on_all_done(self):
        self.progress.stop()
        self.run_btn.configure(state="normal")
        self.export_btn.configure(state="normal")
        done = sum(1 for img in self.images if img["overall"])
        high = sum(1 for img in self.images
                   if img["overall"].get("risk_level") == "HIGH")
        self._set_status(
            f"Analysis complete — {done}/{len(self.images)} images  ·  "
            f"{high} HIGH risk")
        # Show currently selected
        if self.current_idx is not None:
            self._show_image_result(self.current_idx)

    # ── Display selected image ────────────────────────────────────────

    def _show_image_result(self, idx):
        img = self.images[idx]
        if not img["overall"]:
            return
        ov = img["overall"]
        level = ov["risk_level"]
        verdict = ov["verdict"]
        score = ov["risk_score"]
        col = {"HIGH": DETECTED, "MEDIUM": ACCENT3,
               "LOW": CLEAN}.get(level, SUBTEXT)
        self.verdict_lbl.configure(
            text=f"{level} RISK\n{verdict}", fg=col)
        pos = ov["positive_count"]
        tot = ov["total_tests"]
        payload = ov.get("estimated_payload_pct", 0)
        self.risk_lbl.configure(
            text=f"Score: {score*100:.0f}%  ·  {pos}/{tot} positive\n"
            f"Payload: {payload:.1f}%", fg=SUBTEXT)

        self._populate_summary(idx)
        self._populate_lsb_tab(idx)
        self._populate_hist_tab(idx)
        self._populate_extract_tab(idx)
        self._populate_raw_tab(idx)

    # ── Summary tab ──────────────────────────────────────────────────

    def _populate_summary(self, idx):
        img = self.images[idx]
        ov = img["overall"]
        for w in self.tab_summary.winfo_children():
            w.destroy()

        canvas = tk.Canvas(self.tab_summary, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(self.tab_summary, orient="vertical",
                           command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)
        inner = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win, width=canvas.winfo_width()))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        level = ov["risk_level"]
        col = {"HIGH": DETECTED, "MEDIUM": ACCENT3,
               "LOW": CLEAN}.get(level, SUBTEXT)
        fname = os.path.basename(img["path"])

        # Top bar
        tk.Frame(inner, bg=col, height=4).pack(fill="x")
        hdr = tk.Frame(inner, bg=PANEL, highlightthickness=1,
                       highlightbackground=BORDER)
        hdr.pack(fill="x", padx=16, pady=(10, 6))
        tk.Label(hdr, text=f"  {level} RISK", bg=PANEL, fg=col,
                 font=("Segoe UI", 18, "bold")).pack(side="left", padx=10, pady=8)
        tk.Label(hdr, text=ov["verdict"], bg=PANEL, fg=TEXT,
                 font=FONT_BODY).pack(side="left", padx=8)
        tk.Label(hdr, text=fname, bg=PANEL, fg=SUBTEXT,
                 font=FONT_SMALL).pack(side="right", padx=10)

        # Stat boxes
        score = ov["risk_score"]*100
        pos = ov["positive_count"]
        tot = ov["total_tests"]
        payload = ov.get("estimated_payload_pct", 0)
        w_size = ov.get("image_size", ("?", "?"))
        sf = tk.Frame(inner, bg=BG)
        sf.pack(fill="x", padx=16, pady=4)
        for label, val, vc in [
            ("Risk Score",    f"{score:.0f}%",           col),
            ("Tests Positive", f"{pos}/{tot}",            TEXT),
            ("Est. Payload",  f"{payload:.2f}%",         ACCENT3),
            ("Resolution",    f"{w_size[0]}×{w_size[1]}", SUBTEXT),
        ]:
            box = tk.Frame(sf, bg=PANEL,
                           highlightthickness=1, highlightbackground=BORDER)
            box.pack(side="left", padx=(0, 8), pady=4, ipadx=12, ipady=8)
            tk.Label(box, text=val,   bg=PANEL, fg=vc,
                     font=("Segoe UI", 16, "bold")).pack()
            tk.Label(box, text=label, bg=PANEL, fg=SUBTEXT,
                     font=FONT_SMALL).pack()

        # File/Video detection summary (if available)
        file_summary = ov.get("file_detection_summary", "")
        if file_summary and file_summary != "No files or videos detected in LSB data":
            fs_frame = tk.Frame(inner, bg=PANEL, highlightthickness=1,
                                highlightbackground=BORDER)
            fs_frame.pack(fill="x", padx=16, pady=4)
            tk.Label(fs_frame, text="📁 File/Video Detection:",
                     bg=PANEL, fg=ACCENT2, font=("Segoe UI", 9, "bold")).pack(
                side="left", padx=10, pady=6)
            tk.Label(fs_frame, text=file_summary,
                     bg=PANEL, fg=TEXT, font=FONT_MONO).pack(
                side="left", padx=10, pady=6)

        # Risk chart
        try:
            fig = make_risk_figure(ov)
            if fig:
                pil = fig_to_pil(fig)
                tk_img = pil_to_tk(pil, 560, 240)
                self._tk_images.append(tk_img)
                tk.Label(inner, image=tk_img, bg=BG).pack(
                    padx=16, pady=6, anchor="w")
        except Exception:
            pass

        # Per-test cards
        tk.Label(inner, text="Individual Test Results", bg=BG, fg=ACCENT,
                 font=FONT_H2).pack(anchor="w", padx=16, pady=(8, 4))
        cards = tk.Frame(inner, bg=BG)
        cards.pack(fill="x", padx=16, pady=4)

        test_info = [
            ("lsb",         "LSB Analysis",
             lambda r: f"Suspicion: {r.get('avg_correlation', 0):.4f}"),
            ("chi_square",  "Chi-Square Test",
             lambda r: f"p-value: {r.get('average_p_value', 1):.4f}"),
            ("rs_analysis", "RS Analysis",
             lambda r: f"Payload: {r.get('average_payload_estimate', 0)*100:.2f}%"),
            ("histogram",   "Histogram Analysis",
             lambda r: f"Comb: {r.get('avg_comb_score', 0):.3f}"),
            ("file_detection", "File/Video Detection",
             lambda r: f"Files: {len(r.get('file_signatures', []))}  Videos: {len(r.get('video_markers', []))}  Entropy: {r.get('entropy', 0):.2f}"),
        ]
        for key, name, metric_fn in test_info:
            r = img["results"].get(key, {})
            detected = r.get("detected", False)
            sc = DETECTED if detected else CLEAN
            card = tk.Frame(cards, bg=PANEL, highlightthickness=1,
                            highlightbackground=BORDER)
            card.pack(fill="x", pady=3)
            tk.Frame(card, bg=sc, width=4).pack(side="left", fill="y")
            cont = tk.Frame(card, bg=PANEL)
            cont.pack(side="left", fill="both", expand=True, padx=12, pady=8)
            top = tk.Frame(cont, bg=PANEL)
            top.pack(fill="x")
            tk.Label(top, text=name, bg=PANEL, fg=TEXT,
                     font=("Segoe UI", 10, "bold")).pack(side="left")
            tk.Label(top, text="DETECTED" if detected else "CLEAN",
                     bg=PANEL, fg=sc,
                     font=("Segoe UI", 9, "bold")).pack(side="right")
            tk.Label(cont, text=metric_fn(r) if r else "—",
                     bg=PANEL, fg=ACCENT3, font=FONT_MONO, anchor="w").pack(fill="x")
            desc = r.get("description", "")
            if desc:
                tk.Label(cont, text=desc, bg=PANEL, fg=SUBTEXT,
                         font=FONT_SMALL, wraplength=700, anchor="w",
                         justify="left").pack(fill="x")

    # ── LSB tab ──────────────────────────────────────────────────────

    def _populate_lsb_tab(self, idx):
        img = self.images[idx]
        for w in self.tab_lsb.winfo_children():
            w.destroy()
        try:
            fig = make_lsb_figure(img["analyzer"].np_image)
            pil = fig_to_pil(fig)
            tk_img = pil_to_tk(pil, 1100, 380)
            self._tk_images.append(tk_img)
            tk.Label(self.tab_lsb, text="LSB Plane Visualisation",
                     bg=BG, fg=ACCENT, font=FONT_H2).pack(
                         anchor="w", padx=16, pady=(12, 4))
            tk.Label(self.tab_lsb, image=tk_img, bg=BG).pack(padx=16, pady=4)
            lsb = img["results"].get("lsb", {})
            info = tk.Frame(self.tab_lsb, bg=PANEL,
                            highlightthickness=1, highlightbackground=BORDER)
            info.pack(fill="x", padx=16, pady=8)
            for ch, d in lsb.get("channels", {}).items():
                row = tk.Frame(info, bg=PANEL)
                row.pack(fill="x", padx=12, pady=3)
                tk.Label(row, text=f"{ch}:", bg=PANEL, fg=ACCENT,
                         font=("Segoe UI", 9, "bold"), width=8, anchor="w").pack(side="left")
                tk.Label(row,
                         text=(f"Entropy={d['entropy']:.4f}  "
                               f"Ones={d['ones_ratio']:.4f}  "
                               f"Avg Corr={d['avg_correlation']:.4f}"),
                         bg=PANEL, fg=TEXT, font=FONT_MONO).pack(side="left")
        except Exception as e:
            tk.Label(self.tab_lsb, text=f"Error: {e}",
                     bg=BG, fg=ACCENT2, font=FONT_BODY).pack(expand=True)

    # ── Histogram tab ────────────────────────────────────────────────

    def _populate_hist_tab(self, idx):
        img = self.images[idx]
        for w in self.tab_hist.winfo_children():
            w.destroy()
        try:
            fig = make_histogram_figure(img["results"])
            pil = fig_to_pil(fig)
            tk_img = pil_to_tk(pil, 1100, 320)
            self._tk_images.append(tk_img)
            tk.Label(self.tab_hist, text="Pixel-Value Histogram Analysis",
                     bg=BG, fg=ACCENT, font=FONT_H2).pack(
                         anchor="w", padx=16, pady=(12, 4))
            tk.Label(self.tab_hist, image=tk_img, bg=BG).pack(padx=16, pady=4)
            h = img["results"].get("histogram", {})
            info = tk.Frame(self.tab_hist, bg=PANEL,
                            highlightthickness=1, highlightbackground=BORDER)
            info.pack(fill="x", padx=16, pady=8)
            tk.Label(info,
                     text=(f"Avg comb score: {h.get('avg_comb_score', 0):.4f}   "
                           f"PoV symmetry std: {h.get('avg_pov_symmetry_std', 0):.4f}   "
                           f"Detected: {h.get('detected', False)}"),
                     bg=PANEL, fg=TEXT, font=FONT_MONO).pack(padx=12, pady=6)
        except Exception as e:
            tk.Label(self.tab_hist, text=f"Error: {e}",
                     bg=BG, fg=ACCENT2, font=FONT_BODY).pack(expand=True)

    # ── Message Extraction tab ───────────────────────────────────────

    def _populate_extract_tab(self, idx):
        img = self.images[idx]
        ext = img.get("extraction") or {}
        for w in self.tab_extract.winfo_children():
            w.destroy()

        tk.Label(self.tab_extract, text="LSB Message Extraction",
                 bg=BG, fg=ACCENT, font=FONT_H2).pack(
                     anchor="w", padx=16, pady=(12, 4))

        if not ext.get("extraction_attempted"):
            tk.Label(self.tab_extract,
                     text="No extraction data available.",
                     bg=BG, fg=SUBTEXT, font=FONT_BODY).pack(padx=16)
            return

        best = ext.get("best_result", {})
        best_ch = ext.get("best_channel", "—")
        likely = best.get("likely_message", False)
        ratio = best.get("printable_ratio", 0)
        length = best.get("length", 0)

        # Status banner
        sc = DETECTED if likely else CLEAN
        summary = tk.Frame(self.tab_extract, bg=PANEL,
                           highlightthickness=1, highlightbackground=BORDER)
        summary.pack(fill="x", padx=16, pady=(0, 8))
        tk.Frame(summary, bg=sc, width=4).pack(side="left", fill="y")
        sf = tk.Frame(summary, bg=PANEL)
        sf.pack(side="left", fill="both", expand=True, padx=12, pady=10)
        tk.Label(sf,
                 text="HIDDEN MESSAGE DETECTED" if likely else "NO READABLE MESSAGE FOUND",
                 bg=PANEL, fg=sc,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(sf,
                 text=(f"Best channel: {best_ch}   "
                       f"Printable ratio: {ratio:.1%}   "
                       f"Characters: {length}"),
                 bg=PANEL, fg=SUBTEXT, font=FONT_SMALL).pack(anchor="w")

        if not likely:
            tk.Label(self.tab_extract,
                     text=("The extracted bit-stream does not contain readable text.\n"
                           "This indicates no text-based LSB message is present, "
                           "or the message uses a non-ASCII encoding."),
                     bg=BG, fg=SUBTEXT, font=FONT_BODY,
                     wraplength=700, justify="left").pack(
                         anchor="w", padx=16, pady=8)

        # Per-channel results table
        tk.Label(self.tab_extract, text="Per-Channel Results",
                 bg=BG, fg=SUBTEXT,
                 font=("Segoe UI", 9, "bold")).pack(
                     anchor="w", padx=16, pady=(4, 2))

        tbl = tk.Frame(self.tab_extract, bg=PANEL,
                       highlightthickness=1, highlightbackground=BORDER)
        tbl.pack(fill="x", padx=16, pady=(0, 10))

        headers = ["Channel", "Length",
                   "Printable %", "Message Found", "Preview"]
        for ci, h in enumerate(headers):
            tk.Label(tbl, text=h, bg=PANEL, fg=ACCENT,
                     font=("Segoe UI", 8, "bold"),
                     width=16 if ci < 4 else 30,
                     anchor="center").grid(row=0, column=ci, padx=4, pady=4)

        for ri, (ch_name, ch_data) in enumerate(
                ext.get("channels", {}).items(), start=1):
            bg_row = BG if ri % 2 == 0 else PANEL
            lm = ch_data.get("likely_message", False)
            preview = ch_data.get("text", "")[:40].replace("\n", " ")
            row_vals = [
                ch_name,
                str(ch_data.get("length", 0)),
                f"{ch_data.get('printable_ratio', 0):.1%}",
                "YES" if lm else "NO",
                preview,
            ]
            for ci, val in enumerate(row_vals):
                fc = DETECTED if (
                    ci == 3 and lm) else CLEAN if ci == 3 else TEXT
                tk.Label(tbl, text=val, bg=bg_row, fg=fc,
                         font=FONT_MONO if ci in (0, 4) else FONT_SMALL,
                         width=16 if ci < 4 else 30,
                         anchor="center" if ci < 4 else "w").grid(
                             row=ri, column=ci, padx=4, pady=2)

        # Extracted text display
        tk.Label(self.tab_extract,
                 text=f"Extracted Content — {best_ch} Channel",
                 bg=BG, fg=SUBTEXT,
                 font=("Segoe UI", 9, "bold")).pack(
                     anchor="w", padx=16, pady=(4, 2))

        tf = tk.Frame(self.tab_extract, bg=PANEL,
                      highlightthickness=1, highlightbackground=BORDER)
        tf.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        txt_fg = ACCENT if likely else SUBTEXT
        txt = tk.Text(tf, bg=PANEL, fg=txt_fg, font=FONT_MONO,
                      relief="flat", bd=0, wrap="word")
        sb = ttk.Scrollbar(tf, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        extracted = best.get("text", "(no readable content)")
        txt.insert("1.0", extracted if extracted else "(no readable content)")
        txt.configure(state="disabled")

    # ── Raw JSON tab ─────────────────────────────────────────────────

    def _populate_raw_tab(self, idx):
        img = self.images[idx]
        output = {
            "image_path":    img["path"],
            "analysis_time": datetime.now().isoformat(),
            "overall":       img["overall"],
            "extraction_summary": {
                k: {ik: iv for ik, iv in v.items() if ik != "text"}
                for k, v in (img["extraction"] or {}).get("channels", {}).items()
            },
            "details": {
                k: {ik: iv for ik, iv in v.items()
                    if ik not in ("magnitude_spectrum", "histogram")}
                for k, v in img["results"].items()
            },
        }
        text = json.dumps(output, indent=2, default=str)
        self.raw_text.configure(state="normal")
        self.raw_text.delete("1.0", "end")
        self.raw_text.insert("1.0", text)
        self.raw_text.configure(state="disabled")

    # ── PDF Export ───────────────────────────────────────────────────

    def _export_pdf(self):
        done = [img for img in self.images if img["overall"]]
        if not done:
            messagebox.showwarning("No results", "Run analysis first.")
            return
        default = (f"stego_batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                   if len(done) > 1 else
                   os.path.splitext(os.path.basename(done[0]["path"]))[0]
                   + "_stego_report.pdf")
        path = filedialog.asksaveasfilename(
            title="Save PDF Report",
            defaultextension=".pdf",
            initialfile=default,
            filetypes=[("PDF files", "*.pdf"), ("All Files", "*.*")])
        if not path:
            return
        try:
            batch = [{
                "image_path":       img["path"],
                "analysis_results": img["results"],
                "overall":          img["overall"],
                "extraction":       img.get("extraction"),
            } for img in done]
            generate_batch_report(batch, path)
            messagebox.showinfo("Report Saved",
                                f"PDF report saved:\n{path}\n\n{len(done)} image(s) included.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Could not save PDF:\n{e}")


# ── entry ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = StegoDetectorApp()
    app.mainloop()
