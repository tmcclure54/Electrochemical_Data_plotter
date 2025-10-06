import os
import csv
import re
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns

# tkbootstrap
import ttkbootstrap as tb
from ttkbootstrap.constants import *

# --------------------------- Parsing Helpers ---------------------------

COLUMN_SYNONYMS_V = [
    "potential", "voltage", "ewe", "wevsre", "e_vs_ref", "e_(v)", "e(v)", "e/v", "e", "v"
]
COLUMN_SYNONYMS_I = [
    "current", "i", "i_(a)", "i(a)", "i/a", "rawc", "current(a)", "current (a)"
]

def clean_column_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r'[^a-z0-9_\-\s\(\)/]', '', name)
    name = name.replace(' ', '')
    return name

def sniff_delimiter_and_header(file_path):
    with open(file_path, 'r', errors='ignore', newline='') as f:
        sample = f.read(8192)
    delimiter = ',' if sample.count(',') >= sample.count('\t') else '\t'
    lines = sample.splitlines()
    header_row = 0
    for idx, line in enumerate(lines[:200]):
        toks = re.split(rf"{re.escape(delimiter)}", line.strip())
        alpha_tokens = sum(1 for t in toks if re.search(r'[A-Za-z]', t))
        if alpha_tokens >= 2 and 2 <= len(toks) <= 50:
            header_row = idx
            break
    return delimiter, header_row

def try_read_table(file_path):
    delimiter, header_row = sniff_delimiter_and_header(file_path)
    for skip in (header_row, 0, 1, 2, 5, 10, 20, 30):
        try:
            df = pd.read_csv(file_path, sep=delimiter, engine='python', skiprows=skip)
            if df.shape[1] < 2:
                continue
            df = df.dropna(axis=1, how='all')
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    df = pd.read_csv(file_path, sep=None, engine='python')
    df = df.dropna(axis=1, how='all')
    return df

def find_voltage_current_columns(df):
    original_columns = list(df.columns)
    cleaned = [clean_column_name(c) for c in original_columns]

    def first_index_with(keys):
        for pri in keys:
            for j, name in enumerate(cleaned):
                if pri in name:
                    return j
        return None

    v_idx = first_index_with(COLUMN_SYNONYMS_V)
    i_idx = first_index_with(COLUMN_SYNONYMS_I)

    if v_idx is None:
        for j, n in enumerate(cleaned):
            if n in ('ewev', 'potentialv', 'voltagev', 'v'):
                v_idx = j; break
    if i_idx is None:
        for j, n in enumerate(cleaned):
            if n in ('ia', 'currenta', 'i'):
                i_idx = j; break

    if v_idx is None or i_idx is None:
        raise ValueError("Could not find voltage/current columns.\n"
                         f"Columns seen: {original_columns}")
    return v_idx, i_idx


def parse_data_file(file_path):
    df = try_read_table(file_path)
    df = df.apply(lambda s: pd.to_numeric(s, errors='ignore'))
    v_idx, i_idx = find_voltage_current_columns(df)
    V = pd.to_numeric(df.iloc[:, v_idx], errors='coerce')
    I = pd.to_numeric(df.iloc[:, i_idx], errors='coerce')
    mask = V.notna() & I.notna()
    return V[mask].reset_index(drop=True), I[mask].reset_index(drop=True)
    return V[mask].reset_index(drop=True), I[mask].reset_index(drop=True)

# --------------------------- GUI App ---------------------------

class CVPlotterApp:

    def _resolved_ylabel(self, base_label: str, y_unit: str) -> str:
        """Return the final y-axis label. If the user already specified units,
        keep them; otherwise append [unit]."""
        if base_label is None:
            base_label = ""
        # Normalize to detect explicit units like "Current (mA)" or "Current Density (mA/cm²)"
        s = base_label.lower().replace(" ", "")
        has_units = ("current(" in s) or ("currentdensity(" in s) or ("[" in s and "]" in s)
        return base_label if has_units else f"{base_label} [{y_unit}]"

    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title("CV Multi-File Plotter (tkbootstrap • iR • Points • Cursor • Zoom)")

        self.files, self.legend_vars, self.area_vars, self.r_vars = [], [], [], []

        # Point-selection state
        self.select_mode = tk.BooleanVar(value=False)
        self.selected_points = []  # [{'label','x','y','artist'}]
        self._mpl_click_cid = None
        self._mpl_move_cid = None
        self._px_pick_tol = 10

        # Units for cursor/status
        self._x_unit = "V"
        self._y_unit_display = ""  # set after plotting

        # ---------- Top controls ----------
        top = tb.Frame(root, padding=10); top.pack(fill=X)
        tb.Label(top, text="X-axis Label:").grid(row=0, column=0, sticky=W)
        self.x_label_var = tk.StringVar(value="Potential (V)")
        tb.Entry(top, textvariable=self.x_label_var, width=22).grid(row=0, column=1, padx=6)
        tb.Label(top, text="Y-axis Label:").grid(row=0, column=2, sticky=W)
        self.y_label_var = tk.StringVar(value="Current (mA)  or  Current Density (mA/cm²)")
        tb.Entry(top, textvariable=self.y_label_var, width=36).grid(row=0, column=3, padx=6, columnspan=2)
        tb.Label(top, text="Title:").grid(row=0, column=5, sticky=W)
        self.title_var = tk.StringVar(value="Cyclic Voltammograms")
        tb.Entry(top, textvariable=self.title_var, width=28).grid(row=0, column=6, padx=6)

        # ---------- EC settings ----------
        ec = tb.Labelframe(root, text="Electrochemistry Settings (Globals / Defaults)", padding=10)
        ec.pack(fill=X, padx=10, pady=(0,10))
        tb.Label(ec, text="Global Area (cm²):").grid(row=0, column=0, sticky=W)
        self.global_area_var = tk.StringVar(value="")
        tb.Entry(ec, textvariable=self.global_area_var, width=10).grid(row=0, column=1, padx=6)
        tb.Label(ec, text="Global R (Ω):").grid(row=0, column=2, sticky=W)
        self.global_r_var = tk.StringVar(value="")
        tb.Entry(ec, textvariable=self.global_r_var, width=10).grid(row=0, column=3, padx=6)
        self.apply_ir_var = tk.BooleanVar(value=False)
        tb.Checkbutton(ec, text="Apply iR correction (E − i·R)",
                       variable=self.apply_ir_var, bootstyle="round-toggle").grid(row=0, column=4, padx=10)
        tb.Label(ec, text="Reference shift (V):").grid(row=0, column=5, sticky=W)
        self.ref_shift_var = tk.StringVar(value="0.0")
        tb.Entry(ec, textvariable=self.ref_shift_var, width=10).grid(row=0, column=6, padx=6)
        self.invert_x_var = tk.BooleanVar(value=True)
        tb.Checkbutton(ec, text="Invert X-axis", variable=self.invert_x_var,
                       bootstyle="round-toggle").grid(row=0, column=7, padx=10)
        tb.Label(ec, text="Current units:").grid(row=1, column=0, sticky=W, pady=6)
        self.i_units_var = tk.StringVar(value="mA")
        tb.Combobox(ec, textvariable=self.i_units_var, values=["A", "mA", "µA"], width=6,
                    state='readonly').grid(row=1, column=1, sticky=W)
        self.flip_sign_var = tk.BooleanVar(value=True)
        tb.Checkbutton(ec, text="Flip current sign (plot −I)", variable=self.flip_sign_var,
                       bootstyle="round-toggle").grid(row=1, column=2, columnspan=2, sticky=W)

        # ---------- Files list ----------
        files_box = tb.Labelframe(root, text="Files", padding=6)
        files_box.pack(fill=BOTH, expand=True, padx=10, pady=(0,10))
        self.canvas_files = tk.Canvas(files_box, highlightthickness=0)
        self.scrollbar = tb.Scrollbar(files_box, orient=VERTICAL, command=self.canvas_files.yview, bootstyle=ROUND)
        self.scrollable_frame = tb.Frame(self.canvas_files)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas_files.configure(scrollregion=self.canvas_files.bbox("all")))
        self.canvas_files.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas_files.configure(yscrollcommand=self.scrollbar.set)
        self.canvas_files.pack(side=LEFT, fill=BOTH, expand=True)
        self.scrollbar.pack(side=RIGHT, fill=Y)

        # ---------- Buttons row ----------
        btns = tb.Frame(root, padding=10); btns.pack(fill=X)
        tb.Button(btns, text="Add Files", command=self.add_files, bootstyle=PRIMARY).pack(side=LEFT, padx=4)
        tb.Button(btns, text="Clear Files", command=self.clear_files, bootstyle=SECONDARY).pack(side=LEFT, padx=4)
        tb.Button(btns, text="Plot CVs", command=self.plot_all, bootstyle=SUCCESS).pack(side=LEFT, padx=8)

        # Zoom / Pan / Home controls
        self.zoom_btn = tb.Button(btns, text="Zoom (Z)", command=self.toggle_zoom, bootstyle=INFO)
        self.zoom_btn.pack(side=LEFT, padx=6)
        self.pan_btn = tb.Button(btns, text="Pan (P)", command=self.toggle_pan, bootstyle=INFO)
        self.pan_btn.pack(side=LEFT, padx=4)
        tb.Button(btns, text="Home (H)", command=lambda: (self.toolbar.home(), self._sync_zoom_pan_styles()),
                  bootstyle=SECONDARY).pack(side=LEFT, padx=4)

        # Point-selection controls
        tb.Checkbutton(btns, text="Enable Point Selection",
                       variable=self.select_mode, command=self.toggle_point_selection,
                       bootstyle="square-toggle").pack(side=LEFT, padx=12)
        tb.Button(btns, text="Undo Last", command=self.undo_last_point, bootstyle=INFO).pack(side=LEFT, padx=4)
        tb.Button(btns, text="Clear Points", command=self.clear_points, bootstyle=WARNING).pack(side=LEFT, padx=4)
        tb.Button(btns, text="Export Points…", command=self.export_points_csv, bootstyle=INFO).pack(side=LEFT, padx=8)

        self.points_count_var = tk.StringVar(value="Points: 0")
        tb.Label(btns, textvariable=self.points_count_var).pack(side=RIGHT, padx=6)
        tb.Button(btns, text="Save Figure…", command=self.save_figure, bootstyle=DANGER).pack(side=RIGHT, padx=8)

        # ---------- Figure & toolbar ----------
        self.fig, self.ax = plt.subplots(figsize=(8.8, 6.6))

        # On-plot HUD (top-right of axes)
        self.hud_text = self.ax.text(
            0.99, 0.98, "x: —   y: —",
            transform=self.ax.transAxes,
            ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
            zorder=10
        )
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(fill=BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas_fig, root)
        self.toolbar.update()
        self.canvas_fig.get_tk_widget().pack()

        # --- Cursor status bar (always-on live coords) ---
        status = tb.Frame(root, padding=(10, 6, 10, 0)); status.pack(fill=X)
        tb.Label(status, text="Cursor:", width=8).pack(side=LEFT)
        self.cursor_var = tk.StringVar(value="x: —   y: —")
        tb.Label(status, textvariable=self.cursor_var, bootstyle=SECONDARY).pack(side=LEFT)

        # ---------- Progress bar ----------
        pb_frame = tb.Frame(root, padding=(10, 2, 10, 10)); pb_frame.pack(fill=X)
        self.pb = tb.Progressbar(pb_frame, mode='determinate', maximum=100, bootstyle=STRIPED)
        self.pb.pack(fill=X)

        # ---------- Aesthetics (no grid lines) ----------
        sns.set_context("talk")
        sns.set_style("white")  # clean background, no grid
        plt.rcParams.update({
            "font.family": "Arial",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.direction": 'in',
            "ytick.direction": 'in',
            "legend.fontsize": 11,
            "legend.title_fontsize": 12,
            "savefig.dpi": 300,
            "figure.dpi": 110,
            "axes.grid": False,
        })

        # Connect mouse-move for status bar
        self._mpl_move_cid = self.canvas_fig.mpl_connect('motion_notify_event', self._on_motion)

        # Keybindings for zoom/pan/home
        self.root.bind('<KeyPress-z>', lambda e: self.toggle_zoom())
        self.root.bind('<KeyPress-p>', lambda e: self.toggle_pan())
        self.root.bind('<KeyPress-h>', lambda e: (self.toolbar.home(), self._sync_zoom_pan_styles()))

        # Minimum size
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_width(), self.root.winfo_height())

    # --------------------------- File controls ---------------------------

    def add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV & TXT files", "*.csv *.txt"), ("All files", "*.*")])
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self._add_file_entry(p)

    def _add_file_entry(self, filepath):
        row = tb.Frame(self.scrollable_frame, padding=4); row.pack(fill=X, expand=True)
        tb.Label(row, text=os.path.basename(filepath), width=32, anchor="w").pack(side=LEFT)
        var_label = tk.StringVar(value=os.path.basename(filepath))
        tb.Label(row, text="Label:").pack(side=LEFT, padx=(10,0))
        tb.Entry(row, textvariable=var_label, width=24).pack(side=LEFT, padx=4)
        self.legend_vars.append(var_label)
        var_area = tk.StringVar(value="")
        tb.Label(row, text="Area (cm²):").pack(side=LEFT, padx=(10,0))
        tb.Entry(row, textvariable=var_area, width=9).pack(side=LEFT, padx=4)
        self.area_vars.append(var_area)
        var_r = tk.StringVar(value="")
        tb.Label(row, text="R (Ω):").pack(side=LEFT, padx=(10,0))
        tb.Entry(row, textvariable=var_r, width=9).pack(side=LEFT, padx=4)
        self.r_vars.append(var_r)

    def clear_files(self):
        self.files.clear(); self.legend_vars.clear(); self.area_vars.clear(); self.r_vars.clear()
        for w in self.scrollable_frame.winfo_children(): w.destroy()
        self.ax.clear(); self.clear_points(); self.canvas_fig.draw()

    # --------------------------- Plotting ---------------------------

    def current_unit_scale(self):
        unit = self.i_units_var.get()
        if unit == "A": return 1.0, "A"
        if unit == "mA": return 1e3, "mA"
        if unit == "µA": return 1e6, "µA"
        return 1e3, "mA"

    def parse_float_or_default(self, s, default=None):
        s = (s or "").strip()
        try:
            return float(s) if s != "" else default
        except Exception:
            return default

    def plot_all(self):
        if not self.files:
            messagebox.showwarning("No files", "Please add at least one file.")
            return

        self.ax.clear()
        self.clear_points(redraw=False)

        colors = sns.color_palette("tab10", max(10, len(self.files)))

        global_area = self.parse_float_or_default(self.global_area_var.get(), None)
        global_r = self.parse_float_or_default(self.global_r_var.get(), None)
        apply_ir = self.apply_ir_var.get()
        ref_shift = self.parse_float_or_default(self.ref_shift_var.get(), 0.0)
        flip_sign = self.flip_sign_var.get()
        i_scale, i_unit = self.current_unit_scale()

        n = len(self.files)
        self.pb['value'] = 0; step = 100.0 / max(1, n)
        y_label_unit = i_unit  # will be updated if density is used

        for idx, path in enumerate(self.files):
            try:
                V, I = parse_data_file(path)
                I = I.astype(float); V = V.astype(float)
                R_local = self.parse_float_or_default(self.r_vars[idx].get(), global_r)
                Vcorr = V - I * R_local if (apply_ir and R_local is not None) else V.copy()
                if ref_shift: Vcorr = Vcorr + ref_shift
                I_plot = I * i_scale * (-1 if flip_sign else 1)
                area_local = self.parse_float_or_default(self.area_vars[idx].get(), global_area)
                label_str = self.legend_vars[idx].get().strip() or os.path.basename(path)
                if area_local and area_local > 0:
                    Y = I_plot / area_local
                    y_label_unit = f"{i_unit}/cm²"
                    label = f"{label_str} (density)"
                else:
                    Y = I_plot
                    y_label_unit = f"{i_unit}"
                    label = label_str
                self.ax.plot(Vcorr, Y, label=label, color=colors[idx % len(colors)], linewidth=2)
            except Exception as e:
                messagebox.showerror("Parse Error", f"Error processing {os.path.basename(path)}:\n{e}")
            self.pb['value'] += step; self.root.update_idletasks()

        self.ax.set_xlabel(self.x_label_var.get())
        ylab = self.y_label_var.get()
        self.ax.set_ylabel(self._resolved_ylabel(ylab, y_label_unit))
        self._y_unit_display = y_label_unit


        self.ax.set_title(self.title_var.get())
        if self.invert_x_var.get(): self.ax.invert_xaxis()
        self.ax.legend(loc='best', frameon=True)
        self.ax.grid(False)
        self.fig.tight_layout()
        self.canvas_fig.draw()
        self.pb['value'] = 100

        # Ensure toolbar also shows coords (bottom-right of toolbar)
        self._install_format_coord()

    # --------------------------- Save ---------------------------

    def save_figure(self):
        if self.fig is None:
            messagebox.showwarning("No figure", "Nothing to save yet. Plot first.")
            return
        fpath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")]
        )
        if not fpath:
            return
        try:
            hud_vis = None
            if hasattr(self, "hud_text"):
                hud_vis = self.hud_text.get_visible()
                self.hud_text.set_visible(False)  # hide HUD for export
            self.fig.savefig(fpath, bbox_inches='tight')
            if hasattr(self, "hud_text"):
                self.hud_text.set_visible(hud_vis)  # restore after export
            messagebox.showinfo("Saved", f"Figure saved to:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save figure:\n{e}")

    # --------------------------- Point selection ---------------------------

    def toggle_point_selection(self):
        if self.select_mode.get():
            if self._mpl_click_cid is None:
                self._mpl_click_cid = self.canvas_fig.mpl_connect('button_press_event', self._on_click)
        else:
            if self._mpl_click_cid is not None:
                self.canvas_fig.mpl_disconnect(self._mpl_click_cid)
                self._mpl_click_cid = None

    def _on_click(self, event):
        if event.inaxes != self.ax: return
        # Allow adding even while zoom/pan active; comment next line to block
        # if self.toolbar.mode != '': return
        if event.button == 1:
            snap = self._snap_to_nearest_curve(event)
            if snap is None: return
            x, y, label = snap
            artist = self.ax.scatter([x], [y], s=50, zorder=5)
            self.selected_points.append({'label': label, 'x': float(x), 'y': float(y), 'artist': artist})
            self._update_points_count(); self.canvas_fig.draw_idle()
        elif event.button == 3:
            if not self.selected_points: return
            idx = self._nearest_selected_point_index(event)
            if idx is None: return
            pt = self.selected_points.pop(idx)
            try: pt['artist'].remove()
            except Exception: pass
            self._update_points_count(); self.canvas_fig.draw_idle()

    def _snap_to_nearest_curve(self, event):
        lines = self.ax.get_lines()
        if not lines: return None
        ex, ey = event.x, event.y
        best = (None, None, None, float('inf'))
        for line in lines:
            xdata = line.get_xdata(orig=False)
            ydata = line.get_ydata(orig=False)
            if len(xdata) == 0: continue
            xy = np.column_stack([xdata, ydata])
            disp = self.ax.transData.transform(xy)
            d2 = (disp[:,0] - ex)**2 + (disp[:,1] - ey)**2
            j = int(np.argmin(d2)); dist = float(np.sqrt(d2[j]))
            if dist < best[3]:
                best = (xdata[j], ydata[j], line.get_label(), dist)
        x, y, label, _ = best
        return (x, y, label)

    def _nearest_selected_point_index(self, event):
        if not self.selected_points: return None
        ex, ey = event.x, event.y
        best_idx, best_dist = None, float('inf')
        for i, pt in enumerate(self.selected_points):
            px, py = self.ax.transData.transform((pt['x'], pt['y']))
            dist = np.hypot(px - ex, py - ey)
            if dist < best_dist:
                best_dist = dist; best_idx = i
        return best_idx if best_dist <= self._px_pick_tol else None

    def undo_last_point(self):
        if not self.selected_points: return
        pt = self.selected_points.pop()
        try: pt['artist'].remove()
        except Exception: pass
        self._update_points_count(); self.canvas_fig.draw_idle()

    def clear_points(self, redraw=True):
        for pt in self.selected_points:
            try: pt['artist'].remove()
            except Exception: pass
        self.selected_points.clear()
        self._update_points_count()
        if redraw: self.canvas_fig.draw_idle()

    def _update_points_count(self):
        self.points_count_var.set(f"Points: {len(self.selected_points)}")

    # --------------------------- Cursor & Toolbar readout ---------------------------

    def _on_motion(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.cursor_var.set("x: —   y: —")
            # Dim the HUD when outside axes
            if hasattr(self, "hud_text"):
                self.hud_text.set_alpha(0.3)
                self.hud_text.set_text("x: —   y: —")
            return

        x = float(event.xdata);
        y = float(event.ydata)
        yunit = self._y_unit_display or self._infer_y_unit_from_label()
        msg = f"x: {x:.3f} {self._x_unit}   y: {y:.3f} {yunit}"
        self.cursor_var.set(msg)
        if hasattr(self, "hud_text"):
            self.hud_text.set_alpha(0.9)
            self.hud_text.set_text(msg)
        # (keep the rest of your method as-is)

    def _install_format_coord(self):
        """Also show coordinates in Matplotlib toolbar using format_coord."""
        yunit = self._y_unit_display or self._infer_y_unit_from_label()
        def fmt(x, y, yu=yunit):
            try:
                return f"x={x:.3f} {self._x_unit}, y={y:.3f} {yu}"
            except Exception:
                return ""
        self.ax.format_coord = fmt
        # Redraw so toolbar picks up the new formatter
        self.canvas_fig.draw_idle()

    def _infer_y_unit_from_label(self):
        lab = self.ax.get_ylabel() or ""
        m = re.search(r'\[(.*?)\]', lab)
        if m: return m.group(1)
        m = re.search(r'\((.*?)\)', lab)
        if m: return m.group(1)
        return ""

    # --------------------------- Export points ---------------------------

    def export_points_csv(self):
        if not self.selected_points:
            messagebox.showinfo("No points", "No points have been selected."); return
        fpath = filedialog.asksaveasfilename(defaultextension=".csv",
                    filetypes=[("CSV", "*.csv")], initialfile="selected_points.csv")
        if not fpath: return
        _, i_unit = self.current_unit_scale()
        # Determine if density based on legend text
        y_is_density = any("(density)" in (h.get_text() if hasattr(h, "get_text") else str(h))
                           for h in (self.ax.legend_.texts if self.ax.legend_ else []))
        y_unit = f"{i_unit}/cm²" if y_is_density else i_unit
        try:
            with open(fpath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["series_label", "x_value", "y_value", "x_unit", "y_unit"])
                for pt in self.selected_points:
                    writer.writerow([pt['label'], pt['x'], pt['y'], "V", y_unit])
            messagebox.showinfo("Exported", f"Selected points exported to:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export CSV:\n{e}")

    # --------------------------- Zoom / Pan helpers ---------------------------

    def toggle_zoom(self):
        # Matplotlib toolbar toggles zoom with .zoom()
        if self.toolbar.mode == 'zoom':
            self.toolbar.zoom()  # turn off
        else:
            if self.toolbar.mode == 'pan': self.toolbar.pan()  # ensure mutual exclusion
            self.toolbar.zoom()
        self._sync_zoom_pan_styles()

    def toggle_pan(self):
        if self.toolbar.mode == 'pan':
            self.toolbar.pan()
        else:
            if self.toolbar.mode == 'zoom': self.toolbar.zoom()
            self.toolbar.pan()
        self._sync_zoom_pan_styles()

    def _sync_zoom_pan_styles(self):
        # Green = active, Blue = inactive
        self.zoom_btn.configure(bootstyle=SUCCESS if self.toolbar.mode == 'zoom' else INFO)
        self.pan_btn.configure(bootstyle=SUCCESS if self.toolbar.mode == 'pan' else INFO)

# --------------------------- Main ---------------------------

if __name__ == '__main__':
    root = tb.Window(themename="flatly")  # try "darkly" for dark mode
    app = CVPlotterApp(root)
    root.mainloop()
