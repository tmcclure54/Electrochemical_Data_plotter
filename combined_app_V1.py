import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ttkbootstrap as ttk
from matplotlib.ticker import ScalarFormatter
from ttkbootstrap.constants import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

FARADAY_CONSTANT = 96485.0  # C/mol e−

# ---------- Shared Helpers ----------
def clean_column_name(name):
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def parse_data_file(file_path):
    """Auto-detect delimiter and return DataFrame"""
    with open(file_path, 'r', errors='ignore') as f:
        first_line = f.readline()
    if ',' in first_line:
        df = pd.read_csv(file_path, sep=',', engine="python")
    elif '\t' in first_line:
        df = pd.read_csv(file_path, sep='\t', engine="python")
    else:
        df = pd.read_csv(file_path, sep=None, engine="python")
    return df

def set_nature_chem_style():
    sns.set_context("talk")
    sns.set_style("white")
    plt.rcParams.update({
        "font.family": "Arial",
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": 'in',
        "ytick.direction": 'in',
        "legend.fontsize": 12,
        "savefig.dpi": 300,
        "figure.dpi": 100
    })

def find_col(df, key):
    """Find a column containing a key word family after cleaning"""
    for c in df.columns:
        if key in clean_column_name(c):
            return c
    return None

def infer_current_units(I_raw):
    """Return 'A' or 'mA' based on magnitude heuristic."""
    if I_raw is None:
        return 'A'
    try:
        arr = pd.to_numeric(I_raw, errors='coerce').to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 'A'
        q95 = np.nanpercentile(np.abs(arr), 95)
        if q95 <= 0.02:
            return 'A'     # <= 20 mA typical CV range → values in A
        if q95 <= 200:
            return 'mA'    # looks like mA-scale numbers
        return 'mA'
    except Exception:
        return 'A'

# ---------- Tab 1: CV Plotter with iR compensation ----------
class CVPlotter(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)

        self.files, self.legend_vars, self.area_vars, self.rs_vars, self.unit_vars = [], [], [], [], []

        # Top controls
        top = ttk.Frame(self)
        top.pack(fill=X, pady=6)
        ttk.Label(top, text="X-axis:").pack(side=LEFT)
        self.x_label = ttk.Entry(top, width=18)
        self.x_label.insert(0, "Voltage (V)")
        self.x_label.pack(side=LEFT, padx=6)

        ttk.Label(top, text="Y-axis:").pack(side=LEFT)
        self.y_label = ttk.Entry(top, width=32)
        self.y_label.insert(0, "Current (mA) / Current Density (mA/cm²)")
        self.y_label.pack(side=LEFT, padx=6)

        ttk.Label(top, text="Title:").pack(side=LEFT)
        self.title = ttk.Entry(top, width=28)
        self.title.insert(0, "Cyclic Voltammograms")
        self.title.pack(side=LEFT, padx=6)

        # Texas sign toggle (visualization only)
        self.texas_sign = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Texas sign (plot −I)", variable=self.texas_sign).pack(side=LEFT, padx=10)

        # Paned layout
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill=BOTH, expand=True, pady=6)

        # Left pane: file list + buttons
        left_pane = ttk.Frame(paned)
        paned.add(left_pane, weight=1)

        btns = ttk.Frame(left_pane)
        btns.pack(fill=X, pady=4)
        ttk.Button(btns, text="Add Files", command=self.add_files).pack(side=LEFT, padx=5)
        ttk.Button(btns, text="Clear", command=self.clear_files).pack(side=LEFT, padx=5)
        ttk.Button(btns, text="Plot", bootstyle=PRIMARY, command=self.plot_all).pack(side=LEFT, padx=5)
        ttk.Button(btns, text="Save Plot", bootstyle=SUCCESS, command=self.save_plot).pack(side=LEFT, padx=5)

        list_frame = ttk.Frame(left_pane)
        list_frame.pack(fill=BOTH, expand=True, padx=4, pady=4)

        self.scroll_canvas = tk.Canvas(list_frame, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.scroll_canvas.yview)
        self.inner_frame = ttk.Frame(self.scroll_canvas)
        self.inner_frame.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))
        )
        self.scroll_canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Right pane: big plot
        right_pane = ttk.Frame(paned)
        paned.add(right_pane, weight=4)

        set_nature_chem_style()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, right_pane)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_pane)
        self.toolbar.update()

        # initial sash position so plot dominates
        self.after(50, lambda: paned.sashpos(0, 360))

    def add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[("CSV/TXT files","*.csv *.txt")])
        for p in paths:
            if p not in self.files:
                self.files.append(p)
                self._add_entry(p)

    def _add_entry(self, filepath):
        row = ttk.Frame(self.inner_frame)
        row.pack(fill=X, pady=3)

        ttk.Label(row, text=os.path.basename(filepath), width=24, anchor="w").pack(side=LEFT)

        # Label
        ttk.Label(row, text="Label:").pack(side=LEFT, padx=(6,2))
        var_label = tk.StringVar(value=os.path.basename(filepath))
        ttk.Entry(row, textvariable=var_label, width=18).pack(side=LEFT)
        self.legend_vars.append(var_label)

        # Area
        ttk.Label(row, text="Area (cm²):").pack(side=LEFT, padx=(8,2))
        var_area = tk.StringVar()
        ttk.Entry(row, textvariable=var_area, width=10).pack(side=LEFT)
        self.area_vars.append(var_area)

        # Per-trace Rs
        ttk.Label(row, text="Rs (Ω):").pack(side=LEFT, padx=(8,2))
        var_rs = tk.StringVar()
        ttk.Entry(row, textvariable=var_rs, width=10).pack(side=LEFT)
        self.rs_vars.append(var_rs)

        # Units override
        ttk.Label(row, text="I units:").pack(side=LEFT, padx=(8,2))
        var_unit = tk.StringVar(value="auto")
        cb = ttk.Combobox(row, textvariable=var_unit, width=6, values=["auto","A","mA"], state="readonly")
        cb.pack(side=LEFT)
        self.unit_vars.append(var_unit)

    def clear_files(self):
        self.files, self.legend_vars, self.area_vars, self.rs_vars, self.unit_vars = [], [], [], [], []
        for w in self.inner_frame.winfo_children():
            w.destroy()
        self.ax.clear()
        self.canvas.draw()

    def _coerce_numeric(self, s):
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)

    def plot_all(self):
        if not self.files:
            messagebox.showwarning("No files","Add at least one file.")
            return

        self.ax.clear()
        colors = sns.color_palette("husl", len(self.files))
        plotted = False

        for i, path in enumerate(self.files):
            try:
                df = parse_data_file(path)
                vcol = find_col(df, "volt") or find_col(df, "potent")
                ccol = find_col(df, "curr")
                if vcol is None or ccol is None:
                    raise ValueError(f"Voltage/Current columns not found in {os.path.basename(path)}")

                # Raw numeric arrays (keep time order)
                V_meas = self._coerce_numeric(df[vcol])
                I_raw  = self._coerce_numeric(df[ccol])

                # --- Unit handling: convert to A for physics ---
                unit_sel = self.unit_vars[i].get() if i < len(self.unit_vars) else "auto"
                units_in = unit_sel if unit_sel in ("A","mA") else infer_current_units(I_raw)
                I_A_phys = I_raw if units_in == "A" else I_raw / 1000.0

                # ----- iR compensation uses PHYSICAL current (Texas sign must not affect it) -----
                V_plot = V_meas.copy()
                rs_txt = (self.rs_vars[i].get() or "").strip()
                if rs_txt:
                    try:
                        Rs = float(rs_txt)
                        if Rs < 0:
                            raise ValueError
                        V_plot = V_meas - (I_A_phys * Rs)
                    except Exception:
                        messagebox.showwarning(
                            "Invalid Rs",
                            f"Could not parse Rs for {os.path.basename(path)}; skipping iR compensation for this trace."
                        )

                # ----- Texas sign is visualization ONLY -----
                I_A_plot = -I_A_phys if self.texas_sign.get() else I_A_phys
                I_mA_plot = I_A_plot * 1000.0

                # ----- area normalization (optional; visualization only) -----
                area_str = (self.area_vars[i].get() or "").strip()
                if area_str:
                    try:
                        A = float(area_str)
                        if A > 0:
                            I_mA_plot = I_mA_plot / A  # mA/cm²
                    except Exception:
                        messagebox.showwarning(
                            "Invalid area",
                            f"Could not parse area for {os.path.basename(path)}; plotting raw current."
                        )

                # Drop NaNs for plotting
                mask = np.isfinite(V_plot) & np.isfinite(I_mA_plot)
                if not np.any(mask):
                    raise ValueError("No finite data points after parsing.")
                self.ax.plot(
                    V_plot[mask], I_mA_plot[mask],
                    label=(self.legend_vars[i].get() or os.path.basename(path)),
                    color=colors[i], lw=2, alpha=0.95
                )
                plotted = True

            except Exception as e:
                messagebox.showerror("Parse Error", f"Error processing {path}:\n{e}")

        if not plotted:
            return

        self.ax.set_xlabel(self.x_label.get())
        self.ax.set_ylabel(self.y_label.get())
        self.ax.set_title(self.title.get())
        # Electrochemistry convention: decreasing potential to the right
        self.ax.invert_xaxis()
        self.ax.axhline(0, color='0.7', lw=1, ls='--')
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        if not self.files:
            messagebox.showwarning("Nothing to save", "Plot something first.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")],
            title="Save Plot As"
        )
        if save_path:
            self.fig.savefig(save_path, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Plot saved to:\n{save_path}")

# ---------- Tab 2: Bulk Electrolysis ----------
class ElectrolysisPlotter(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.file = None
        self.ax2 = None  # secondary axis (for cumulative charge)
        self._df = None  # cache df

        # --- Title & Axis label controls ---
        lbls = ttk.Frame(self)
        lbls.pack(fill=X, pady=6)
        ttk.Label(lbls, text="X-axis:").pack(side=LEFT)
        self.x_label = ttk.Entry(lbls, width=20)
        self.x_label.insert(0, "Time (s)")
        self.x_label.pack(side=LEFT, padx=6)

        ttk.Label(lbls, text="Y-axis:").pack(side=LEFT)
        self.y_label = ttk.Entry(lbls, width=20)
        self.y_label.insert(0, "Current (mA)")
        self.y_label.pack(side=LEFT, padx=6)

        ttk.Label(lbls, text="Title:").pack(side=LEFT)
        self.title = ttk.Entry(lbls, width=25)
        self.title.insert(0, "Bulk Electrolysis")
        self.title.pack(side=LEFT, padx=6)

        # --- Top controls (file + plot/save) ---
        control = ttk.Frame(self)
        control.pack(fill=X, pady=6)
        ttk.Button(control, text="Load File", command=self.load_file).pack(side=LEFT, padx=5)
        ttk.Button(control, text="Plot", bootstyle=PRIMARY, command=self.plot_data).pack(side=LEFT, padx=5)
        ttk.Button(control, text="Save Plot", bootstyle=SUCCESS, command=self.save_plot).pack(side=LEFT, padx=5)

        # Primary signal: Current or Voltage (single selectable)
        mode_frame = ttk.Frame(self)
        mode_frame.pack(fill=X, pady=2)
        ttk.Label(mode_frame, text="Primary Y:").pack(side=LEFT, padx=(6, 2))
        self.y_mode = tk.StringVar(value="current")
        ttk.Radiobutton(mode_frame, text="Current (mA)", variable=self.y_mode, value="current").pack(side=LEFT)
        ttk.Radiobutton(mode_frame, text="Voltage (V)", variable=self.y_mode, value="voltage").pack(side=LEFT, padx=8)

        # Overlay cumulative charge option
        overlay_frame = ttk.Frame(self)
        overlay_frame.pack(fill=X, pady=2)
        self.show_cumq = tk.BooleanVar(value=False)
        ttk.Checkbutton(overlay_frame, text="Overlay cumulative charge (mC)", variable=self.show_cumq).pack(side=LEFT, padx=6)

        # --- Column mapping (auto-populated after load) ---
        map_frame = ttk.Labelframe(self, text="Column mapping")
        map_frame.pack(fill=X, padx=6, pady=6)

        ttk.Label(map_frame, text="Time:").grid(row=0, column=0, sticky="w", padx=6, pady=2)
        self.time_col = ttk.Combobox(map_frame, width=24, state="readonly")
        self.time_col.grid(row=0, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(map_frame, text="Current:").grid(row=1, column=0, sticky="w", padx=6, pady=2)
        self.curr_col = ttk.Combobox(map_frame, width=24, state="readonly")
        self.curr_col.grid(row=1, column=1, sticky="w", padx=6, pady=2)

        ttk.Label(map_frame, text="Current column units (input):").grid(row=1, column=2, sticky="w", padx=(18,6), pady=2)
        self.curr_units = ttk.Combobox(map_frame, width=8, values=["A", "mA"], state="readonly")
        self.curr_units.set("A")
        self.curr_units.grid(row=1, column=3, sticky="w", padx=6, pady=2)

        ttk.Label(map_frame, text="Voltage:").grid(row=2, column=0, sticky="w", padx=6, pady=2)
        self.volt_col = ttk.Combobox(map_frame, width=24, state="readonly")
        self.volt_col.grid(row=2, column=1, sticky="w", padx=6, pady=2)

        # --- Faradaic Efficiency inputs ---
        fe_frame = ttk.Labelframe(self, text="Faradaic Metrics (optional)")
        fe_frame.pack(fill=X, padx=6, pady=6)
        prod = ttk.Frame(fe_frame); prod.pack(fill=X, pady=4)
        ttk.Label(prod, text="Product mass:").pack(side=LEFT, padx=(4, 2))
        self.product_mass = ttk.Entry(prod, width=10); self.product_mass.pack(side=LEFT)
        self.mass_unit = ttk.Combobox(prod, width=6, values=["mg", "g"], state="readonly")
        self.mass_unit.set("mg"); self.mass_unit.pack(side=LEFT, padx=4)
        ttk.Label(prod, text="Molar mass (g/mol):").pack(side=LEFT, padx=(10, 2))
        self.molar_mass = ttk.Entry(prod, width=10); self.molar_mass.pack(side=LEFT)
        ttk.Label(prod, text="Electrons per product (n):").pack(side=LEFT, padx=(10, 2))
        self.n_e_product = ttk.Entry(prod, width=8); self.n_e_product.pack(side=LEFT)

        sub = ttk.Frame(fe_frame); sub.pack(fill=X, pady=4)
        ttk.Label(sub, text="Substrate amount (mmol):").pack(side=LEFT, padx=(4, 2))
        self.substrate_mmol = ttk.Entry(sub, width=10); self.substrate_mmol.pack(side=LEFT)
        ttk.Label(sub, text="Electrons per substrate (n):").pack(side=LEFT, padx=(10, 2))
        self.n_e_sub = ttk.Entry(sub, width=8); self.n_e_sub.pack(side=LEFT)

        # --- Big plot area ---
        paned = ttk.Panedwindow(self, orient="horizontal")
        paned.pack(fill=BOTH, expand=True, pady=6)
        left_stub = ttk.Frame(paned); paned.add(left_stub, weight=1)
        right_plot = ttk.Frame(paned); paned.add(right_plot, weight=4)

        set_nature_chem_style()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, right_plot)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_plot); self.toolbar.update()
        self.after(50, lambda: paned.sashpos(0, 200))

    # ---------- helpers ----------
    def _as_numeric_ndarray(self, x):
        y = pd.to_numeric(x, errors="coerce")
        y = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        return y.astype(float, copy=False)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV/TXT files","*.csv *.txt")])
        if not path:
            return
        try:
            df = parse_data_file(path)
            self.file = path
            self._df = df
            cols = list(df.columns)

            # populate dropdowns
            for cb in (self.time_col, self.curr_col, self.volt_col):
                cb.configure(values=cols)

            # smart defaults
            self.time_col.set(self._best_col(cols, ["time", "sec", "s"]))
            self.curr_col.set(self._best_col(cols, ["current", "i"]))
            self.volt_col.set(self._best_col(cols, ["voltage", "potential", "ewe", "v"]))
            if not self.time_col.get() and cols:
                self.time_col.set(cols[0])

            messagebox.showinfo("Loaded", f"Loaded file:\n{os.path.basename(path)}\n\n"
                                          f"Confirm the column mapping before plotting.")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _best_col(self, cols, keys):
        def score(c):
            cc = clean_column_name(c)
            if any(x in cc for x in ["noise","std","density","dens","aux"]):
                return -1
            for k in keys:
                if cc == k or cc.startswith(k) or k in cc:
                    return 2 if (cc == k or cc.startswith(k)) else 1
            return 0
        scored = sorted(((score(c), c) for c in cols), reverse=True)
        return scored[0][1] if scored and scored[0][0] > 0 else ""

    def _to_amps(self, I_values, input_units):
        if input_units == "A":
            return I_values
        elif input_units == "mA":
            return I_values / 1000.0
        else:
            return I_values

    def _to_mA(self, I_values_A):
        return I_values_A * 1000.0

    def _integrate_charge(self, t_s, I_A_sorted):
        if t_s.size < 2:
            raise ValueError("Not enough valid time/current points to integrate.")
        return abs(np.trapz(I_A_sorted, t_s))

    def _cumulative_charge(self, t_s, I_A_sorted):
        Q = np.zeros_like(I_A_sorted, dtype=float)
        if t_s.size < 2:
            return Q
        dt = np.diff(t_s)
        avgI = 0.5 * (I_A_sorted[1:] + I_A_sorted[:-1])
        Q[1:] = np.cumsum(avgI * dt)
        return np.abs(Q)

    def plot_data(self):
        if self._df is None:
            messagebox.showwarning("No file", "Load a file first.")
            return
        try:
            df = self._df
            tcol = self.time_col.get() or ""
            ccol = self.curr_col.get() or ""
            vcol = self.volt_col.get() or ""

            if tcol not in df.columns:
                raise ValueError("Please choose a valid Time column before plotting.")

            t_num = self._as_numeric_ndarray(df[tcol])
            mask_t = np.isfinite(t_num)
            t_sorted = t_num[mask_t]
            order = np.argsort(t_sorted)
            t_s = t_sorted[order]

            I_A_sorted = None
            I_mA_sorted = None
            if ccol in df.columns and ccol:
                I_num_all = self._as_numeric_ndarray(df[ccol])
                I_num = I_num_all[mask_t][order]
                I_A_sorted = self._to_amps(I_num, self.curr_units.get())
                I_mA_sorted = self._to_mA(I_A_sorted)

            V_sorted = None
            if vcol in df.columns and vcol:
                V_all = self._as_numeric_ndarray(df[vcol])
                V_sorted = V_all[mask_t][order]

            charge_C = charge_mC = mmol_e = None
            if I_A_sorted is not None:
                charge_C = self._integrate_charge(t_s, I_A_sorted)
                charge_mC = charge_C * 1000.0
                mmol_e = (charge_C / FARADAY_CONSTANT) * 1000.0

            self.ax.clear()
            if self.ax2 is not None:
                self.ax2.remove(); self.ax2 = None

            lines, labels = [], []

            if self.y_mode.get() == "voltage":
                if V_sorted is None:
                    raise ValueError("No Voltage column selected.")
                maskV = np.isfinite(t_s) & np.isfinite(V_sorted)
                lineV, = self.ax.plot(t_s[maskV], V_sorted[maskV], lw=2, label="Voltage (V)")
                self.ax.set_ylabel("Voltage (V)")
                lines.append(lineV); labels.append("Voltage (V)")
            else:
                if I_mA_sorted is None:
                    raise ValueError("No Current column selected.")
                maskI = np.isfinite(t_s) & np.isfinite(I_mA_sorted)
                lineI, = self.ax.plot(t_s[maskI], I_mA_sorted[maskI], lw=2, label="Current (mA)")
                self.ax.set_ylabel("Current (mA)")
                lines.append(lineI); labels.append("Current (mA)")

            self.ax.set_xlabel(self.x_label.get())
            self.ax.set_title(self.title.get())

            fmt = ScalarFormatter(useOffset=False)
            fmt.set_scientific(False)
            self.ax.yaxis.set_major_formatter(fmt)
            self.ax.ticklabel_format(axis='y', style='plain', useOffset=False)

            if self.show_cumq.get() and I_A_sorted is not None:
                maskI = np.isfinite(t_s) & np.isfinite(I_A_sorted)
                Q_cum_mC = self._cumulative_charge(t_s[maskI], I_A_sorted[maskI]) * 1000.0
                self.ax2 = self.ax.twinx()
                lineQ, = self.ax2.plot(t_s[maskI], Q_cum_mC, lw=2, linestyle="--", label="Charge (mC)")
                self.ax2.set_ylabel("Charge (mC)")
                self.ax2.grid(False)
                fmt2 = ScalarFormatter(useOffset=False)
                fmt2.set_scientific(False)
                self.ax2.yaxis.set_major_formatter(fmt2)
                self.ax2.ticklabel_format(axis='y', style='plain', useOffset=False)
                lines.append(lineQ); labels.append("Charge (mC)")

            self.fig.tight_layout()
            self.ax.legend(lines, labels, loc="best")
            self.canvas.draw()

            if I_A_sorted is not None and charge_C is not None:
                FE_percent = None
                m_txt = (self.product_mass.get() or "").strip()
                M_txt = (self.molar_mass.get() or "").strip()
                n_prod_txt = (self.n_e_product.get() or "").strip()
                if m_txt and M_txt and n_prod_txt and charge_C > 0:
                    try:
                        mass_g = (float(m_txt) / 1000.0) if self.mass_unit.get() == "mg" else float(m_txt)
                        M = float(M_txt)
                        n_e_prod = float(n_prod_txt)
                        if mass_g >= 0 and M > 0 and n_e_prod > 0:
                            n_product_mol = mass_g / M
                            FE_percent = (n_product_mol * n_e_prod * FARADAY_CONSTANT) / charge_C * 100.0
                    except Exception:
                        messagebox.showwarning("FE inputs", "Could not parse product-based FE inputs. Skipping FE.")

                pct_of_theory = None
                sub_txt = (self.substrate_mmol.get() or "").strip()
                n_sub_txt = (self.n_e_sub.get() or "").strip()
                if sub_txt and n_sub_txt:
                    try:
                        mmol_sub = float(sub_txt); n_sub = float(n_sub_txt)
                        if mmol_sub > 0 and n_sub > 0:
                            mmol_e = (charge_C / FARADAY_CONSTANT) * 1000.0
                            pct_of_theory = (mmol_e / (mmol_sub * n_sub)) * 100.0
                    except Exception:
                        messagebox.showwarning("Substrate inputs", "Could not parse substrate-based inputs. Skipping % theoretical.")

                summary = [f"Total charge: {charge_mC:.2f} mC",
                           f"Electrons delivered: {mmol_e:.3f} mmol e⁻"]
                if FE_percent is not None and np.isfinite(FE_percent):
                    summary.append(f"Faradaic Efficiency (product-based): {FE_percent:.1f}%")
                if pct_of_theory is not None and np.isfinite(pct_of_theory):
                    summary.append(f"Charge vs. theoretical for full substrate conversion: {pct_of_theory:.1f}%")
                messagebox.showinfo("Results", "\n".join(summary))
            else:
                messagebox.showinfo("Results", "Plotted Voltage vs Time.\n(Current not selected → charge metrics unavailable.)")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_plot(self):
        if self._df is None:
            messagebox.showwarning("Nothing to save","Load and plot data first.")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")],
            title="Save Plot As"
        )
        if save_path:
            self.fig.savefig(save_path, bbox_inches="tight")
            messagebox.showinfo("Saved", f"Plot saved to:\n{save_path}")

# ---------- Main ----------
class MainApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Electrochemistry Plotter")
        self.geometry("1200x800")

        nb = ttk.Notebook(self)
        nb.pack(fill=BOTH, expand=True)

        nb.add(CVPlotter(nb), text="Voltammograms")
        nb.add(ElectrolysisPlotter(nb), text="Bulk Electrolysis")

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
