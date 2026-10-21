"""Shared ROOT plotting helpers for the flavour-aware jet notebooks.

* Text sizes in pixels (font precision 3: Helvetica = 43), so that the top and bottom pad of a two-pad plot get
  the same text size. Sizes given as fractions refer to the pad, which differs between the two pads.
* Legends sized to their content and placed where they don't cover anything drawn in the frame.
"""
import ctypes
import math

import numpy as np
import ROOT

FONT = 43
LABEL_PX, TITLE_PX, LEGEND_PX = 22, 24, 22
LEFT_MARGIN = 0.13
X_TITLE_OFFSET, Y_TITLE_OFFSET = 1.1, 2.0
BOTTOM_MARGIN_PX = 80  # room for the x labels and title of the bottom pad

SYMBOL_PX = 40  # width of the legend symbol column
ROW_PX = 1.3 * LEGEND_PX  # height of a legend row
GAP_PX = 8  # free space kept between the legend, the data and the frame


def pixel_axes(*axes):
    for ax in axes:
        ax.SetLabelFont(FONT)
        ax.SetLabelSize(LABEL_PX)
        ax.SetTitleFont(FONT)
        ax.SetTitleSize(TITLE_PX)


def pixel_legend(leg):
    leg.SetTextFont(FONT)
    leg.SetTextSize(LEGEND_PX)


def text_width_px(text):
    t = ROOT.TLatex(0., 0., text)
    t.SetTextFont(FONT)
    t.SetTextSize(LEGEND_PX)
    w, h = ctypes.c_uint(0), ctypes.c_uint(0)
    t.GetBoundingBox(w, h)
    return w.value


def make_legend(entries, ncols=1):
    """TLegend with its entries [(object, text, option)] and the size it needs; returns (legend, width_px,
    height_px). Draw it with place_legend or draw_legend_at."""
    col_w = SYMBOL_PX + max(text_width_px(text) for _, text, _ in entries) + 10
    rows = math.ceil(len(entries) / ncols)
    leg = ROOT.TLegend(0., 0., 1., 1.)
    pixel_legend(leg)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetNColumns(ncols)
    leg.SetMargin(SYMBOL_PX / col_w)
    for obj, text, opt in entries:
        leg.AddEntry(obj, text, opt)
    return leg, ncols * col_w, rows * ROW_PX + 4


class _Frame:
    """Pixel geometry of the frame of a pad (after pad.Update())."""

    def __init__(self, pad):
        self.wpx = pad.GetWw() * pad.GetAbsWNDC()
        self.hpx = pad.GetWh() * pad.GetAbsHNDC()
        self.x0 = pad.GetLeftMargin() * self.wpx
        self.y0 = pad.GetBottomMargin() * self.hpx
        self.fw = int(self.wpx * (1. - pad.GetLeftMargin() - pad.GetRightMargin()))
        self.fh = int(self.hpx * (1. - pad.GetBottomMargin() - pad.GetTopMargin()))
        self.xmin, self.xmax = pad.GetUxmin(), pad.GetUxmax()
        self.ymin, self.ymax = pad.GetUymin(), pad.GetUymax()
        self.occ = np.zeros((self.fh, self.fw), dtype=bool)  # [row from the bottom, column]

    def px(self, x):
        return (x - self.xmin) / (self.xmax - self.xmin) * self.fw

    def py(self, y):
        return (y - self.ymin) / (self.ymax - self.ymin) * self.fh

    def fill(self, x1, x2, y1, y2):
        """Mark the user-coordinate rectangle as occupied (clipped to the frame)."""
        c1, c2 = sorted((self.px(x1), self.px(x2)))
        r1, r2 = sorted((self.py(y1), self.py(y2)))
        c1, c2 = max(0, int(c1) - 1), min(self.fw, int(math.ceil(c2)) + 1)
        r1, r2 = max(0, int(r1) - 1), min(self.fh, int(math.ceil(r2)) + 1)
        if c1 < c2 and r1 < r2:
            self.occ[r1:r2, c1:c2] = True

    def segment(self, xa, ya, xb, yb):
        n = int(max(abs(self.px(xb) - self.px(xa)), abs(self.py(yb) - self.py(ya)))) + 2
        for t in np.linspace(0., 1., n):
            x, y = xa + t * (xb - xa), ya + t * (yb - ya)
            self.fill(x, x, y, y)

    def add(self, obj, filled=False):
        if isinstance(obj, ROOT.THStack):
            obj = obj.GetStack().Last()
            filled = True
        if isinstance(obj, ROOT.TH2):
            for ix in range(1, obj.GetNbinsX() + 1):
                for iy in range(1, obj.GetNbinsY() + 1):
                    if obj.GetBinContent(ix, iy) > 0:
                        ax, ay = obj.GetXaxis(), obj.GetYaxis()
                        self.fill(ax.GetBinLowEdge(ix), ax.GetBinUpEdge(ix), ay.GetBinLowEdge(iy), ay.GetBinUpEdge(iy))
        elif isinstance(obj, ROOT.TH1):
            ax = obj.GetXaxis()
            prev = None
            for i in range(1, obj.GetNbinsX() + 1):
                lo, hi, v = ax.GetBinLowEdge(i), ax.GetBinUpEdge(i), obj.GetBinContent(i)
                if hi <= self.xmin or lo >= self.xmax:
                    continue
                self.fill(lo, hi, self.ymin if filled else v, v)
                if prev is not None:
                    self.fill(lo, lo, prev, v)  # vertical step between bins
                prev = v
        elif isinstance(obj, ROOT.TGraph):
            xs, ys = obj.GetX(), obj.GetY()
            for i in range(obj.GetN() - 1):
                self.segment(xs[i], ys[i], xs[i + 1], ys[i + 1])
        elif isinstance(obj, ROOT.TLine):
            self.segment(obj.GetX1(), obj.GetY1(), obj.GetX2(), obj.GetY2())
        elif isinstance(obj, ROOT.TMarker):
            self.fill(obj.GetX(), obj.GetX(), obj.GetY(), obj.GetY())
            r = 6
            c, rr = int(self.px(obj.GetX())), int(self.py(obj.GetY()))
            self.occ[max(0, rr - r):rr + r, max(0, c - r):c + r] = True

    def free_spot(self, w, h):
        """Bottom-left pixel corner (in the frame) of a free w x h box, preferring the top right corner."""
        w, h = int(math.ceil(w)) + 2 * GAP_PX, int(math.ceil(h)) + 2 * GAP_PX
        if w > self.fw or h > self.fh:
            return None
        # summed-area table for O(1) box tests
        s = np.zeros((self.fh + 1, self.fw + 1), dtype=np.int32)
        s[1:, 1:] = self.occ.cumsum(0).cumsum(1)
        free = lambda r, c: s[r + h, c + w] - s[r, c + w] - s[r + h, c] + s[r, c] == 0
        rows = range(self.fh - h, -1, -2)  # from the top down
        cols = list(range(self.fw - w, -1, -2))  # from the right
        corners = [(self.fh - h, self.fw - w), (self.fh - h, 0)]
        for r, c in corners + [(r, c) for r in rows for c in cols]:
            if free(r, c):
                return r + GAP_PX, c + GAP_PX
        return None


def place_legend(pad, leg, w, h, objects, grow=None, max_steps=8):
    """Put leg (w x h pixels) where it covers none of objects [obj or (obj, filled)] in the frame of pad.

    If there is no free spot, grow() is called (e.g. to raise the y maximum) and the search is repeated.
    Returns the drawn legend, or None if no spot was found (the legend is then not drawn).
    """
    for _ in range(max_steps + 1):
        pad.Modified()
        pad.Update()
        fr = _Frame(pad)
        for o in objects:
            fr.add(*o) if isinstance(o, tuple) else fr.add(o)
        spot = fr.free_spot(w, h)
        if spot is not None:
            r, c = spot
            x1 = (fr.x0 + c) / fr.wpx
            y1 = (fr.y0 + r) / fr.hpx
            return draw_legend_at(pad, leg, x1, y1, x1 + w / fr.wpx, y1 + h / fr.hpx)
        if grow is None:
            return None
        grow()
    return None


def draw_legend_at(pad, leg, x1, y1, x2, y2):
    """Draw a copy of leg at the given NDC coordinates of pad and return it. A TLegend has to be constructed at
    its final position: changing the NDC coordinates afterwards does not move it when it is painted."""
    new = ROOT.TLegend(x1, y1, x2, y2)
    pixel_legend(new)
    new.SetBorderSize(0)
    new.SetFillStyle(0)
    new.SetNColumns(leg.GetNColumns())
    new.SetMargin(leg.GetMargin())
    for e in leg.GetListOfPrimitives():
        new.AddEntry(e.GetObject(), e.GetLabel(), e.GetOption())
    pad.cd()
    new.Draw()
    pad._legends = getattr(pad, "_legends", []) + [new]  # keep the Python reference alive
    return new


def headroom(obj, ymin, ymax, step=0.08):
    """grow() callback for place_legend: raise the y maximum of a histogram, stack or frame drawn from ymin to
    ymax by step times the range each time (THStack.GetMaximum() ignores SetMaximum, so track it here)."""
    state = [ymax]

    def grow():
        state[0] += step * (ymax - ymin)
        obj.SetMaximum(state[0])
    return grow
