#!/usr/bin/env python3
"""Assemble the 180-mm × 220-mm Figure 2 at 300 DPI."""

import os
from copy import copy
from PIL import Image
from PyPDF2 import PdfReader, PdfWriter, Transformation
from PyPDF2._page import PageObject

OUT_TAG   = os.environ.get("VOICE_RESID_OUT_TAG", "").strip()
STEP3_DIR = os.environ.get("VOICE_STEP3_DIR", "step3_voice_age_ridge").strip()

def tagged_name(name):
    return f"{name}_{OUT_TAG}" if OUT_TAG else name

BASE = "/home/davidkro/PycharmProjects/DeepVoice/analysis_outputs/step5_volcano"
LOLLIPOP = os.path.join(BASE, tagged_name("voice_residualized"), "lollipop_combined_p25_.png")
BOXPLOTS = os.path.join(BASE, tagged_name("voice_residualized"), "boxplots", "boxplots_panel.png")
LOLLIPOP_PDF = os.path.join(
    BASE, tagged_name("voice_residualized"), "lollipop_combined_p25_.pdf"
)
BOXPLOTS_PDF = os.path.join(
    BASE, tagged_name("voice_residualized"), "boxplots", "boxplots_panel.pdf"
)
OUTDIR   = os.path.join(BASE, tagged_name("voice_residualized"))
OUT_PNG  = os.path.join(OUTDIR, "figure2_combined.png")
OUT_PDF  = os.path.join(OUTDIR, "figure2_combined.pdf")

DPI = 300
GAP_MM = 3
GAP_PX = round(GAP_MM / 25.4 * DPI)


def assemble_vector_pdf() -> None:
    """Place the two source PDF pages side by side without rasterizing text."""
    lollipop_reader = PdfReader(LOLLIPOP_PDF)
    boxplots_reader = PdfReader(BOXPLOTS_PDF)
    lollipop_page = lollipop_reader.pages[0]
    boxplots_page = boxplots_reader.pages[0]

    width_points = 180 / 25.4 * 72
    height_points = 220 / 25.4 * 72
    gap_points = GAP_MM / 25.4 * 72
    combined_page = PageObject.create_blank_page(
        width=width_points,
        height=height_points,
    )
    lollipop_copy = copy(lollipop_page)
    combined_page.merge_page(lollipop_copy)

    boxplots_copy = copy(boxplots_page)
    boxplots_copy.add_transformation(
        Transformation().translate(
            tx=float(lollipop_page.mediabox.width) + gap_points, ty=0
        )
    )
    combined_page.merge_page(boxplots_copy)

    writer = PdfWriter()
    writer.add_page(combined_page)
    with open(OUT_PDF, "wb") as handle:
        writer.write(handle)


def assemble():
    lol = Image.open(LOLLIPOP).convert("RGB")
    box = Image.open(BOXPLOTS).convert("RGB")

    # Align heights: scale the shorter one up to match the taller
    h_target = max(lol.height, box.height)

    def resize_to_height(img, h):
        if img.height == h:
            return img
        w = round(img.width * h / img.height)
        return img.resize((w, h), Image.LANCZOS)

    lol = resize_to_height(lol, h_target)
    box = resize_to_height(box, h_target)

    total_w = lol.width + GAP_PX + box.width
    combined = Image.new("RGB", (total_w, h_target), color=(255, 255, 255))
    combined.paste(lol, (0, 0))
    combined.paste(box, (lol.width + GAP_PX, 0))

    os.makedirs(OUTDIR, exist_ok=True)
    combined.save(OUT_PNG, dpi=(DPI, DPI))
    assemble_vector_pdf()
    print(f"Saved → {OUT_PNG}")

    w_mm = combined.width / DPI * 25.4
    h_mm = combined.height / DPI * 25.4
    print(f"Size: {w_mm:.1f} x {h_mm:.1f} mm")


if __name__ == "__main__":
    assemble()
