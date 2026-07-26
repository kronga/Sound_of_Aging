#!/usr/bin/env python3
"""Compatibility entry point for the final Figure 2 boxplot panel.

The submitted analysis uses AHI, waist circumference, and VAT area.  The
implementation lives in ``run_boxplots_panel.py``; this historical script name
is retained so existing commands generate the current panel rather than the
superseded VAT-mass version.
"""

from run_boxplots_panel import main


if __name__ == "__main__":
    main()
