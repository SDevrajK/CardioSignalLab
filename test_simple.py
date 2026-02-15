"""Simplest possible PyQtGraph test with auto-range."""
import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)

# Create plot window
win = pg.PlotWidget()
win.setWindowTitle("Simple Test - Auto Range")
win.resize(800, 600)

# Create data
x = np.linspace(0, 10, 1000)
y = np.sin(2 * np.pi * x)

# Plot using the simplest method possible
win.plot(x, y, pen='r')

# Enable auto range - this should automatically fit the data
win.enableAutoRange()

# Show
win.show()

print(f"Data range: x=[{x.min()}, {x.max()}], y=[{y.min()}, {y.max()}]")
print("Auto-range enabled. Window should show full sine wave.")

sys.exit(app.exec())
