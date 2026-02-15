"""Minimal PyQtGraph test to verify plotting works."""
import sys
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication

# Create Qt application
app = QApplication(sys.argv)

# Create plot window
win = pg.PlotWidget(title="Minimal PyQtGraph Test")
win.setWindowTitle("Test - Should show red sine wave")
win.resize(800, 600)

# Create simple data
x = np.linspace(0, 10, 1000)
y = np.sin(2 * np.pi * x)

# Test 1: Try ScatterPlotItem instead of PlotDataItem
print("Creating ScatterPlotItem...")
scatter = pg.ScatterPlotItem(x, y, size=5, pen=pg.mkPen('r', width=2), brush=pg.mkBrush('r'))
win.addItem(scatter)
print(f"ScatterPlotItem: {scatter}, visible: {scatter.isVisible()}, flags: {scatter.flags()}")

# Test 2: Also try PlotDataItem
print("\nCreating PlotDataItem...")
curve = win.plot(x, y, pen=pg.mkPen('b', width=3))
from PySide6.QtWidgets import QGraphicsItem
curve.setFlag(QGraphicsItem.GraphicsItemFlag.ItemHasNoContents, False)
print(f"PlotDataItem: {curve}, visible: {curve.isVisible()}, flags: {curve.flags()}")

# CRITICAL FIX: Set the ViewBox range to show all data
print("\nSetting ViewBox range to show all data...")
win.getPlotItem().getViewBox().setRange(xRange=(0, 10), yRange=(-1.5, 1.5), padding=0)
print(f"ViewBox range set to x=(0,10), y=(-1.5,1.5)")

# Show window
win.show()

# Force updates
win.getPlotItem().getViewBox().update()
win.update()
win.repaint()

print("Window shown. You should see a red sine wave from 0-10.")
print(f"\nViewBox bounds: {win.getPlotItem().getViewBox().viewRange()}")
print(f"ViewBox rect: {win.getPlotItem().getViewBox().rect()}")

# Try to force the scene to update
scene = win.scene()
if scene:
    scene.update()
    print(f"Scene updated. Items in scene: {len(scene.items())}")

# Process events to force rendering
app.processEvents()

# One more diagnostic - check if items are actually in the scene
print("\nChecking scene items:")
for item in scene.items()[:20]:  # First 20 items
    print(f"  {type(item).__name__}: visible={item.isVisible()}, pos={item.pos()}")
print(f"Curve: {curve}")
print(f"Curve visible: {curve.isVisible()}")
print(f"Curve has data: {len(curve.xData) if curve.xData is not None else 0} points")

# Run
sys.exit(app.exec())
