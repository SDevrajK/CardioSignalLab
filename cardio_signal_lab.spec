# -*- mode: python ; coding: utf-8 -*-
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# NeuroKit2 ships data files (signal templates, sample datasets) - bundle everything.
nk_datas, nk_binaries, nk_hiddenimports = collect_all('neurokit2')

# PyEMD is pure-Python but PyInstaller sometimes misses dynamic submodule imports.
emd_hiddenimports = collect_submodules('PyEMD')

# pyxdf is pure-Python - collect submodules to be safe.
pyxdf_hiddenimports = collect_submodules('pyxdf')

hiddenimports = (
    nk_hiddenimports
    + emd_hiddenimports
    + pyxdf_hiddenimports
    + [
        'scipy.signal',
        'scipy.special',
        'scipy.stats',
        'scipy._lib.messagestream',
    ]
)

a = Analysis(
    ['main.py'],
    pathex=['src'],
    binaries=nk_binaries,
    datas=nk_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib.tests',
        'numpy.tests',
        'scipy.tests',
        'pandas.tests',
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CardioSignalLab',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='CardioSignalLab',
)

# On macOS, wrap the COLLECT output in a proper .app bundle so users can
# drag-and-drop to /Applications. No-op on Windows/Linux.
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='CardioSignalLab.app',
        icon=None,
        bundle_identifier='com.hebertlab.cardiosignallab',
        info_plist={
            'NSHighResolutionCapable': True,
            'NSPrincipalClass': 'NSApplication',
            'CFBundleShortVersionString': '0.2.0',
            'CFBundleVersion': '0.2.0',
        },
    )
