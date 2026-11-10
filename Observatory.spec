# Build on Windows: python -m PyInstaller --noconfirm Observatory.spec
from pathlib import Path

root = Path(SPECPATH)
analysis = Analysis(
    [str(root / 'launch_observatory.pyw')],
    pathex=[str(root)],
    datas=[(str(root / 'simulations/star_system/catalog-sources.json'), 'simulations/star_system')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={'matplotlib': {'backends': ['QtAgg', 'Agg']}},
    excludes=['PySide2', 'PySide6', 'PyQt5', 'tkinter', 'pytest', 'IPython', 'scipy', 'pandas'],
    noarchive=False,
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz, analysis.scripts, analysis.binaries, analysis.datas, [],
    name='Observatory', debug=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
)
