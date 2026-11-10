"""Native sidebar controls: catalog, body information and optional precise edits."""
import math

import numpy as np
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QLineEdit, QGroupBox,
)

from .data import AU_KM, CATALOG, SOLAR_KG
from .rendering import planet_sprite, star_sprite
from .templates import override_properties

AUYR_TO_KMS = AU_KM/(365.25*86400)
TYPES = {'Stellar': ('star',), 'Planetary': ('planet',), 'Lunar': ('moon',), 'Other': ('asteroid', 'particle')}


class Dropdown(QComboBox):
    """A native combo with a readable chevron on every platform theme."""
    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor('#b4c8d9'), 1.5))
        x, y = self.width()-16, self.height()/2
        painter.drawPolyline(QPolygonF([QPointF(x-4, y-2), QPointF(x, y+2), QPointF(x+4, y-2)]))


def label(text='', role=None):
    widget = QLabel(text)
    widget.setWordWrap(True)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    if role:
        widget.setProperty('role', role)
    return widget


def action(text, callback, primary=False):
    button = QPushButton(text)
    button.setMinimumHeight(36)
    if primary:
        button.setProperty('primary', True)
    button.clicked.connect(callback)
    return button


def spin(value, minimum=.001, maximum=1e6, decimals=3):
    widget = QDoubleSpinBox()
    widget.setRange(minimum, maximum)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setKeyboardTracking(False)
    return widget


class BodyInformation(QWidget):
    def __init__(self, preview_size=120):
        super().__init__()
        self.preview_size = preview_size
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumHeight(preview_size)
        self.name = label('', 'subheading')
        self.details = label('', 'muted')
        layout.addWidget(self.image)
        layout.addWidget(self.name)
        layout.addWidget(self.details)

    def set_spec(self, spec):
        rgba = star_sprite(spec) if spec.kind == 'star' else planet_sprite(spec, (1, 0))
        data = np.ascontiguousarray(np.clip(rgba*255, 0, 255).astype(np.uint8))
        image = QImage(data.data, data.shape[1], data.shape[0], data.strides[0], QImage.Format.Format_RGBA8888).copy()
        self.image.setPixmap(QPixmap.fromImage(image).scaled(self.preview_size, self.preview_size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self.name.setText(spec.name)
        mass = f'{spec.mass*SOLAR_KG:.5g} kg' if spec.mass is not None else 'Unknown'
        radius = f'{spec.radius*AU_KM:,.5g} km' if spec.radius is not None else 'Unknown'
        text = f'{spec.kind.title()}\nMass: {mass}\nRadius: {radius}'
        if spec.kind == 'star':
            text += f'\nTemperature: {spec.temperature:,.0f} K' if spec.temperature else '\nTemperature: unknown'
        self.details.setText(text)


class BodyCatalog(QWidget):
    add_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(label('Choose a body, review it, then add it to the scene.', 'muted'))
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.type_combo = Dropdown()
        self.type_combo.setAccessibleName('Body type')
        self.type_combo.addItems(TYPES)
        self.type_combo.setCurrentText('Planetary')
        self.body_combo = Dropdown()
        self.body_combo.setAccessibleName('Body definition')
        form.addRow('Type', self.type_combo)
        form.addRow('Body', self.body_combo)
        layout.addLayout(form)
        self.information = BodyInformation(preview_size=80)
        layout.addWidget(self.information)
        placement = QFormLayout()
        placement.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.reference_combo = Dropdown()
        self.reference_combo.setAccessibleName('Placement reference')
        self.reference_combo.setMinimumContentsLength(8)
        self.reference_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.motion_combo = Dropdown()
        self.motion_combo.setAccessibleName('Starting motion')
        self.motion_combo.addItem('Initial circular orbit', True)
        self.motion_combo.addItem('Set speed and direction', False)
        placement.addRow('Relative to', self.reference_combo)
        placement.addRow('Starting motion', self.motion_combo)
        layout.addLayout(placement)
        self.manual = QWidget()
        manual_form = QFormLayout(self.manual)
        manual_form.setContentsMargins(0, 0, 0, 0)
        self.speed = spin(0, 0, 1e6, 5)
        self.direction = spin(0, -360, 360, 2)
        manual_form.addRow('Speed (km/s)', self.speed)
        manual_form.addRow('Direction (°)', self.direction)
        layout.addWidget(self.manual)
        self.note = label('The circular option sets a starting velocity. Gravity determines the motion after that.', 'muted')
        layout.addWidget(self.note)
        self.add_button = action('Add', self.request_add, primary=True)
        layout.addWidget(self.add_button)
        layout.addStretch()
        self.type_combo.currentTextChanged.connect(self.populate_bodies)
        self.body_combo.currentIndexChanged.connect(self.show_body)
        self.motion_combo.currentIndexChanged.connect(self.motion_changed)
        self.populate_bodies()
        self.motion_changed()

    def populate_bodies(self):
        self.body_combo.blockSignals(True)
        self.body_combo.clear()
        for key, spec in CATALOG.items():
            if spec.kind in TYPES[self.type_combo.currentText()]:
                self.body_combo.addItem(spec.name, key)
        default = self.body_combo.findData('earth')
        if default >= 0:
            self.body_combo.setCurrentIndex(default)
        self.body_combo.blockSignals(False)
        self.show_body()

    def show_body(self):
        key = self.body_combo.currentData()
        if key:
            self.information.set_spec(CATALOG[key])
        self.add_button.setEnabled(key is not None)

    def set_references(self, prepared, selected=None):
        previous = self.reference_combo.currentData()
        had_reference = self.reference_combo.currentIndex() >= 0
        self.reference_combo.clear()
        lunar = self.type_combo.currentText() == 'Lunar'
        for key, spec in zip(prepared.body_ids, prepared.body_specs):
            if spec.mass and spec.mass > 0 and (not lunar or spec.kind == 'planet'):
                self.reference_combo.addItem(spec.name, key)
        if not lunar:
            self.reference_combo.addItem('World (no parent)', None)
        index = self.reference_combo.findData(selected if selected is not None else previous) if (selected is not None or had_reference) else 0
        self.reference_combo.setCurrentIndex(max(0, index))
        self.add_button.setEnabled(self.body_combo.currentData() is not None and self.reference_combo.count() > 0)

    def motion_changed(self):
        self.manual.setVisible(not self.motion_combo.currentData())

    def request_add(self):
        theta = math.radians(self.direction.value())
        velocity = self.speed.value()/AUYR_TO_KMS*np.array([math.cos(theta), math.sin(theta)])
        self.add_requested.emit({'spec': CATALOG[self.body_combo.currentData()],
            'reference_id': self.reference_combo.currentData(), 'circular': self.motion_combo.currentData(),
            'velocity': velocity})


class BodyProperties(QWidget):
    apply_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.body_id = None
        self.base_spec = None
        layout = QVBoxLayout(self)
        layout.addWidget(label('Edits change the frozen starting setup. The original experiment stays available.', 'muted'))
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.replacement = Dropdown()
        self.replacement.setAccessibleName('Replacement definition')
        self.name = QLineEdit()
        self.mass_factor = spin(1, .000001, 1e6, 6)
        self.mass_factor.setAccessibleName('Mass multiplier')
        self.mass_factor.setSuffix(' ×')
        self.mass_factor.setToolTip('Multiply the selected definition\'s mass; keep its starting motion unchanged.')
        form.addRow('Body definition', self.replacement)
        form.addRow('Name', self.name)
        form.addRow('Mass multiplier', self.mass_factor)
        layout.addLayout(form)
        self.mass_note = label('', 'muted')
        layout.addWidget(self.mass_note)
        self.precise = QGroupBox('Exact properties and starting motion')
        self.precise.setCheckable(True)
        self.precise.setChecked(False)
        self.exact = QWidget()
        exact_form = QFormLayout(self.exact)
        self.fields = {}
        for key, title in [('mass', 'Mass (kg)'), ('radius', 'Radius (km)'), ('temperature', 'Temperature (K)'),
                           ('x', 'Relative x (AU)'), ('y', 'Relative y (AU)'),
                           ('vx', 'Relative vx (km/s)'), ('vy', 'Relative vy (km/s)')]:
            field = QLineEdit()
            self.fields[key] = field
            exact_form.addRow(title, field)
        box = QVBoxLayout(self.precise)
        box.addWidget(self.exact)
        self.exact.setVisible(False)
        self.precise.toggled.connect(self.exact.setVisible)
        layout.addWidget(self.precise)
        self.note = label('', 'muted')
        layout.addWidget(self.note)
        self.apply_button = action('Apply to starting setup', self.request_apply, primary=True)
        layout.addWidget(self.apply_button)
        layout.addStretch()
        self.replacement.currentIndexChanged.connect(self.replace_definition)
        self.mass_factor.valueChanged.connect(self.mass_changed)

    def set_body(self, body):
        self.body_id, self.base_spec = body.id, body.spec
        self._original_spec = body.spec
        self.original_body = body
        self.replacement.blockSignals(True)
        self.replacement.clear()
        self.replacement.addItem('Keep current definition', None)
        for key, spec in CATALOG.items():
            if spec.kind == body.spec.kind or (body.spec.kind == 'particle' and spec.kind == 'asteroid'):
                self.replacement.addItem(spec.name, key)
        self.replacement.blockSignals(False)
        self.name.setText(body.spec.name)
        self.mass_factor.blockSignals(True)
        self.mass_factor.setValue(1)
        self.mass_factor.blockSignals(False)
        for key, value in [('x', body.r[0]), ('y', body.r[1]),
                           ('vx', body.v[0]*AUYR_TO_KMS), ('vy', body.v[1]*AUYR_TO_KMS)]:
            self.fields[key].setText(f'{value:.16g}')
        self.vector_texts = {key: self.fields[key].text() for key in ('x', 'y', 'vx', 'vy')}
        self.fill_physical()
        self.note.setText('Starting motion is preserved when you change mass or replace the body.')

    def fill_physical(self):
        for key, conversion in [('mass', SOLAR_KG), ('radius', AU_KM), ('temperature', 1)]:
            value = getattr(self.base_spec, key)
            if value is not None and key == 'mass':
                value *= self.mass_factor.value()
            self.fields[key].setText('' if value is None else f'{value*conversion:.16g}')
        self.mass_changed()

    def replace_definition(self):
        key = self.replacement.currentData()
        self.base_spec = CATALOG[key] if key else self.original_spec
        self.name.setText(self.base_spec.name)
        self.mass_factor.setValue(1)
        self.fill_physical()

    @property
    def original_spec(self):
        return self._original_spec

    def mass_changed(self):
        if self.base_spec is None:
            return
        mass = self.base_spec.mass
        if mass is not None:
            value = mass*self.mass_factor.value()*SOLAR_KG
            self.mass_note.setText(f'Resulting mass: {value:.5g} kg')
            self.fields['mass'].setText(f'{value:.16g}')

    def request_apply(self):
        try:
            values = {}
            for key, conversion in [('mass', SOLAR_KG), ('radius', AU_KM), ('temperature', 1)]:
                text = self.fields[key].text().strip()
                expected = getattr(self.base_spec, key)
                if expected is not None and key == 'mass':
                    expected *= self.mass_factor.value()
                # A display unit conversion must not edit an untouched value.
                values[key] = (expected if expected is not None and text == f'{expected*conversion:.16g}' else
                               float(text)/conversion if text else None)
            spec = override_properties(self.base_spec, name=self.name.text().strip(), **values)
            r = tuple(self.original_body.r[i] if self.fields[key].text() == self.vector_texts[key] else
                      float(self.fields[key].text()) for i, key in enumerate(('x', 'y')))
            v = tuple(self.original_body.v[i] if self.fields[key].text() == self.vector_texts[key] else
                      float(self.fields[key].text())/AUYR_TO_KMS for i, key in enumerate(('vx', 'vy')))
            self.apply_requested.emit({'body_id': self.body_id, 'spec': spec, 'r': r, 'v': v})
        except (ValueError, OverflowError) as exc:
            self.note.setText(f'Please check the values: {exc}')
