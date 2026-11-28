"""Native sidebar controls: catalog, body information and optional precise edits."""
import math

import numpy as np
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QLineEdit, QGroupBox, QCheckBox,
)

from .data import AU_KM, CATALOG, SOLAR_KG
from .model import BodySpec, PropertySource
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


def _parse_measure(text, unit_name):
    text = text.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f'Enter a finite number for {unit_name}.') from None
    if not np.isfinite(value):
        raise ValueError(f'Enter a finite number for {unit_name}.')
    return value


def custom_spec(name, kind, mass, radius, temperature):
    """A body typed in the observatory. Unknown fields stay unknown."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError('A custom body needs a name.')
    units = {'mass': 'solar masses', 'radius': 'AU', 'temperature': 'K'}
    fields = {'mass': mass, 'radius': radius}
    if kind == 'star':
        fields['temperature'] = temperature
    sources = {field: PropertySource(
        'user://custom', 'observatory', '', value, units[field], 'identity',
        status='user-authored' if value is not None else 'unknown',
        note='Entered in the observatory. Not a catalog measurement.')
        for field, value in fields.items()}
    return BodySpec(name.strip(), kind, mass, radius, temperature if kind == 'star' else None,
                    provenance='custom', sources=sources)


def _property_text(spec):
    """Physical values with units, and the catalog status behind each one."""
    conversions = {'mass': SOLAR_KG, 'radius': AU_KM, 'temperature': 1.}
    units = {'mass': 'kg', 'radius': 'km', 'temperature': 'K'}
    titles = {'mass': 'Mass', 'radius': 'Radius', 'temperature': 'Temperature'}
    fields = ('mass', 'radius', 'temperature') if spec.kind == 'star' else ('mass', 'radius')
    lines = [spec.kind.title()]
    if spec.provenance != 'catalog':
        lines.append('Custom values are marked as entered here, not as catalog measurements.')
    for field in fields:
        value = getattr(spec, field)
        if value is None:
            if field == 'temperature':
                shown = 'unknown'
            else:
                shown = 'Unknown — not estimated'
                if field == 'radius':
                    shown += '. The marker is a drawing, not a physical radius'
        else:
            shown = f'{value*conversions[field]:,.5g} {units[field]}'
        lines.append(f'{titles[field]}: {shown}')
        source = spec.sources.get(field) if spec.sources else None
        if source is None:
            continue
        detail = source.status or 'unspecified'
        uncertainty = source.uncertainty
        if uncertainty not in (None, '', '(None, None)') and not (
                isinstance(uncertainty, tuple) and all(item is None for item in uncertainty)):
            detail += f'; uncertainty {uncertainty} ({source.original_units})'
        lines.append(f'  {detail}')
        if source.url:
            lines.append(f'  {source.url}')
    if spec.kind == 'star' and spec.catalog_key is None and spec.provenance and spec.provenance != 'catalog':
        lines.append(spec.provenance)
    return '\n'.join(lines)


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
        self.details.setText(_property_text(spec))


class BodyCatalog(QWidget):
    add_requested = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(label('Choose a body, review it, then add it to the scene.', 'muted'))
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.type_combo = Dropdown()
        self.type_combo.addItems(TYPES)
        self.type_combo.setCurrentText('Planetary')
        self.body_combo = Dropdown()
        form.addRow('Type', self.type_combo)
        form.addRow('Body', self.body_combo)
        layout.addLayout(form)
        self.information = BodyInformation(preview_size=80)
        layout.addWidget(self.information)
        placement = QFormLayout()
        placement.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.reference_combo = Dropdown()
        self.reference_combo.setMinimumContentsLength(8)
        self.reference_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.motion_combo = Dropdown()
        self.motion_combo.addItem('Initial circular orbit', True)
        self.motion_combo.addItem('Set speed and direction', False)
        self.reference_label = label('Relative to')
        placement.addRow(self.reference_label, self.reference_combo)
        placement.addRow('Starting motion', self.motion_combo)
        layout.addLayout(placement)
        self.distance_box = QWidget()
        distance_form = QFormLayout(self.distance_box)
        distance_form.setContentsMargins(0, 0, 0, 0)
        distance_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.distance = spin(0, 0, 1e9, 1)
        distance_form.addRow('Distance from parent (km)', self.distance)
        layout.addWidget(self.distance_box)
        self.manual = QWidget()
        manual_form = QFormLayout(self.manual)
        manual_form.setContentsMargins(0, 0, 0, 0)
        self.speed = spin(0, 0, 1e6, 5)
        self.direction = spin(0, -360, 360, 2)
        manual_form.addRow('Speed (km/s)', self.speed)
        manual_form.addRow('Direction (°)', self.direction)
        layout.addWidget(self.manual)
        self.custom_check = QCheckBox('Custom body')
        self.custom_check.setToolTip('Type a name and physical properties instead of using a catalog entry.')
        layout.addWidget(self.custom_check)
        self.custom_box = QWidget()
        custom_form = QFormLayout(self.custom_box)
        custom_form.setContentsMargins(0, 0, 0, 0)
        custom_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.custom_name = QLineEdit('Custom body')
        self.custom_kind = Dropdown()
        for title, kind in (('Planet', 'planet'), ('Star', 'star'), ('Moon', 'moon'), ('Asteroid', 'asteroid')):
            self.custom_kind.addItem(title, kind)
        self.custom_mass = QLineEdit('1e21')
        self.custom_mass.setPlaceholderText('kilograms, required')
        self.custom_radius = QLineEdit()
        self.custom_radius.setPlaceholderText('kilometers, blank if unknown')
        self.custom_temperature = QLineEdit('5772')
        self.temperature_holder = QWidget()
        temperature_form = QFormLayout(self.temperature_holder)
        temperature_form.setContentsMargins(0, 0, 0, 0)
        temperature_form.addRow('Temperature (K)', self.custom_temperature)
        custom_form.addRow('Name', self.custom_name)
        custom_form.addRow('Kind', self.custom_kind)
        custom_form.addRow('Mass (kg)', self.custom_mass)
        custom_form.addRow('Radius (km)', self.custom_radius)
        custom_form.addRow(self.temperature_holder)
        self.custom_box.setVisible(False)
        self.temperature_holder.setVisible(False)
        layout.addWidget(self.custom_box)
        self.note = label('', 'muted')
        layout.addWidget(self.note)
        self.draft_readout = label('', 'notice')
        layout.addWidget(self.draft_readout)
        self.add_button = action('Add', self.request_add, primary=True)
        layout.addWidget(self.add_button)
        layout.addStretch()
        self.type_combo.currentTextChanged.connect(self.populate_bodies)
        self.body_combo.currentIndexChanged.connect(self.show_body)
        self.motion_combo.currentIndexChanged.connect(self.motion_changed)
        self.custom_check.toggled.connect(self._custom_toggled)
        self.custom_kind.currentIndexChanged.connect(self._kind_changed)
        for field in (self.custom_name, self.custom_mass, self.custom_radius, self.custom_temperature):
            field.editingFinished.connect(self.show_body)
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
        self._refresh_note()
        self.show_body()
        self.sync_lunar_fields()

    def sync_lunar_fields(self):
        lunar = self._lunar_reference()
        self.distance_box.setVisible(lunar)
        self.reference_label.setText('Parent planet' if lunar else 'Relative to')
        if lunar:
            self.note.setText('Distance is center-to-center from the parent planet, in kilometers. '
                              'It is not that planet\'s distance from its star. '
                              'A circular start is only a first guess; gravity determines the motion.')

    def _custom_toggled(self, checked):
        self.custom_box.setVisible(checked)
        self.body_combo.setEnabled(not checked)
        self._kind_changed()

    def _kind_changed(self):
        self.temperature_holder.setVisible(self.custom_check.isChecked() and self.custom_kind.currentData() == 'star')
        self._refresh_note()
        if getattr(self, '_prepared', None) is not None:
            self.set_references(self._prepared, getattr(self, '_selected', None))
        else:
            self.show_body()

    def _refresh_note(self):
        self.sync_lunar_fields()
        if self._lunar_reference():
            return
        stellar = self.type_combo.currentText() == 'Stellar' or (
            self.custom_check.isChecked() and self.custom_kind.currentData() == 'star')
        text = ('The circular option is a two-body start around the chosen body. '
                'Gravity takes over afterward and does not guarantee an orbit.')
        if stellar:
            text += ' Another star is an ordinary body. The same aid does not solve a binary.'
        self.note.setText(text)

    def _lunar_reference(self):
        return self.type_combo.currentText() == 'Lunar' or (
            self.custom_check.isChecked() and self.custom_kind.currentData() == 'moon')

    def show_body(self):
        if self.custom_check.isChecked():
            try:
                self.information.set_spec(self.custom_definition())
                ready = True
            except ValueError as exc:
                self.information.name.setText(self.custom_name.text().strip() or 'Custom body')
                self.information.details.setText(str(exc))
                ready = False
            self.add_button.setEnabled(ready and self.reference_combo.count() > 0)
            return
        key = self.body_combo.currentData()
        if key:
            self.information.set_spec(CATALOG[key])
        self.add_button.setEnabled(key is not None and self.reference_combo.count() > 0)

    def custom_definition(self):
        mass_kg = _parse_measure(self.custom_mass.text(), 'mass (kg)')
        if mass_kg is None:
            raise ValueError('Enter a mass in kilograms. Gravity needs a known mass.')
        radius_km = _parse_measure(self.custom_radius.text(), 'radius (km)')
        temperature = _parse_measure(self.custom_temperature.text(), 'temperature (K)')
        try:
            return custom_spec(self.custom_name.text(), self.custom_kind.currentData(),
                               mass_kg/SOLAR_KG, None if radius_km is None else radius_km/AU_KM, temperature)
        except ValueError as exc:
            raise ValueError(str(exc)) from None

    def set_references(self, prepared, selected=None):
        self._prepared, self._selected = prepared, selected
        previous = self.reference_combo.currentData()
        had_reference = self.reference_combo.currentIndex() >= 0
        self.reference_combo.blockSignals(True)
        self.reference_combo.clear()
        lunar = self._lunar_reference()
        for key, spec in zip(prepared.body_ids, prepared.body_specs):
            if spec.mass and spec.mass > 0 and (not lunar or spec.kind == 'planet'):
                self.reference_combo.addItem(spec.name, key)
        if not lunar:
            self.reference_combo.addItem('World (no parent)', None)
        index = self.reference_combo.findData(selected if selected is not None else previous) if (selected is not None or had_reference) else 0
        self.reference_combo.setCurrentIndex(max(0, index))
        self.reference_combo.blockSignals(False)
        self.show_body()

    def motion_changed(self):
        self.manual.setVisible(not bool(self.motion_combo.currentData()))

    def placement_payload(self):
        theta = math.radians(self.direction.value())
        velocity = self.speed.value()/AUYR_TO_KMS*np.array([math.cos(theta), math.sin(theta)])
        lunar = self._lunar_reference()
        spec = self.custom_definition() if self.custom_check.isChecked() else CATALOG[self.body_combo.currentData()]
        return {'spec': spec, 'reference_id': self.reference_combo.currentData(),
                'circular': bool(self.motion_combo.currentData()), 'velocity': velocity,
                'distance_km': self.distance.value() if lunar else None}

    def request_add(self):
        try:
            self.add_requested.emit(self.placement_payload())
        except ValueError as exc:
            self.note.setText(str(exc))


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
        self.name = QLineEdit()
        self.mass_factor = spin(1, .000001, 1e6, 6)
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
        if body.spec.kind == 'moon':
            distance = float(np.linalg.norm(body.r))*AU_KM
            self.note.setText(f'Starting distance from the parent: {distance:,.6g} km. '
                              'Relative x/y are that planet-centered distance in AU, not the planet\'s position from its star. '
                              'Starting motion is preserved when you change mass or replace the body.'
                              + (' Overhead fields edit x and y. The stored z component stays.' if len(body.r) == 3 else ''))
        else:
            self.note.setText('Starting motion is preserved when you change mass or replace the body.'
                              + (' Overhead fields edit x and y. The stored z component stays.' if len(body.r) == 3 else ''))

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
            if len(self.original_body.r) == 3:
                r = r + (float(self.original_body.r[2]),)
            if len(self.original_body.v) == 3:
                v = v + (float(self.original_body.v[2]),)
            self.apply_requested.emit({'body_id': self.body_id, 'spec': spec, 'r': r, 'v': v})
        except (ValueError, OverflowError) as exc:
            self.note.setText(f'Please check the values: {exc}')
