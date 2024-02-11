from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUiType
from os import path
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pickle
import pyqtgraph as pg
from scipy import signal

FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "design.ui"))

x_axis_offset = 10

class FilterDesign(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(FilterDesign, self).__init__(parent)
        QMainWindow.__init__(self, parent=None)
        self.setupUi(self)
        self.setWindowTitle("Filter Design")
        self.zplaneCanvas = ZplaneCanvas(self, self.dragRadioButton, self.deleteRadioButton, self.zplaneWidget)
        self.zplaneLayout.addWidget(self.zplaneCanvas)
        self.deleteFilterButton.setEnabled(0)
        self.clearComboBox.setCurrentText("Clear")
        self.filtersComboBox.setCurrentText("Choose Filter")

        self.conjugateRadioButton.toggled.connect(self.updateConjugateMode)
        self.clearComboBox.currentIndexChanged.connect(self.clearPoints)
        
        # Set default tab to Filter Design tab
        self.tabWidget.setCurrentIndex(0)

        # Set up icons for buttons & sliders
        self.deleteIcon = QtGui.QIcon("icons/deleteIcon.png")
        self.confirmIcon = QtGui.QIcon("icons/confirmIcon.png")
        
        # Set the text for radio buttons
        self.zerosRadioButton.setText("O\nZeros")
        self.polesRadioButton.setText("X\nPoles")
        self.dragRadioButton.setText("Drag\nSymbol")
        
        # Set icons for buttons
        # self.deleteButton.setIcon(self.deleteIcon)
        self.deleteFilterButton.setIcon(self.deleteIcon)
        self.displayFilterButton.setIcon(self.confirmIcon)
        
        # Apply style sheet for sliders
        self.slidersStyleHorizontal1 = "QSlider::groove:horizontal { border: 1px solid #999999; background: white; width: 8px; border-radius: 4px; }"
        self.slidersStyleHorizontal2 = "QSlider::handle:horizontal { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #d3d3d3, stop:1 #c0c0c0); border: 1px solid #5c5c5c; width: 8px; height: 14px; margin: -2px 0; border-radius: 4px; }"

        self.scene = QGraphicsScene(self)
        self.touchPad.setScene(self.scene)
       
        ##################### Sliders ##################
        self.speedSlider.setStyleSheet(self.slidersStyleHorizontal1)
        self.speedSlider.setStyleSheet(self.slidersStyleHorizontal2)
        self.speedSlider.valueChanged.connect(self.updateSpeedValueLabel)
        self.speedSlider.setValue(50)
        self.speedSlider.valueChanged.connect(self.update_playback_speed_graph)
        ##################### Sliders ##################
        
        ##################### Connections ##################
        self.addFilterButton.clicked.connect(self.addFilterOption)
        self.filtersComboBox.currentIndexChanged.connect(self.updateDeleteButton)
        self.deleteFilterButton.clicked.connect(self.deleteFilterOption)
        self.pushButton.clicked.connect(self.delete_all_pass)
        self.zerosRadioButton.toggled.connect(self.setZerosMode)
        self.polesRadioButton.toggled.connect(self.setPolesMode)
        self.uploadSignalButton.clicked.connect(self.open_file)
        self.displayFilterButton.clicked.connect(self.update_plot_Allpass)
        self.applyFiltersButton.clicked.connect(self.apply)
        self.touchPad.mouseMoveEvent = self.mouse_move_event
        ##################### Connections ##################
        
        ##################### Variables ##################
        self.zeros_positions = []
        self.poles_positions = []
        self.original_signal_data = []
        self.filtered_signal_data = []
        self.allpasszeros = []
        self.allpasspoles = []
        ##################### Variables ##################

        ##################### Real-time Plotting ##################
        self.update_interval_graph1 = 200
        self.update_interval_graph2 = 200
        self.update_interval_ms = 300
        self.timer = self.create_timer()
        self.plot_widget1 = pg.PlotWidget()
        self.plot_widget2 = pg.PlotWidget()
        self.plot_widget3 = pg.PlotWidget()
        self.plot_widget4 = pg.PlotWidget()
        self.plot_widget5 = pg.PlotWidget()
        self.plot_widget6 = pg.PlotWidget()
        self.realTimeLayout.addWidget(self.plot_widget1)
        self.filteredLayout.addWidget(self.plot_widget2)
        self.magnitudeLayout.addWidget(self.plot_widget3)
        self.phaseLayout.addWidget(self.plot_widget4)
        self.filtersLayout.addWidget(self.plot_widget5)
        self.finalPhaseLayout.addWidget(self.plot_widget6)
        self.current_set_start_index = 0
        ##################### Real-time Plotting ##################

        ##################### Artificial Plotting ##################
        self.initiate_graph(self.plot_widget1)
        self.initiate_graph(self.plot_widget2)
        self.generated_signal = np.array([])
        self.filtered_generated_signal = np.array([])
        self.generated_timer = QTimer(self)
        self.generated_timer.timeout.connect(self.update_plot)
        self.generated_timer.start(50)
        self.generated_current_index = 0    
        ##################### Artificial Plotting ##################

    def mouse_move_event(self, event):
        y = event.pos().y()
        self.generated_signal = np.append(self.generated_signal, y)
        self.filtered_generated_signal = self.applyFilter(self.generated_signal)

    def update_plot(self):
            self.generated_timer.setInterval(50)
            self.generated_current_index += 1
            if len(self.generated_signal) >= 2:
                self.plot_widget1.plot(self.generated_signal, pen=pg.mkPen('r'))
                self.plot_widget2.plot(np.abs(self.filtered_generated_signal), pen=pg.mkPen('g'))
                self.adjustView(self.plot_widget1)
                self.adjustView(self.plot_widget2)

    def adjustView(self, plot_widget):
        view_box = plot_widget.getViewBox()
        current_view = view_box.viewRange()[0]
        view_width = current_view[1] - current_view[0]
        new_view_x = [self.generated_current_index - view_width * 0.5, self.generated_current_index + view_width * 0.5]
        new_view_x[0] -= x_axis_offset
        new_view_x[1] -= x_axis_offset
        view_box.setXRange(new_view_x[0], new_view_x[1], padding=0)

    def initiate_graph(self, plot_widget):
        plot_widget.clear()
        plot_widget.setBackground('k')
        view_box1 = plot_widget.getViewBox()
        view_box1.setLimits(xMin=0)
        view_box1.setMouseEnabled(x=False, y=True)
        view_box1.setRange(xRange=[0, 10], yRange=[-500,500], padding=0.05)
        plot_widget.plotItem.getViewBox().scaleBy((6, 1))

    def plot_mag_phase(self):
        self.plot_widget3.clear()
        self.plot_widget4.clear()
        self.plot_widget6.clear()
        zeros = [complex(z[0], z[1]) for z in filter_design.zeros_positions]
        poles = [complex(p[0], p[1]) for p in filter_design.poles_positions]
        if filter_design.zeros_positions or filter_design.poles_positions:
            # Convert zeros, poles, and gain to numerator and denominator coefficients
            num, den = signal.zpk2tf(zeros, poles, 1)

            # Compute the frequency response
            self.w, FreqResp = signal.freqz(num, den)

        self.plot_widget3.plot(self.w, 20 * np.log10(abs(FreqResp)))
        self.plot_widget4.plot(self.w, np.angle(FreqResp, deg=True))
        self.plot_widget6.plot(self.w, np.angle(FreqResp, deg=True))

    def represent_allpass(self, a):
        z, p, k = signal.tf2zpk([-a, 1.0], [1.0, -a])
        self.allpasszeros = z
        self.allpasspoles = p
        allpass_frequencies, allpass_transfer_function = signal.freqz([-a, 1.0], [1.0, -a])
        phase_response = np.unwrap(np.angle(allpass_transfer_function))
        self.plot_widget5.clear()
        self.plot_widget5.plot(allpass_frequencies, phase_response, pen='r')

    def delete_all_pass(self):
        current_index = self.comboBox.currentIndex()
        if current_index != -1:
            selected_text = self.comboBox.itemText(current_index)
            selected_text = selected_text.replace(" ", "")  # Remove spaces
            a = complex(selected_text)
            z, p, k = signal.tf2zpk([-a, 1.0], [1.0, -a])
            # Find the first occurrence of a in zeros_positions and remove it
            found_zero = False
            for zero in self.zeros_positions:
                if complex(zero[0], zero[1]) == z and not found_zero:
                    self.zeros_positions.remove(zero)
                    found_zero = True

            # Find the first occurrence of a in poles_positions and remove it
            found_pole = False
            for pole in self.poles_positions:
                if complex(pole[0], pole[1]) == p and not found_pole:
                    self.poles_positions.remove(pole)
                    found_pole = True

            # Update the plot after modification
            self.plot_mag_phase()
            self.comboBox.removeItem(current_index)

    def update_plot_Allpass(self):
            selected_text = self.filtersComboBox.currentText()
            selected_text = selected_text.replace(" ", "")  # Remove spaces
            a = complex(selected_text)
            self.represent_allpass(a)

    def apply(self):
        for zero,pole in zip(self.allpasszeros,self.allpasspoles):
            self.zeros_positions.append([zero.real, zero.imag])
            self.poles_positions.append([pole.real, pole.imag])
        selected_text = self.filtersComboBox.currentText()
        selected_text = selected_text.replace(" ", "")  # Remove spaces
        self.comboBox.addItem(selected_text)
        self.plot_mag_phase()

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Signal File", "", "Signal Files (*.pkl);;All Files (*)")
        if file_name:
            with open(file_name, 'rb') as file:
                file_data = pickle.load(file)
            self.reset()
            self.initiate_graph(self.plot_widget1)
            self.initiate_graph(self.plot_widget2)
            self.load_signal_for_graph(file_data)

    def load_signal_for_graph(self, signal_data):
        self.original_signal_data.append(signal_data)
        self.generated_timer.stop()
        self.applyFilterToGraph2()
        self.timer.start(self.update_interval_ms)
        self.current_index_graph1 = 0
        self.current_index_graph2 = 0

    def create_timer(self):
        timer = QTimer()
        timer.timeout.connect(self.timerEvent)
        return timer

    def timerEvent(self):
        self.timer.setInterval(self.update_interval_graph1)
        self.plot_signal(self.plot_widget1)
        self.current_index_graph1 += 1
        self.plot_signal(self.plot_widget2)
        self.current_index_graph2 += 1

    def reset(self):
        self.plot_widget1.clear()
        self.plot_widget2.clear()
        self.original_signal_data.clear()
        self.filtered_signal_data.clear()

    def update_playback_speed_graph(self, value):
        min_speed = 0.25
        max_speed = 4
        speed_multiplier = min_speed + (max_speed - min_speed) * (value / 100.0)
        self.update_interval_graph1 = int(self.update_interval_ms / speed_multiplier)
        self.update_interval_graph2 = int(self.update_interval_ms / speed_multiplier)
        self.speedSlider.setValue(value)

    def plot_signal(self, plot_widget):
        if plot_widget == self.plot_widget1:
            signal_data_list = self.original_signal_data
            current_index = self.current_index_graph1
            color = 'r'
        elif plot_widget == self.plot_widget2:
            signal_data_list = self.filtered_signal_data
            current_index = self.current_index_graph2
            color = 'b'
            
        if signal_data_list:
            min_value = float('inf')
            max_value = float('-inf')

            for i, signal_data in enumerate(signal_data_list):
                start = max(0, current_index - int(1000 * 1.0))
                data_to_plot_temp = signal_data[start:current_index + 1]

                if len(data_to_plot_temp) > 0:

                    min_value = min(min_value, np.min(data_to_plot_temp))
                    max_value = max(max_value, np.max(data_to_plot_temp))

                    plot_widget.plot(data_to_plot_temp, pen=pg.mkPen(color))

            if min_value != float('inf') and max_value != float('-inf'):
                plot_widget.setYRange(min_value, max_value)
            else:
                plot_widget.setYRange(0, 1)

        view_box = plot_widget.getViewBox()
        current_view = view_box.viewRange()[0]
        view_width = current_view[1] - current_view[0]
        new_view_x = [current_index - view_width * 0.5, current_index + view_width * 0.5]
        x_axis_offset = 31
        new_view_x[0] -= x_axis_offset
        new_view_x[1] -= x_axis_offset
        view_box.setXRange(new_view_x[0], new_view_x[1], padding=0)

    def extractFilterCoefficients(self):
        zeros = filter_design.zeros_positions
        poles = filter_design.poles_positions

        # Extract the coefficients for the numerator (zeros) and denominator (poles)
        real_zeros = [z[0] for z in zeros]
        imag_zeros = [z[1] for z in zeros]
        real_poles = [p[0] for p in poles]
        imag_poles = [p[1] for p in poles]

        numerator_coefficients = np.poly(np.array(real_zeros) + 1j * np.array(imag_zeros)) if zeros else [1]
        denominator_coefficients = np.poly(np.array(real_poles) + 1j * np.array(imag_poles)) if poles else [1]

        return numerator_coefficients, denominator_coefficients
    
    def applyFilter(self, input_signal):
            numerator_coefficients, denominator_coefficients = self.extractFilterCoefficients()
            output_signal = signal.lfilter(numerator_coefficients, denominator_coefficients, input_signal)
            return output_signal
    
    def applyFilterToGraph2(self):
        input_signal = self.original_signal_data[-1]
        filtered_signal = np.abs(self.applyFilter(input_signal))
        self.filtered_signal_data.append(filtered_signal)

    def updateConjugateMode(self, state):
     state = self.conjugateRadioButton.isChecked()

     if state:
        self.zplaneCanvas.setConjugateMode(True)
     else:
        self.zplaneCanvas.setConjugateMode(False)

    def updateSpeedValueLabel(self, value):
        self.speedValueLabel.setText(f"{value}x")

    def updateDeleteButton(self):
        self.deleteFilterButton.setEnabled(self.filtersComboBox.currentIndex() != -1)

    def deleteFilterOption(self):
        current_index = self.filtersComboBox.currentIndex()
        if current_index != -1:
            self.filtersComboBox.removeItem(current_index)

    def addFilterOption(self):
        real_value = self.realSpinBox.value()
        imaginary_value = self.imaginarySpinBox.value()
        filter_option = f"{real_value} + {imaginary_value}j"
        self.filtersComboBox.addItem(filter_option)

    def setZerosMode(self, checked):
        if checked:
            self.zplaneCanvas.setMode("zeros")
            self.zplaneCanvas.setConjugateMode(False)

    def setPolesMode(self, checked):
        if checked:
            self.zplaneCanvas.setMode("poles")
            self.zplaneCanvas.setConjugateMode(False)

    def clearPoints(self, index):
        if index == 0:  
            self.zplaneCanvas.clearZeros()
        elif index == 1:  
            self.zplaneCanvas.clearPoles()
        elif index == 2:  
            self.zplaneCanvas.clearAllPoints()
        self.restore_generation()
        filter_design.plot_mag_phase()
        self.clearComboBox.setCurrentIndex(-1)

    def restore_generation(self):
        self.generated_timer.stop()
        self.plot_widget1.clear()
        self.plot_widget2.clear()
        self.generated_signal = np.array([])
        self.filtered_generated_signal = np.array([])
        self.generated_timer.start(100)

class ZplaneCanvas(FigureCanvas):
    def __init__(self, filter_design, drag_radio_button, delete_radio_button, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        super(ZplaneCanvas, self).__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.filter_design = filter_design
        self.drag_radio_button = drag_radio_button
        self.delete_radio_button = delete_radio_button
        self.plot_zplane()
        self.mpl_connect('button_press_event', self.place)
        self.mpl_connect('motion_notify_event', self.on_drag)
        self.mpl_connect('button_release_event', self.on_release)
        self.selected_symbol = None
        self.mpl_connect('pick_event', self.on_pick)
        self.threshold = 0.2
        self.drag_cid = None
        
        # Dictionary to store symbols and their corresponding artists
        self.zeros_symbols = {}
        self.poles_symbols = {}

        # Mode: "zeros" or "poles"
        self.mode = "zeros"

        # Connect pick and motion notify events
        self.connect_events()
        
        # Counters for zeros and poles
        self.zeros_count = 0
        self.poles_count = 0
        
        # Currently selected symbol
        self.selected_symbol = None
        self.conjugate_mode = False

    def on_drag(self, event):
        if event.inaxes and self.selected_symbol:
            x, y = event.xdata, event.ydata

            initial_position = self.selected_symbol.get_offsets()[0]

            self.selected_symbol.set_offsets([(x, y)])

            closest_index = None
            closest_distance = float('inf')

            if self.selected_symbol in self.zeros_symbols:
                # Dragging a zero symbol
                symbols = list(self.zeros_symbols.keys())
            elif self.selected_symbol in self.poles_symbols:
                # Dragging a pole symbol
                symbols = list(self.poles_symbols.keys())
            else:
                return

            for i, symbol in enumerate(symbols):
                pos = symbols[i].get_offsets()[0]
                distance = np.linalg.norm(np.array(initial_position) - np.array(pos))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_index = i

            if closest_index is not None:
                if self.selected_symbol in self.zeros_symbols:
                    # Update the position of the dragged zero symbol in the dictionary
                    self.zeros_symbols[self.selected_symbol] = (x, y)
                    filter_design.zeros_positions[closest_index] = (x, y, self.selected_symbol)
                elif self.selected_symbol in self.poles_symbols:
                    # Update the position of the dragged pole symbol in the dictionary
                    self.poles_symbols[self.selected_symbol] = (x, y)
                    filter_design.poles_positions[closest_index] = (x, y, self.selected_symbol)

            filter_design.plot_mag_phase()
            self.draw()
            filter_design.restore_generation()


    def on_drag_conjugate(self, event):
        if event.inaxes and self.selected_symbol:
            x, y = event.xdata, event.ydata

            initial_position = self.selected_symbol.get_offsets()[0]

            self.selected_symbol.set_offsets([(x, y)])

            closest_index = None
            closest_distance = float('inf')

            # Get the list of symbols and positions based on the selected symbol
            if self.selected_symbol in self.zeros_symbols:
                symbols = list(self.zeros_symbols.keys())
                positions = filter_design.zeros_positions
            elif self.selected_symbol in self.poles_symbols:
                symbols = list(self.poles_symbols.keys())
                positions = filter_design.poles_positions
            else:
                return

            # Find the closest symbol
            for i, symbol in enumerate(symbols):
                pos = symbols[i].get_offsets()[0]
                distance = np.linalg.norm(np.array(initial_position) - np.array(pos))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_index = i

            # Update the position of the dragged symbol and its conjugate (if it exists)
            if closest_index is not None:
                dragged_symbol = symbols[closest_index]
                dragged_position = positions[closest_index]

                dragged_symbol.set_offsets([(x, y)])

                if self.selected_symbol in self.zeros_symbols:
                    # Update the position of the dragged zero symbol and its conjugate
                    self.zeros_symbols[dragged_symbol] = (x, y)
                    positions[closest_index] = (x, y, dragged_symbol)

                    # Update the position of the conjugate if it exists
                    conjugate_symbol = dragged_position[2]
                    if conjugate_symbol in self.zeros_symbols:  # Check if the conjugate exists
                        conjugate_x, conjugate_y = self.calculate_conjugate(x, y)
                        conjugate_symbol.set_offsets([(conjugate_x, conjugate_y)])
                        self.zeros_symbols[conjugate_symbol] = (conjugate_x, conjugate_y, dragged_symbol)
                        filter_design.zeros_positions[closest_index] = (conjugate_x, conjugate_y, conjugate_symbol)
                elif self.selected_symbol in self.poles_symbols:
                    # Update the position of the dragged pole symbol and its conjugate
                    self.poles_symbols[dragged_symbol] = (x, y)
                    positions[closest_index] = (x, y, dragged_symbol)

                    # Update the position of the conjugate if it exists
                    conjugate_symbol = dragged_position[2]
                    if conjugate_symbol in self.poles_symbols:  # Check if the conjugate exists
                        conjugate_x, conjugate_y = self.calculate_conjugate(x, y)
                        conjugate_symbol.set_offsets([(conjugate_x, conjugate_y)])
                        self.poles_symbols[conjugate_symbol] = (conjugate_x, conjugate_y, dragged_symbol)
                        filter_design.poles_positions[closest_index] = (conjugate_x, conjugate_y, conjugate_symbol)

            filter_design.plot_mag_phase()
            self.draw()
            filter_design.restore_generation()

    def on_release(self, event):
        if self.selected_symbol and event.name == 'button_release_event':
            self.mpl_disconnect(self.drag_cid)
            self.drag_cid = self.mpl_connect('motion_notify_event', self.on_drag)
            self.selected_symbol = None
            filter_design.restore_generation()

    def on_pick(self, event):
        if event.artist.get_label() == 'Zeros' and self.zerosRadioButton.isChecked():
            self.selected_symbol = event.artist
        elif event.artist.get_label() == 'Poles' and self.polesRadioButton.isChecked():
            self.selected_symbol = event.artist
        filter_design.restore_generation()

    def plot_conjugate(self, x, y, original_symbol, flag):
        if flag == 1:
            conjugate_symbol = self.ax.scatter(x, y, marker='o', color='r', label='Conjugate Zeros')
            self.zeros_symbols[conjugate_symbol] = (x, y, original_symbol)
        else:
            conjugate_symbol = self.ax.scatter(x, y, marker='x', color='b', label='Conjugate Poles')
            self.poles_symbols[conjugate_symbol] = (x, y, original_symbol)
        self.draw()
        self.filter_design.generated_signal.fill(0)
        self.filter_design.filtered_generated_signal.fill(0)

        filter_design.restore_generation()
        return conjugate_symbol

    def calculate_conjugate(self, x, y):
        x_conjugate = x
        y_conjugate = -y
        return x_conjugate, y_conjugate

    def setConjugateMode(self, mode):
        self.conjugate_mode = mode

    def place(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata

            if self.selected_symbol:
                self.mpl_disconnect(self.drag_cid)
                self.selected_symbol = None
            else:
                if self.mode == "zeros" and filter_design.conjugateRadioButton.isChecked():
                    self.zeros_count += 1
                    symbol = self.plotZero(x, y)
                    x_conjugate, y_conjugate = self.calculate_conjugate(x, y)
                    conjugate_symbol = self.plot_conjugate(x_conjugate, y_conjugate, symbol, 1)
                    filter_design.zeros_positions.append((x_conjugate, y_conjugate, conjugate_symbol))
                    self.draw()

                elif self.mode == "poles" and filter_design.conjugateRadioButton.isChecked():
                    self.poles_count += 1
                    symbol = self.plotPole(x, y)
                    x_conjugate, y_conjugate = self.calculate_conjugate(x, y)
                    conjugate_symbol = self.plot_conjugate(x_conjugate, y_conjugate, symbol, 2)
                    filter_design.poles_positions.append((x_conjugate, y_conjugate, conjugate_symbol))
                    self.draw()
    
                else:
                    if self.drag_radio_button.isChecked():
                        nearest_symbol = self.find_nearest_symbol(x, y)
                        nearest_distance = np.sqrt((nearest_symbol[0] - x)**2 + (nearest_symbol[1] - y)**2) if nearest_symbol else float('inf')
                        if nearest_distance < self.threshold:
                            self.selected_symbol = nearest_symbol[2]
                            self.drag_cid = self.mpl_connect('motion_notify_event', self.on_drag)

                    elif self.delete_radio_button.isChecked():
                        nearest_symbol = self.find_nearest_symbol(x, y)
                        nearest_distance = np.sqrt((nearest_symbol[0] - x)**2 + (nearest_symbol[1] - y)**2) if nearest_symbol else float('inf')
                        if nearest_distance < self.threshold:
                            self.remove_symbol(nearest_symbol)
                    else:
                        if self.mode == "zeros" and not self.conjugate_mode:
                            self.zeros_count += 1
                            symbol = self.plotZero(x, y)

                        elif self.mode == "poles" and not self.conjugate_mode:
                            self.poles_count += 1
                            symbol = self.plotPole(x, y)

                        self.draw()

        self.place_cid = self.mpl_connect('button_press_event', self.place)
        filter_design.restore_generation()


    def remove_symbol(self, symbol):
        if symbol in filter_design.zeros_positions:
            filter_design.zeros_positions.remove(symbol)
            self.zeros_symbols.pop(symbol[2], None)
        elif symbol in filter_design.poles_positions:
            filter_design.poles_positions.remove(symbol)
            self.poles_symbols.pop(symbol[2], None)
        if len(filter_design.zeros_positions) == 0 and len(filter_design.poles_positions) == 0:
            self.clearAllPoints()
        filter_design.plot_mag_phase()
        symbol[2].remove()
        self.draw()
        filter_design.restore_generation()

    def find_nearest_symbol(self, x, y):
        zeros = list(self.zeros_symbols.values())
        poles = list(self.poles_symbols.values())
        symbols = zeros + poles
        if not symbols:
            return None
        
        # Extract (x, y) coordinates from each tuple
        coordinates = [(symbol[0], symbol[1]) for symbol in symbols]

        distances = np.sqrt((np.array(coordinates)[:, 0] - x)**2 + (np.array(coordinates)[:, 1] - y)**2)
        min_distance_index = np.argmin(distances)

        if min_distance_index < len(zeros):
            if min_distance_index < len(filter_design.zeros_positions):
                return filter_design.zeros_positions[min_distance_index]
        elif min_distance_index < len(zeros) + len(poles):
            if min_distance_index - len(zeros) < len(filter_design.poles_positions):
                return filter_design.poles_positions[min_distance_index - len(zeros)]

        filter_design.restore_generation()
        return None

    def clearZeros(self):
        for symbol in self.zeros_symbols:
            symbol.remove()
        filter_design.zeros_positions = []
        self.zeros_symbols = {}
        filter_design.zeros_positions = [] 
        self.draw()

    def clearPoles(self):
        for symbol in self.poles_symbols:
            symbol.remove()
        filter_design.poles_positions = []
        self.poles_symbols = {}
        filter_design.poles_positions = []  
        self.draw()

    def clearAllPoints(self):
        filter_design.zeros_positions = []
        self.zeros_symbols = {}
        filter_design.poles_positions = []
        self.poles_symbols = {}
        self.ax.clear() 
        self.plot_zplane()
        self.connect_events()
        self.draw()

    def plot_zplane(self):
        # Plot the unit circle
        t = np.linspace(0, 2 * np.pi, 100)
        self.ax.plot(np.cos(t), np.sin(t), 'k--', label='Unit Circle')

        # Set plot limits and labels
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)
        self.ax.grid(color='gray', linestyle='--', linewidth=0.5)
        self.draw()

    def plotZero(self, x, y):
        symbol = self.ax.scatter(x, y, marker='o', color='r', label='Zeros')
        self.draw()
        self.zeros_symbols[symbol] = (x, y)
        filter_design.zeros_positions.append((x, y, symbol))
        filter_design.plot_mag_phase()
        filter_design.restore_generation()
        return symbol

    def plotPole(self, x, y):
        symbol = self.ax.scatter(x, y, marker='x', color='b', label='Poles')
        self.draw()
        filter_design.poles_positions.append((x, y, symbol))
        filter_design.plot_mag_phase()
        self.poles_symbols[symbol] = (x, y)
        filter_design.restore_generation()
        return symbol

    def setMode(self, mode):
        self.mode = mode
        self.selected_symbol = None  # Reset selected symbol when mode changes

    def connect_events(self):
        # Connect pick and motion notify events
        self.mpl_connect('pick_event', self.on_pick)
        self.mpl_connect('motion_notify_event', self.on_drag)
            
if __name__ == "__main__":
    app = QApplication([])
    filter_design = FilterDesign()
    filter_design.show()
    app.exec_()
