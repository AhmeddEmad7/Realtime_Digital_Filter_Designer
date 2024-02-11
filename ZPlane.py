from PyQt5.QtWidgets import *
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from FilterDesign import filter_design

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
            self.restore_generation()

    def on_release(self, event):
        if self.selected_symbol and event.name == 'button_release_event':
            self.mpl_disconnect(self.drag_cid)
            self.drag_cid = self.mpl_connect('motion_notify_event', self.on_drag)
            self.selected_symbol = None
            self.restore_generation()

    def on_pick(self, event):
        if event.artist.get_label() == 'Zeros' and self.zerosRadioButton.isChecked():
            self.selected_symbol = event.artist
        elif event.artist.get_label() == 'Poles' and self.polesRadioButton.isChecked():
            self.selected_symbol = event.artist
        self.restore_generation()

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

        self.restore_generation()
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
                if self.mode == "zeros" and self.conjugate_mode:
                    self.zeros_count += 1
                    symbol = self.plotZero(x, y)
                    x_conjugate, y_conjugate = self.calculate_conjugate(x, y)
                    conjugate_symbol = self.plot_conjugate(x_conjugate, y_conjugate, symbol, 1)
                    filter_design.zeros_positions.append((x_conjugate, y_conjugate, conjugate_symbol))
                    self.draw()
                elif self.mode == "poles" and self.conjugate_mode:
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
        self.restore_generation()


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
        self.restore_generation()

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

        self.restore_generation()
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
        self.restore_generation()
        return symbol

    def plotPole(self, x, y):
        symbol = self.ax.scatter(x, y, marker='x', color='b', label='Poles')
        self.draw()
        filter_design.poles_positions.append((x, y, symbol))
        filter_design.plot_mag_phase()
        self.poles_symbols[symbol] = (x, y)
        self.restore_generation()
        return symbol

    def setMode(self, mode):
        self.mode = mode
        self.selected_symbol = None  # Reset selected symbol when mode changes

    def connect_events(self):
        # Connect pick and motion notify events
        self.mpl_connect('pick_event', self.on_pick)
        self.mpl_connect('motion_notify_event', self.on_drag)