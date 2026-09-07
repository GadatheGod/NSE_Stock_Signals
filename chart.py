import yfinance as yf
import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QApplication
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT


class CandlestickChartWindow(QMainWindow):
    """Standalone 15-minute candlestick chart for a single symbol.

    Accepts an already-fetched DataFrame; otherwise fetches data itself and
    refreshes automatically every 15 minutes.
    """

    def __init__(self, symbol, df=None):
        super().__init__()
        self.setWindowTitle(f"{symbol} — 15m Candlestick")
        self.resize(1000, 650)

        self.symbol = symbol
        self.df = df
        self.central = QWidget()
        self.setCentralWidget(self.central)
        layout = QVBoxLayout(self.central)

        self.status = QLabel(f"Loading {symbol}…")
        layout.addWidget(self.status)

        self.figure = plt.figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)

        self.refresh_button = QPushButton("Refresh Chart")
        self.refresh_button.clicked.connect(self.plot_chart)
        layout.addWidget(self.refresh_button)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.plot_chart)
        self.timer.start(900000)  # refresh every 15 minutes

        if df is not None and not df.empty:
            self.plot_chart()
        else:
            self.fetch_and_plot()

    @staticmethod
    def _clean(df):
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        df = df.astype({'Open': 'float64', 'High': 'float64', 'Low': 'float64',
                        'Close': 'float64', 'Volume': 'float64'})
        return df

    def fetch_stock_data(self):
        df = yf.download(self.symbol, interval='15m', period='1d',
                         auto_adjust=False, progress=False)
        return self._clean(df)

    def fetch_and_plot(self):
        df = self.fetch_stock_data()
        if df is None or df.empty:
            self.status.setText(f"No data available for {self.symbol}")
            return
        self.df = df
        self.plot_chart()

    def plot_chart(self):
        df = self.df
        if df is None or df.empty:
            df = self.fetch_stock_data()
            if df is None or df.empty:
                self.status.setText(f"No data available for {self.symbol}")
                return
            self.df = df
        try:
            self.figure.clear()
            gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 1])
            ax = self.figure.add_subplot(gs[0])
            vol_ax = self.figure.add_subplot(gs[1])
            mpf.plot(df, type='candle', style='charles', ax=ax, volume=vol_ax,
                     ylabel='Price', tight_layout=True)
            self.canvas.draw()
            self.status.setText(f"{self.symbol} — {len(df)} candles")
        except Exception as e:
            self.status.setText(f"Plot error: {e}")
