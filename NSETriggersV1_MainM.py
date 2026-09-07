import sys
import os
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import (QMainWindow, QApplication, QDialog, QFileDialog,
                             QMessageBox, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                             QWidget, QLineEdit, QPushButton, QCheckBox, QLabel)
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import pandas as pd
import numpy as np
import time
import shutil

from chart import CandlestickChartWindow
from get_stock_data import (get_signals, daily_memory, monthly_memory,
                            daily_cache_dir, monthly_cache_dir)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default symbol-list files, resolved relative to this script (cross-platform).
DEFAULT_SYMBOL_FILES = {
    "radioButton_1": os.path.join(BASE_DIR, "nifty50.txt"),
    "radioButton_2": os.path.join(BASE_DIR, "niftyn50.txt"),
    "radioButton_3": os.path.join(BASE_DIR, "nifty100.txt"),
    "radioButton_4": os.path.join(BASE_DIR, "nifty200.txt"),
    "radioButton_5": os.path.join(BASE_DIR, "nifty500.txt"),
    "radioButton_6": os.path.join(BASE_DIR, "TotalList.txt"),
}

TABLES = ["buyd_tab", "selld_tab", "buym_tab", "sellm_tab", "buyc_tab", "sellc_tab"]

DARK_STYLE = """
    QMainWindow, QDialog { background:#1e1e1e; color:#e0e0e0; }
    QWidget { background:#1e1e1e; color:#e0e0e0; }
    QToolBar { background:#252525; border:none; spacing:6px; }
    QLineEdit { background:#2d2d2d; color:#e0e0e0; border:1px solid #444; padding:4px; }
    QPushButton { background:#0d6efd; color:#fff; border:none; padding:5px 12px; border-radius:3px; }
    QPushButton:hover { background:#2b8ff6; }
    QTreeWidget { background:#1e1e1e; color:#e0e0e0; border:1px solid #333; }
    QTreeWidget::item { padding:3px; }
    QHeaderView::section { background:#252525; color:#e0e0e0; border:1px solid #333; padding:4px; }
    QComboBox, QTabBar::tab { background:#2d2d2d; color:#e0e0e0; border:1px solid #333; }
    QTabBar::tab { padding:6px 14px; }
    QProgressBar { border:1px solid #333; border-radius:3px; text-align:center; }
    QProgressBar::chunk { background:#0d6efd; }
    QStatusBar { background:#1e1e1e; }
"""

LIGHT_STYLE = ""


class ProcessWorker(QObject):
    progress = pyqtSignal(dict)
    update_progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, symbols, filters):
        super().__init__()
        self.symbols = symbols
        self.filters = filters
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        total = len(self.symbols)
        for idx, symbol in enumerate(self.symbols):
            if self._stop_requested:
                self.error.emit("Processing stopped by user.")
                break

            symbol = symbol.strip()
            if not symbol:
                continue
            print(f"Processing: {symbol}")
            try:
                signals = get_signals(symbol, self.filters)
                for signal in signals:
                    self.progress.emit(signal)
            except Exception as e:
                self.error.emit(f"Error processing symbol {symbol}: {e}")

            progress_percent = int(((idx + 1) / total) * 100) if total else 100
            self.update_progress.emit(progress_percent)

        self.finished.emit()


class SettingsDialog(QDialog):
    def __init__(self, parent=None, cache_dir=""):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(400, 220)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Cache location:"))
        info = QLabel(cache_dir)
        info.setWordWrap(True)
        info.setStyleSheet("color:#9aa0a6;")
        layout.addWidget(info)

        self.clear_btn = QPushButton("Clear Data Cache")
        self.clear_btn.clicked.connect(self._clear_cache)
        layout.addWidget(self.clear_btn)

        self.theme_cb = QCheckBox("Use dark theme")
        self.theme_cb.setChecked(True)
        layout.addWidget(self.theme_cb)

        ok = QPushButton("OK")
        ok.clicked.connect(self.accept)
        layout.addWidget(ok)

    def _clear_cache(self):
        try:
            daily_memory.clear()
            monthly_memory.clear()
            if os.path.exists(daily_cache_dir):
                shutil.rmtree(daily_cache_dir, ignore_errors=True)
            if os.path.exists(monthly_cache_dir):
                shutil.rmtree(monthly_cache_dir, ignore_errors=True)
            self.clear_btn.setText("Cache cleared ✓")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to clear cache: {e}")

    def theme_enabled(self):
        return self.theme_cb.isChecked()


class NSETriggersV1M(QMainWindow):
    def __init__(self):
        super(NSETriggersV1M, self).__init__()
        loadUi('NSETriggersV1M.ui', self)

        self.setStyleSheet(DARK_STYLE)

        self.yesterday_close_buy_signal = {}
        self.yesterday_close_sell_signal = {}
        self.buyd_tab.itemDoubleClicked.connect(self.open_candlestick_chart)
        self.selld_tab.itemDoubleClicked.connect(self.open_candlestick_chart)

        # Column headers (combined tabs carry an extra Status column).
        self.combined_tables = {self.buyc_tab, self.sellc_tab}
        for name in TABLES:
            table = getattr(self, name)
            if table in self.combined_tables:
                table.setHeaderLabels(["Symbol", "Timestamp", "Close", "Zones", "Status"])
            else:
                table.setHeaderLabels(["Symbol", "Timestamp", "Close", "Zones"])

        # Toolbar: search + settings.
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search symbols…")
        self.search_edit.textChanged.connect(self.filter_tables)
        toolbar.addWidget(self.search_edit)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.show_settings)
        toolbar.addWidget(self.settings_button)

        # Connect UI controls.
        self.symbol_file_button.clicked.connect(self.select_file)
        self.process_button.clicked.connect(self.start_processing)
        self.export_button.clicked.connect(self.export_to_excel)
        self.open_folder_button.clicked.connect(self.open_script_folder)
        self.stop_button.clicked.connect(self.stop_processing)
        self.clear_button.clicked.connect(self.clear_tables)
        self.close_button.clicked.connect(self.close)
        for i in range(1, 7):
            getattr(self, f"radioButton_{i}").clicked.connect(
                lambda checked, n=i: self.select_default_file(n))
        self.clear_memory_button.clicked.connect(self.clear_memory)

        self.symbols_list = []
        self.worker_thread = None
        self.worker = None

    # --- Chart ------------------------------------------------------------
    def open_candlestick_chart(self, item):
        symbol = item.text(0)
        df = self.fetch_15m_data(symbol)
        if df is not None:
            self.candlestick_window = CandlestickChartWindow(symbol, df)
            self.candlestick_window.show()

    def fetch_15m_data(self, symbol):
        import yfinance as yf
        try:
            df = yf.download(symbol, interval='15m', period='1d',
                             auto_adjust=False, progress=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna()
            df = df.astype({'Open': 'float64', 'High': 'float64', 'Low': 'float64',
                            'Close': 'float64', 'Volume': 'float64'})
            return df
        except Exception:
            return None

    def open_script_folder(self):
        folder_path = os.path.dirname(os.path.abspath(__file__))
        try:
            os.startfile(folder_path)
        except AttributeError:
            import subprocess
            subprocess.Popen(['xdg-open', folder_path])

    # --- Default symbol files --------------------------------------------
    def select_default_file(self, n):
        path = DEFAULT_SYMBOL_FILES.get(f"radioButton_{n}")
        if path:
            self.symbol_file_entry.setText(path)
            self.load_symbols_from_file(path)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            self.symbol_file_entry.setText(file_path)
            self.load_symbols_from_file(file_path)

    def load_symbols_from_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                self.symbols_list = [s.strip() for s in file.read().split(',') if s.strip()]
            print("Loaded symbols:", len(self.symbols_list))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to read file: {e}")

    # --- Processing -------------------------------------------------------
    def start_processing(self):
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.information(self, "Busy", "Processing is already running. Stop it first.")
            return

        self.clear_tables()
        file_path = self.symbol_file_entry.text()
        if not file_path:
            QMessageBox.information(self, "No file", "Please select a file first.")
            return

        self.load_symbols_from_file(file_path)
        if not self.symbols_list:
            QMessageBox.information(self, "No symbols", "The selected file has no symbols.")
            return

        self.start_time = time.time()

        date_arg_map = {"Today": "CD", "Today-1": "CD-1", "Today-2": "CD-2",
                        "Today-3": "CD-3", "Today-4": "CD-4"}
        month_arg_map = {"This Month": "CM", "Last Month": "LM", "Last Last Month": "LLM"}
        date_filter = self.date_filter_combobox.currentText()
        month_filter = self.month_filter_combobox.currentText()

        filters = []
        if self.process_symbolsm_checkbox.isChecked() and month_filter in month_arg_map:
            filters.append(month_arg_map[month_filter])
        if self.process_symbolsd_checkbox.isChecked() and date_filter in date_arg_map:
            filters.append(date_arg_map[date_filter])
        if not filters:
            QMessageBox.information(self, "No filters", "Select at least one process option.")
            return

        self.worker_thread = QThread()
        self.worker = ProcessWorker(self.symbols_list, filters)
        self.worker.moveToThread(self.worker_thread)

        self.worker.progress.connect(self.handle_progress)
        self.worker.error.connect(self.handle_error)
        self.worker.finished.connect(self.handle_finished)
        self.worker.update_progress.connect(self.update_progress_bar)
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        self.worker_thread.start()

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            print("Stop requested by user.")

    def handle_progress(self, signal_data):
        if "symbol" not in signal_data:
            return

        symbol = signal_data["symbol"]
        stype = signal_data.get("signal_type", "")

        if "1d Buy Signal" in stype:
            self.add_to_table(self.buym_tab, symbol, signal_data["timestamp"], str(signal_data["close_price"]), signal_data["zones"])
        elif "1d Sell Signal" in stype:
            self.add_to_table(self.sellm_tab, symbol, signal_data["timestamp"], str(signal_data["close_price"]), signal_data["zones"])
        elif "15m Buy Signal" in stype:
            self.add_to_table(self.buyd_tab, symbol, signal_data["timestamp"], str(signal_data["close_price"]), signal_data["zones"])
        elif "15m Sell Signal" in stype:
            self.add_to_table(self.selld_tab, symbol, signal_data["timestamp"], str(signal_data["close_price"]), signal_data["zones"])
        elif "Yesterday Close > AD" in stype:
            self.yesterday_close_buy_signal[symbol] = True
        elif "Yesterday Close < BD" in stype:
            self.yesterday_close_sell_signal[symbol] = True

    def handle_error(self, error_message):
        QMessageBox.warning(self, "Processing Error", error_message)

    def handle_finished(self):
        self.update_common_tabs()
        self.worker_thread.quit()
        self.worker_thread.wait()

    def add_to_table(self, table, symbol, timestamp, close_price, zones, status=""):
        if not isinstance(table, QTreeWidget):
            return
        if isinstance(zones, dict):
            zones_text = f"AD: {zones['AD']}, BD: {zones['BD']}, CD: {zones['CD']}, DD: {zones['DD']}"
        else:
            zones_text = str(zones)
        columns = [symbol, timestamp, close_price, zones_text]
        if table in self.combined_tables:
            columns.append(status)
        table.addTopLevelItem(QTreeWidgetItem(columns))

    def update_common_tabs(self):
        buym_symbols = {self.buym_tab.topLevelItem(i).text(0).strip() for i in range(self.buym_tab.topLevelItemCount())}
        buyd_symbols = {self.buyd_tab.topLevelItem(i).text(0).strip() for i in range(self.buyd_tab.topLevelItemCount())}

        for i in range(self.buyd_tab.topLevelItemCount()):
            item = self.buyd_tab.topLevelItem(i)
            symbol = item.text(0).strip()
            buy_signal_value = self.yesterday_close_buy_signal.get(symbol, False)
            sell_signal_value = self.yesterday_close_sell_signal.get(symbol, False)
            if symbol in buym_symbols and symbol in buyd_symbols and buy_signal_value and not sell_signal_value:
                self.add_to_table(self.buyc_tab, symbol, item.text(1), item.text(2),
                                  self.parse_zones(item.text(3)), status="BUY")

        sellm_symbols = {self.sellm_tab.topLevelItem(i).text(0).strip() for i in range(self.sellm_tab.topLevelItemCount())}
        selld_symbols = {self.selld_tab.topLevelItem(i).text(0).strip() for i in range(self.selld_tab.topLevelItemCount())}

        for i in range(self.selld_tab.topLevelItemCount()):
            item = self.selld_tab.topLevelItem(i)
            symbol = item.text(0).strip()
            sell_signal_value = self.yesterday_close_sell_signal.get(symbol, False)
            if symbol in sellm_symbols and symbol in selld_symbols and sell_signal_value:
                self.add_to_table(self.sellc_tab, symbol, item.text(1), item.text(2),
                                  self.parse_zones(item.text(3)), status="SELL")

    def clear_memory(self):
        try:
            daily_memory.clear()
            monthly_memory.clear()
            if os.path.exists(daily_cache_dir):
                shutil.rmtree(daily_cache_dir, ignore_errors=True)
            if os.path.exists(monthly_cache_dir):
                shutil.rmtree(monthly_cache_dir, ignore_errors=True)
            QMessageBox.information(self, "Cache Cleared", "Memory cache cleared successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to clear cache: {e}")

    def parse_zones(self, zones_text):
        try:
            zones = {}
            for part in zones_text.split(", "):
                key, value = part.split(": ")
                zones[key.strip()] = float(value.strip())
            return zones
        except Exception:
            return None

    def clear_tables(self):
        for name in TABLES:
            getattr(self, name).clear()
        self.progressBar.setValue(0)
        self.progressBar.setFormat("0%")

    # --- Search / settings ------------------------------------------------
    def filter_tables(self, text):
        text = text.strip().lower()
        for name in TABLES:
            table = getattr(self, name)
            for i in range(table.topLevelItemCount()):
                item = table.topLevelItem(i)
                match = (not text) or (item.text(0).lower().find(text) >= 0)
                item.setHidden(not match)

    def show_settings(self):
        dialog = SettingsDialog(self, cache_dir=daily_cache_dir)
        if dialog.exec_():
            self.setStyleSheet(DARK_STYLE if dialog.theme_enabled() else LIGHT_STYLE)

    def export_to_excel(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Excel Files (*.xlsx);;All Files (*)")
        if not file_path:
            return
        try:
            with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
                for tab_name, table in {
                    "Buy_1d": self.buyd_tab,
                    "Sell_1d": self.selld_tab,
                    "Buy_Monthly": self.buym_tab,
                    "Sell_Monthly": self.sellm_tab,
                    "Buy_Combined": self.buyc_tab,
                    "Sell_Combined": self.sellc_tab
                }.items():
                    data = [[table.topLevelItem(r).text(i)
                             for i in range(table.columnCount())]
                            for r in range(table.topLevelItemCount())]
                    if not data:
                        continue
                    headers = [table.headerItem().text(i) if table.headerItem() else str(i)
                               for i in range(table.columnCount())]
                    pd.DataFrame(data, columns=headers).to_excel(writer, sheet_name=tab_name, index=False)
            QMessageBox.information(self, "Export Successful", f"Data exported to {file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export: {e}")

    def update_progress_bar(self, value):
        self.progressBar.setValue(value)
        elapsed_time = time.time() - self.start_time
        self.progressBar.setFormat(f"{value}% - Elapsed Time: {elapsed_time:.1f}s")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("fusion")
    window = NSETriggersV1M()
    window.show()
    sys.exit(app.exec_())
