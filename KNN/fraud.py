import numpy as np
import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QApplication, QMessageBox, QFileDialog, QStackedWidget, QGridLayout

# Improved KNN Implementation with Debugging Statements
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0], k_indices, k_nearest_labels

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Fetch data from MySQL
def fetch_data_from_db():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='frauddetection'
    )
    query = "SELECT * FROM sqlpart1"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Load and preprocess data
def load_and_preprocess_data_from_db():
    df = fetch_data_from_db()
    X = df.drop(['class'], axis=1).values
    y = df['class'].values
    return X, y

def load_transaction_from_csv(filepath):
    try:
        df = pd.read_csv(filepath, delimiter=';', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, delimiter=';', encoding='latin1')
    
    if df.empty:
        raise ValueError("The CSV file is empty. Please provide a valid file with data.")
    df = df.apply(pd.to_numeric, errors='coerce')
    transaction = df.values
    return transaction

def predict_transaction(knn, scaler, transaction_data):
    transaction = np.array(transaction_data).reshape(1, -1)
    transaction = scaler.transform(transaction)
    prediction, k_indices, k_nearest_labels = knn._predict(transaction[0])
    result = 'Non-Fraudulent' if prediction == 0 else 'Fraudulent'
    return result, k_indices, k_nearest_labels


class MainPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setStyleSheet("background-color: lightgreen;")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Load and display an image
        self.img_label = QLabel(self)
        self.img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.img_label)

        # Create a title label and set it as a child of the image label
        self.title = QLabel("Credit Card Fraud Detection\nWelcome to credit card fraud detection system", self.img_label)
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 35px; color: white; background-color: rgba(0, 0, 0, 0.5);")
        self.title.setFont(QFont("Helvetica", 24))
        

        # Start button
        self.start_button = QPushButton("Start now", self.img_label)
        self.start_button.setStyleSheet("font-size: 35px; background-color: white; color: black;")
        self.start_button.clicked.connect(self.main_window.show_knn_selection_page)
        
        self.update_image("money2.jpg")

        self.setLayout(layout)

    def update_image(self, img_path):
        screen_geometry = QApplication.desktop().screenGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        pixmap = QPixmap(img_path)
        pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatioByExpanding)
        self.img_label.setPixmap(pixmap)
        self.img_label.setFixedSize(pixmap.size())
        # self.title.setFixedSize(pixmap.size().width(), 100)
        # self.start_button.setFixedSize(200, 50)
        # self.start_button.move(int((self.img_label.width() - 200) / 2), int((self.img_label.height() - 50) / 2 + 100))
        self.update_positions()
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image("money2.jpg")

    def update_positions(self):
        self.title.setGeometry(0, 50, self.img_label.width(), 100)
        self.start_button.setGeometry(
            int((self.img_label.width() - 200) / 2),  # Center horizontally
            int((self.img_label.height() - 50) / 2 + 100),  # Position below the title
            200,  # Button width
            50  # Button height
        )

        
class KNNSelectionPage(QWidget):  
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.setStyleSheet("background-color: white;")
        
        
        
        outer_layout = QVBoxLayout()
        outer_layout.setAlignment(Qt.AlignCenter) 
        
        container = QWidget(self)
        container.setFixedWidth(600)
        container.setFixedHeight(400)
        container_layout = QVBoxLayout(container)
        container_layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel("Enter the value of k for KNN:", self)
        self.label.setFont(QFont("Helvetica", 30))
        container_layout.addWidget(self.label)

        self.k_entry = QLineEdit(self)
        self.k_entry.setFont(QFont("Helvetica", 14))
        container_layout.addWidget(self.k_entry)
        self.submit_button = QPushButton("Submit", self)
        self.submit_button.setFont(QFont("Helvetica", 30))
        self.submit_button.setStyleSheet("""
            QPushButton {
                border-radius: 10px;
                background-color: green;
                color: black;
            }
            QPushButton:hover {
                background-color: green;
                color: white;
            }
        """)

        self.submit_button.clicked.connect(self.on_submit)
        container_layout.addWidget(self.submit_button)

        outer_layout.addWidget(container)
        self.setLayout(outer_layout)

    def on_submit(self):
        k = int(self.k_entry.text())
        self.main_window.show_choice_page(k)

class KNNSelectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KNN Configuration")
        screen_rect = app.desktop().screenGeometry()
        self.setGeometry(0, 0, screen_rect.width(), screen_rect.height())
        self.setStyleSheet("background-color: lightgreen;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_page = MainPage(self)
        self.knn_page = KNNSelectionPage(self) 
        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(self.main_page)
        self.stacked_widget.addWidget(self.knn_page)
        
        layout = QVBoxLayout()
        layout.addWidget(self.stacked_widget)
        self.central_widget.setLayout(layout)
        
        self.stacked_widget.setCurrentWidget(self.main_page)  

    def show_knn_selection_page(self):
        self.stacked_widget.setCurrentWidget(self.knn_page)

    def show_choice_page(self, k):
        self.choice_window = ChoiceWindow(k)
        self.choice_window.show()
        self.close()



class ChoiceWindow(QMainWindow):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.setWindowTitle("Input Method Choice")
        
        screen_rect = app.desktop().screenGeometry()
        self.setGeometry(0, 0, screen_rect.width(), screen_rect.height())
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet("background-color: lightgreen;")
        
        outer_layout = QVBoxLayout(self.central_widget)
        outer_layout.setAlignment(Qt.AlignCenter) 
        
        container = QWidget(self)
        container.setFixedWidth(600)
        container.setFixedHeight(400)
        container_layout = QVBoxLayout(container)
        container_layout.setAlignment(Qt.AlignCenter)
        container.setStyleSheet("background-color: white; border: 1px solid #000; border-radius: 10px;")
        
        self.label = QLabel("Choose input method:", self)
        self.label.setFont(QFont("Helvetica", 16))
        self.label.setStyleSheet("font-size: 30px;")
        container_layout.addWidget(self.label)
        
        self.manual_button = QPushButton("Manual Entry", self)
        self.manual_button.setFont(QFont("Helvetica", 14))
        self.manual_button.clicked.connect(self.on_manual_entry)
        container_layout.addWidget(self.manual_button)
        
        self.csv_button = QPushButton("Upload CSV", self)
        self.csv_button.setFont(QFont("Helvetica", 14))
        self.csv_button.clicked.connect(self.on_csv_upload)
        container_layout.addWidget(self.csv_button)
        
        self.back_button = QPushButton("Back", self)
        self.back_button.setFont(QFont("Helvetica", 14))
        self.back_button.clicked.connect(self.go_back)
        container_layout.addWidget(self.back_button)
        
        outer_layout.addWidget(container)
        self.setLayout(outer_layout)
        
        # Style buttons and labels
        self.manual_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px;
                font-size: 30px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.csv_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px;
                font-size: 30px;
                
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 10px;
                padding: 10px;
                font-size: 30px;
                
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)

    def on_manual_entry(self):
        self.close()
        self.manual_entry_window = TransactionInputWindow(self.k)
        self.manual_entry_window.show()

    def on_csv_upload(self):
        self.close()
        self.csv_upload_window = CSVUploadWindow(self.k)
        self.csv_upload_window.show()

    def go_back(self):
        self.close()
        self.knn_selection_window = KNNSelectionWindow()
        self.knn_selection_window.show()


class TransactionInputWindow(QMainWindow):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.setWindowTitle("Transaction Input")
        
        screen_rect = app.desktop().screenGeometry()
        self.setGeometry(0, 0, screen_rect.width(), screen_rect.height())
        
        
        self.setStyleSheet("background-color: lightgreen;")
        
        X, y = load_and_preprocess_data_from_db()
        global scaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        global knn
        knn = KNN(k=self.k)
        knn.fit(X, y)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        outer_layout = QVBoxLayout(self.central_widget)
        outer_layout.setAlignment(Qt.AlignCenter)

        container = QWidget(self)
        container.setFixedWidth(1600)
        container.setStyleSheet("background-color: white; border: 1px solid #000; border-radius: 10px; padding: 20px;")
        
        layout = QVBoxLayout(container)
        
        self.grid_layout = QGridLayout()
        
        self.entries = []
        for i in range(28):
            entry = QLineEdit(self)
            entry.setFont(QFont("Helvetica", 12))
            row = i % 10
            col = i // 10
            self.grid_layout.addWidget(QLabel(f"Field {i + 1}:", self), row, col * 2)
            self.grid_layout.addWidget(entry, row, col * 2 + 1)
            self.entries.append(entry)
        
        layout.addLayout(self.grid_layout)
        
        self.submit_button = QPushButton("Submit", self)
        self.submit_button.setFont(QFont("Helvetica", 14))
        self.submit_button.clicked.connect(self.on_submit)
        layout.addWidget(self.submit_button)
        
        self.back_button = QPushButton("Back", self)
        self.back_button.setFont(QFont("Helvetica", 14))
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)
        
        outer_layout.addWidget(container)
        
        # Style input fields and buttons
        for entry in self.entries:
            entry.setStyleSheet("border: 5px solid purple; padding: 5px; border-radius: 5px;")

        self.submit_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
    def on_submit(self):
        transaction_data = [float(entry.text()) for entry in self.entries]
        result, k_indices, k_nearest_labels = predict_transaction(knn, scaler, transaction_data)
        self.result_label.setText(f"The transaction is: {result}")
        self.result_label.setStyleSheet("color: purple;")
        QMessageBox.information(self, "Prediction Result", f"The transaction is: {result}")

        k_nearest_text = f"k-nearest neighbors indices: {k_indices}\nk-nearest labels: {k_nearest_labels}"
        self.k_nearest_label.setText(k_nearest_text)
        
    def go_back(self):
        self.close()
        self.choice_window = ChoiceWindow(self.k)
        self.choice_window.show()


class CSVUploadWindow(QMainWindow):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.setWindowTitle("Upload CSV")
        
        screen_rect = app.desktop().screenGeometry()
        self.setGeometry(0, 0, screen_rect.width(), screen_rect.height())
               
        self.setStyleSheet("background-color: lightgreen;")
        
        X, y = load_and_preprocess_data_from_db()
        global scaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        global knn
        knn = KNN(k=self.k)
        knn.fit(X, y)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        outer_layout = QVBoxLayout()
        outer_layout.setAlignment(Qt.AlignCenter)
        
        self.upload_button = QPushButton("Upload CSV", self)
        self.upload_button.setFont(QFont("Helvetica", 14))
        self.upload_button.clicked.connect(self.on_submit)
        outer_layout.addWidget(self.upload_button)
        
        self.back_button = QPushButton("Back", self)
        self.back_button.setFont(QFont("Helvetica", 14))
        self.back_button.clicked.connect(self.go_back)
        outer_layout.addWidget(self.back_button)
        
         # Container for the result labels
        self.result_container = QWidget(self)
        self.result_container.setFixedWidth(1800)
        self.result_container.setStyleSheet("background-color: white; border: 1px solid #000; border-radius: 10px; padding: 20px;")
        self.result_container.setVisible(False)  # Initially hide the container
        
        result_layout = QVBoxLayout(self.result_container)
        
        self.result_label = QLabel("", self)
        self.result_label.setFont(QFont("Helvetica", 14))
        self.result_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.result_label)
        
        self.k_nearest_label = QLabel("", self)
        self.k_nearest_label.setFont(QFont("Helvetica", 12))
        self.k_nearest_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.k_nearest_label)
        
        outer_layout.addWidget(self.result_container, alignment=Qt.AlignCenter)
        
        self.central_widget.setLayout(outer_layout)
        

        # Style buttons
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.back_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
    def on_submit(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if filepath:
            transaction_data = load_transaction_from_csv(filepath)[0]
            result, k_indices, k_nearest_labels = predict_transaction(knn, scaler, transaction_data)
            self.result_label.setText(f"The transaction is: {result}")
            self.result_label.setStyleSheet("color: purple; font-size:50px")
            QMessageBox.information(self, "Prediction Result", f"The transaction is: {result}")

            k_nearest_text = f"k-nearest neighbors indices: {k_indices}\nk-nearest labels: {k_nearest_labels}"
            self.k_nearest_label.setText(k_nearest_text)
            self.k_nearest_label.setStyleSheet("font-size:35px")
            
            
            self.result_container.setVisible(True)
        
    def go_back(self):
        self.close()
        self.choice_window = ChoiceWindow(self.k)
        self.choice_window.show()

if __name__ == "__main__":
  
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QLabel {
            font-size: 14px;
            color: #333;
        }
        QLineEdit {
            border: 1px solid #ccc;
            padding: 5px;
            border-radius: 5px;
        }
        QPushButton {
            font-size: 14px;
            padding: 10px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
    """)
    main_window = KNNSelectionWindow()
    main_window.show()
    sys.exit(app.exec_())
    
