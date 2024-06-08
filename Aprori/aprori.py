import sys
import pandas as pd
from itertools import combinations, chain
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTextEdit, QFileDialog, QMessageBox, QMainWindow, QStackedWidget)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

def load_data(application_path, credit_path):
    application_df = pd.read_csv(application_path)
    credit_df = pd.read_csv(credit_path)
    return application_df, credit_df

def preprocess_data(application_df, credit_df):
    merged_df = pd.merge(application_df, credit_df, on='ID')
    
    transactions = []
    grouped = merged_df.groupby('ID')
    for name, group in grouped:
        transaction = set()
        transaction.update(group['STATUS'].unique())
        transaction.update([f"{col}:{val}" for col, val in group.iloc[0].items() if col != 'ID'])
        transactions.append(transaction)
    
    return transactions

def create_initial_candidates(transactions):
    candidates = set()
    for transaction in transactions:
        for item in transaction:
            candidates.add(frozenset([item]))
    return candidates

def generate_candidates(frequent_itemsets, k):
    candidates = set()
    itemsets = list(frequent_itemsets)
    for i in range(len(itemsets)):
        for j in range(i + 1, len(itemsets)):
            candidate = itemsets[i].union(itemsets[j])
            if len(candidate) == k:
                candidates.add(candidate)
    return candidates

def prune_candidates(transactions, candidates, min_support):
    item_counts = {candidate: 0 for candidate in candidates}
    for transaction in transactions:
        for candidate in candidates:
            if candidate.issubset(transaction):
                item_counts[candidate] += 1
    
    num_transactions = len(transactions)
    frequent_items = set()
    item_support = {}
    
    for candidate, count in item_counts.items():
        support = count / num_transactions
        if support >= min_support:
            frequent_items.add(candidate)
            item_support[candidate] = support
    
    return frequent_items, item_support

def apriori(transactions, min_support):
    frequent_itemsets = []
    support_data = {}
    
    candidates = create_initial_candidates(transactions)
    k = 1
    
    while candidates:
        frequent_items, item_support = prune_candidates(transactions, candidates, min_support)
        if not frequent_items:
            break
        frequent_itemsets.append((k, frequent_items))
        support_data.update(item_support)
        k += 1
        candidates = generate_candidates(frequent_items, k)
    
    return frequent_itemsets, support_data

def generate_association_rules(frequent_itemsets, support_data, min_confidence):
    rules = []
    for level, itemsets in frequent_itemsets:
        for itemset in itemsets:
            if len(itemset) > 1:
                subsets = list(chain(*[combinations(itemset, i) for i in range(1, len(itemset))]))
                for subset in subsets:
                    subset = frozenset(subset)
                    remain = itemset - subset
                    if remain:
                        confidence = support_data[itemset] / support_data[subset]
                        if confidence >= min_confidence:
                            rules.append((subset, remain, confidence))
    return rules

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Apriori Algorithm Application')
        self.setGeometry(100, 100, 800, 600)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.main_widget = MainWidget(self)
        self.apriori_widget = AprioriApp(self)

        self.stacked_widget.addWidget(self.main_widget)
        self.stacked_widget.addWidget(self.apriori_widget)

        self.main_widget.start_button.clicked.connect(self.show_apriori)

        

    def show_apriori(self):
        self.stacked_widget.setCurrentWidget(self.apriori_widget)

class MainWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Main layout for centering
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Set background color
        self.setStyleSheet("background-color: purple;")

        # Invisible container
        container = QWidget()
        container.setFixedHeight(800)
        container.setFixedWidth(900)
        container_layout = QVBoxLayout()
        container_layout.setAlignment(Qt.AlignCenter)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel('Welcome to the Apriori Algorithm Application')
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 35px; color: white;")
        container_layout.addWidget(self.title_label)

        self.image_label = QLabel()
        pixmap = QPixmap('money8.jpg').scaled(800, 600, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)
        container_layout.addWidget(self.image_label)

        self.start_button = QPushButton('Start Apriori Analysis')
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #dda0dd;
            }
        """)
        container_layout.addWidget(self.start_button)

        container.setLayout(container_layout)

        # Add the container to the main layout
        main_layout.addWidget(container)

        self.setLayout(main_layout)

class AprioriApp(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.support_label = QLabel('Minimum Support:')
        layout.addWidget(self.support_label)

        self.support_input = QLineEdit()
        layout.addWidget(self.support_input)

        self.confidence_label = QLabel('Minimum Confidence:')
        layout.addWidget(self.confidence_label)

        self.confidence_input = QLineEdit()
        layout.addWidget(self.confidence_input)

        self.run_button = QPushButton('Run Apriori')
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #800080;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #dda0dd;
            }
        """)
        self.run_button.clicked.connect(self.run_apriori)
        layout.addWidget(self.run_button)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.setLayout(layout)

    def run_apriori(self):
        try:
            min_support = float(self.support_input.text())
            min_confidence = float(self.confidence_input.text())
            
            application_path, _ = QFileDialog.getOpenFileName(self, "Select Application Record CSV", "", "CSV Files (*.csv)")
            credit_path, _ = QFileDialog.getOpenFileName(self, "Select Credit Record CSV", "", "CSV Files (*.csv)")
            
            application_df, credit_df = load_data(application_path, credit_path)
            transactions = preprocess_data(application_df, credit_df)
            
            frequent_itemsets, support_data = apriori(transactions, min_support)
            rules = generate_association_rules(frequent_itemsets, support_data, min_confidence)
            
            result = "Frequent Itemsets:\n"
            for level, itemsets in frequent_itemsets:
                result += f"Level {level}:\n"
                for itemset in itemsets:
                    result += f"  {itemset}, support: {support_data[itemset]:.2f}\n"
            
            result += "\nAssociation Rules:\n"
            for rule in rules:
                result += f"{set(rule[0])} -> {set(rule[1])}, confidence: {rule[2]:.2f}\n"
            
            self.result_text.setPlainText(result)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
