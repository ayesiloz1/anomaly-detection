""" 
Currently only have the confusion matrix, and I'm planning to add more visualizations in the future.

"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import os
import pandas as pd


def plot_confusion_matrix(self):
    # Define path to the CSV file
    csv_path = os.path.join(os.getcwd(), "anomaly.csv")
        
    # Read CSV data
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found. Make sure to save the results first.")
        return

    # Load data from CSV
    data = pd.read_csv(csv_path)
        
    # Set a new threshold
    threshold = 3
    data['PredictedLabel'] = data['AnomaliesDetected'].apply(lambda x: 1 if x > threshold else 0)

    # Extract true labels and predicted labels
    y_true = data['TrueLabel']
    y_pred = data['PredictedLabel']
        
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
        
    # Plot confusion matrix
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["Good", "Reject"])
    disp.plot(ax=ax, cmap="Blues")
        
    # Set the title to include the threshold
    ax.set_title(f"Confusion Matrix (Threshold = {threshold})")

    #Optionally, add a text box with the threshold value inside the plot area
    text_box_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8)
    ax.text(0.95, 0.05, f"Threshold = {threshold}", transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=text_box_props)
        
    plt.show()
