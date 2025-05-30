import argparse
from load_data import load_dataset, CLASS_MAP
from preprocessing import preprocess_for_svm, preprocess_for_resnet
from visualization import show_samples_per_class, plot_class_distribution, plot_training_history, visualize_model_results
from baselineSVMmodel import train_pca_svm
from resNet50 import train_resnet50_model
from visualization import make_gradcam_heatmap, show_gradcam_overlay, generate_gradcam_grid
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


def main():
    # CLI: --model svm or --model resnet
    parser = argparse.ArgumentParser(description="Brain Tumor Classification")
    parser.add_argument("--model", type=str, choices=["svm", "resnet"], default="svm", help="Model to train")
    parser.add_argument("--gradcam", action="store_true", help="Enable Grad-CAM visualization (only works with ResNet)")
    args = parser.parse_args()

    print(">> Loading data...")
    X_train, y_train, X_test, y_test = load_dataset(subset_ratio=0.8)
    class_names = list(CLASS_MAP.keys())

    print(">> Visualizing raw samples and class distribution...")
    show_samples_per_class(X_train, y_train, class_names)
    plot_class_distribution(y_train, class_names, title="Raw Training Set Distribution")
    plot_class_distribution(y_test, class_names, title="Raw Test Set Distribution")

    if args.model == "svm":
        print(">> Preprocessing for SVM: CLAHE + normalization...")
        X_train, y_train, X_test, y_test = preprocess_for_svm(X_train, y_train, X_test, y_test)

        print(">> Visualizing preprocessed SVM data...")
        plot_class_distribution(y_train, class_names, title="SVM Training Set Distribution")
        plot_class_distribution(y_test, class_names, title="SVM Testing Set Distribution")

        print(">> Training PCA + SVM baseline model...")
        svm_model = train_pca_svm(X_train, y_train, X_test, y_test, n_components=100)

    elif args.model == "resnet":
        print(">> Preprocessing for ResNet50: CLAHE + keras.applications preprocess_input...")
        X_train, y_train, X_test, y_test = preprocess_for_resnet(X_train, y_train, X_test, y_test)

        print(">> Visualizing preprocessed ResNet data...")
        plot_class_distribution(y_train, class_names, title="ResNet Training Set Distribution")
        plot_class_distribution(y_test, class_names, title="ResNet Testing Set Distribution")

        model, history = train_resnet50_model(X_train, y_train, X_test, y_test, class_names)
        # model, history = train_resnet50_model(X_train, y_train, X_test, y_test, class_names, force_retrain=True)


        print(">> Visualizing model training performance...")
        plot_training_history(history)

        print(">> Visualizing model prediction...")
        visualize_model_results(model, X_test, y_test, class_names, history)
        
        # Check if y_test is one-hot (2D array)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            print(">> y_test detected as one-hot encoded. Converting to class labels...")
            y_test = np.argmax(y_test, axis=1)

        print(">> Evaluating model performance on Test Set...")

        # Predict on test set
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        print(">> Classification Report for ResNet50:")
        print(classification_report(y_test, y_pred, target_names=class_names))

        acc = accuracy_score(y_test, y_pred)
        print(f">> Overall Test Accuracy: {acc:.4f}")

        if args.gradcam:
            print(">> Generating Grad-CAM grid (one sample per class)...")
            generate_gradcam_grid(model, X_test, y_test, class_names)

    print(">> Done.")

if __name__ == "__main__":
    main()