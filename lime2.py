import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import pandas as pd
from scipy import stats

# Function to load and preprocess a single image
def load_and_preprocess_image(image_path, img_size=(280, 274)):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img)
        return img_array / 255.0  # Normalize
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Function to extract statistical features from image array
def extract_features(img_array):
    # Calculate statistics for each channel
    features = {}
    
    # Overall image statistics (grayscale equivalent)
    gray_img = np.mean(img_array, axis=2)
    flat_img = gray_img.flatten()
    
    features['mean'] = np.mean(flat_img)
    features['std'] = np.std(flat_img)
    features['variance'] = np.var(flat_img)
    features['skewness'] = stats.skew(flat_img)
    features['kurtosis'] = stats.kurtosis(flat_img)
    features['argmax'] = np.argmax(flat_img)
    
    # Calculate per-channel statistics
    for i, channel_name in enumerate(['red', 'green', 'blue']):
        channel = img_array[:, :, i].flatten()
        features[f'{channel_name}_mean'] = np.mean(channel)
        features[f'{channel_name}_std'] = np.std(channel)
        features[f'{channel_name}_variance'] = np.var(channel)
        features[f'{channel_name}_skewness'] = stats.skew(channel)
        features[f'{channel_name}_kurtosis'] = stats.kurtosis(channel)
    
    return features

# Main processing function
def process_dataset(base_dir, model_path, num_samples=30):
    # Load the model
    loaded_model = load_model(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Model input shape: {loaded_model.input_shape}")
    print(f"Model output shape: {loaded_model.output_shape}")
    
    # Prepare lists to store data
    all_images = []
    all_labels = []
    all_features_list = []
    
    # Process seizure images
    seizure_dir = os.path.join(base_dir, 'test', 'seizure')
    for filename in os.listdir(seizure_dir)[:num_samples]:
        img_path = os.path.join(seizure_dir, filename)
        img_array = load_and_preprocess_image(img_path)
        if img_array is not None:
            all_images.append(img_array)
            all_labels.append(1)  # Seizure class
            all_features_list.append(extract_features(img_array))
    
    # Process non-seizure images
    non_seizure_dir = os.path.join(base_dir, 'test', 'non seizure')
    for filename in os.listdir(non_seizure_dir)[:num_samples]:
        img_path = os.path.join(non_seizure_dir, filename)
        img_array = load_and_preprocess_image(img_path)
        if img_array is not None:
            all_images.append(img_array)
            all_labels.append(0)  # Non-seizure class
            all_features_list.append(extract_features(img_array))
    
    # Convert to numpy arrays
    images = np.array(all_images)
    labels = np.array(all_labels)
    
    return images, labels, all_features_list, loaded_model

# Function to create a prediction function for LIME
def get_prediction_fn(model):
    def predict_fn(images):
        # Convert from RGB to model's expected format if needed
        processed_images = images.copy()
        
        # Get model predictions
        preds = model.predict(processed_images)
        
        # Return predictions
        return preds
    return predict_fn

# Function to create and visualize LIME explanations for images
def visualize_lime_image_explanations(images, labels, model, all_features_list, num_to_display=5):
    # Create a LIME image explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Get prediction function
    predict_fn = get_prediction_fn(model)
    
    # Create a directory for saving results
    results_dir = 'lime_explanations'
    os.makedirs(results_dir, exist_ok=True)
    
    # Create a dataframe to store feature statistics
    feature_df = pd.DataFrame(all_features_list)
    feature_df['label'] = labels
    
    # Save feature statistics
    feature_df.to_csv(os.path.join(results_dir, 'feature_statistics.csv'), index=False)
    
    # Calculate and print feature statistics by class
    print("\nFeature Statistics by Class:")
    for col in feature_df.columns:
        if col != 'label':
            seizure_mean = feature_df[feature_df['label'] == 1][col].mean()
            non_seizure_mean = feature_df[feature_df['label'] == 0][col].mean()
            print(f"{col}: Seizure mean = {seizure_mean:.4f}, Non-seizure mean = {non_seizure_mean:.4f}")
    
    # Get some random indices to explain
    indices_to_explain = np.random.choice(len(images), num_to_display, replace=False)
    
    # Store LIME explanations for feature importance visualization
    all_lime_weights = []
    lime_feature_names = []
    
    for i, idx in enumerate(indices_to_explain):
        # Get prediction for this image
        pred = model.predict(np.array([images[idx]]))[0]
        pred_label = 'Seizure' if np.argmax(pred) == 1 else 'Non-Seizure'
        true_label = 'Seizure' if labels[idx] == 1 else 'Non-Seizure'
        
        # Extract features for this image and display
        features = all_features_list[idx]
        
        # Get LIME explanation
        explanation = explainer.explain_instance(
            images[idx], 
            predict_fn, 
            top_labels=2,
            hide_color=0, 
            num_samples=1000
        )
        
        # Get the explanation for the predicted class
        pred_class = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            pred_class,
            positive_only=True, 
            num_features=10, 
            hide_rest=False
        )
        
        # Extract feature importance for LIME instance graph
        if i == 0:  # For the first image, get the feature names
            # Get the explanation as a list of tuples
            lime_exp = explanation.local_exp[pred_class]
            # Sort by absolute weight
            sorted_exp = sorted(lime_exp, key=lambda x: abs(x[1]), reverse=True)
            # Get top features (superpixel indices) and their weights
            top_features_idx = [x[0] for x in sorted_exp[:10]]
            top_weights = [x[1] for x in sorted_exp[:10]]
            # Create feature names (superpixel regions)
            lime_feature_names = [f"Region {j+1}" for j in range(len(top_features_idx))]
            all_lime_weights.append(top_weights)
        else:
            lime_exp = explanation.local_exp[pred_class]
            sorted_exp = sorted(lime_exp, key=lambda x: abs(x[1]), reverse=True)
            top_weights = [x[1] for x in sorted_exp[:10]]
            all_lime_weights.append(top_weights)
        
        # Create a figure for visualization
        plt.figure(figsize=(15, 10))
        
        # Show original image
        plt.subplot(2, 2, 1)
        plt.imshow(images[idx])
        plt.title(f"Original Image\nTrue: {true_label}, Pred: {pred_label}")
        plt.axis('off')
        
        # Show explanation
        plt.subplot(2, 2, 2)
        plt.imshow(mark_boundaries(temp, mask))
        plt.title("LIME Explanation (Important Regions)")
        plt.axis('off')
        
        # Show statistical features
        plt.subplot(2, 2, 3)
        key_features = ['mean', 'std', 'variance', 'skewness', 'kurtosis']
        feature_values = [features[k] for k in key_features]
        plt.bar(key_features, feature_values)
        plt.title("Statistical Features")
        plt.xticks(rotation=45)
        
        # Show LIME instance graph (feature importance)
        plt.subplot(2, 2, 4)
        lime_exp = explanation.local_exp[pred_class]
        sorted_exp = sorted(lime_exp, key=lambda x: abs(x[1]), reverse=True)
        top_weights = [x[1] for x in sorted_exp[:10]]
        regions = [f"Region {j+1}" for j in range(len(top_weights))]
        
        colors = ['green' if w > 0 else 'red' for w in top_weights]
        plt.barh(regions, top_weights, color=colors)
        plt.title("LIME Feature Importance")
        plt.xlabel("Weight (Green = Positive, Red = Negative)")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'lime_explanation_{i}.png'))
        plt.close()
    
    # Create aggregate LIME instance graph across all samples
    plt.figure(figsize=(12, 6))
    avg_weights = np.mean(all_lime_weights, axis=0)
    colors = ['green' if w > 0 else 'red' for w in avg_weights]
    
    plt.barh(lime_feature_names, avg_weights, color=colors)
    plt.title("Average LIME Feature Importance Across Samples")
    plt.xlabel("Weight (Green = Positive, Red = Negative)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'aggregate_lime_importance.png'))
    plt.close()
    
    # Create feature distribution plots
    plt.figure(figsize=(15, 10))
    key_features = ['mean', 'std', 'variance', 'skewness', 'kurtosis']
    
    for i, feature in enumerate(key_features):
        plt.subplot(2, 3, i+1)
        
        # Get data by class
        seizure_values = [f[feature] for f, l in zip(all_features_list, labels) if l == 1]
        non_seizure_values = [f[feature] for f, l in zip(all_features_list, labels) if l == 0]
        
        # Plot histograms
        plt.hist(seizure_values, alpha=0.5, bins=10, label='Seizure')
        plt.hist(non_seizure_values, alpha=0.5, bins=10, label='Non-Seizure')
        plt.title(f'{feature} Distribution')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_distributions.png'))
    plt.close()

# Main execution
if __name__ == "__main__":
    # Settings
    base_dir = 'D:/DSNproject/DSN2/EEG dataset + code/dataset'  # Update with your dataset path
    model_path = 'model.h5'  # Path to your saved model
    
    # Process the dataset
    print("Processing dataset...")
    images, labels, all_features_list, loaded_model = process_dataset(base_dir, model_path)
    print(f"Processed {len(images)} images")
    
    # Visualize LIME explanations
    print("Generating LIME explanations...")
    visualize_lime_image_explanations(images, labels, loaded_model, all_features_list)
    
    print("LIME explanations completed and saved to 'lime_explanations' directory")