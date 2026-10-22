import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
from scipy import stats
import shap

def load_and_preprocess_image(image_path, img_size=(280, 274)):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size)
        img_array = np.array(img)
        return img_array / 255.0  
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def extract_features(img_array):
    features = {}

    gray_img = np.mean(img_array, axis=2)
    flat_img = gray_img.flatten()
    features['mean'] = np.mean(flat_img)
    features['std'] = np.std(flat_img)
    features['variance'] = np.var(flat_img)
    features['skewness'] = stats.skew(flat_img)
    features['kurtosis'] = stats.kurtosis(flat_img)
    features['argmax'] = np.argmax(flat_img)
    for i, channel_name in enumerate(['red', 'green', 'blue']):
        channel = img_array[:, :, i].flatten()
        features[f'{channel_name}_mean'] = np.mean(channel)
        features[f'{channel_name}_std'] = np.std(channel)
        features[f'{channel_name}_variance'] = np.var(channel)
        features[f'{channel_name}_skewness'] = stats.skew(channel)
        features[f'{channel_name}_kurtosis'] = stats.kurtosis(channel)
    
    return features

def process_dataset(base_dir, model_path, num_samples=30):
    loaded_model = load_model(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Model input shape: {loaded_model.input_shape}")
    print(f"Model output shape: {loaded_model.output_shape}")
    
    all_images = []
    all_labels = []
    all_features_list = []
    seizure_dir = os.path.join(base_dir, 'test', 'seizure')
    for filename in os.listdir(seizure_dir)[:num_samples]:
        img_path = os.path.join(seizure_dir, filename)
        img_array = load_and_preprocess_image(img_path)
        if img_array is not None:
            all_images.append(img_array)
            all_labels.append(1)
            all_features_list.append(extract_features(img_array))
    non_seizure_dir = os.path.join(base_dir, 'test', 'non seizure')
    for filename in os.listdir(non_seizure_dir)[:num_samples]:
        img_path = os.path.join(non_seizure_dir, filename)
        img_array = load_and_preprocess_image(img_path)
        if img_array is not None:
            all_images.append(img_array)
            all_labels.append(0)  
            all_features_list.append(extract_features(img_array))
    images = np.array(all_images)
    labels = np.array(all_labels)
    return images, labels, all_features_list, loaded_model

def generate_tabular_shap_explanations(images, labels, model, all_features_list, num_to_analyze=30):
    # Create a directory for saving results
    results_dir = 'shap_explanations'
    os.makedirs(results_dir, exist_ok=True)
    print("Generating tabular SHAP explanations...")
    features_df = pd.DataFrame(all_features_list)
    features_df['label'] = labels
    selected_features = ['mean', 'std', 'variance', 'skewness', 'kurtosis', 
                         'red_mean', 'green_mean', 'blue_mean', 
                         'red_std', 'green_std', 'blue_std']
    num_to_analyze = min(num_to_analyze, len(images))
    sample_indices = np.random.choice(len(images), num_to_analyze, replace=False)
    sample_df = features_df.loc[sample_indices, selected_features].copy()
    sample_labels = [labels[i] for i in sample_indices]
    def model_predict(X):
        predictions = []
        for _, row in pd.DataFrame(X, columns=selected_features).iterrows():
            distances = []
            for idx, features in enumerate(all_features_list):
                dist = 0
                for feature in selected_features:
                    if feature in features:
                        dist += (row[feature] - features[feature])**2
                distances.append((idx, np.sqrt(dist)))
            nearest_idx = min(distances, key=lambda x: x[1])[0]
            pred = model.predict(np.array([images[nearest_idx]]), verbose=0)[0]
            predictions.append(pred)
        return np.array([p[1] for p in predictions])
    print("Creating SHAP KernelExplainer...")
    explainer = shap.KernelExplainer(model_predict, sample_df)
    print("Computing SHAP values...")
    shap_values = explainer.shap_values(sample_df)
    shap_df = pd.DataFrame(shap_values, columns=selected_features)
    shap_df['true_label'] = sample_labels
    shap_df.to_csv(os.path.join(results_dir, 'tabular_shap_values.csv'), index=False)
    # 1. Beeswarm plot (main visualization requested)
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, sample_df, show=False)
    plt.title("SHAP Beeswarm Plot - Feature Impact on Predictions", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_beeswarm_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Feature importance bar plot
    plt.figure(figsize=(14, 8))
    shap.summary_plot(shap_values, sample_df, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_feature_importance_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Split by class
    seizure_indices = [i for i, label in enumerate(sample_labels) if label == 1]
    non_seizure_indices = [i for i, label in enumerate(sample_labels) if label == 0]
    
    if seizure_indices:
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            [shap_values[i] for i in seizure_indices], 
            sample_df.iloc[seizure_indices], 
            show=False
        )
        plt.title("SHAP Beeswarm Plot - Seizure Class", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'shap_beeswarm_seizure.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    if non_seizure_indices:
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            [shap_values[i] for i in non_seizure_indices], 
            sample_df.iloc[non_seizure_indices], 
            show=False
        )
        plt.title("SHAP Beeswarm Plot - Non-Seizure Class", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'shap_beeswarm_non_seizure.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Force plots for individual samples
    for i in range(min(5, len(sample_df))):
        plt.figure(figsize=(20, 3))
        class_label = "Seizure" if sample_labels[i] == 1 else "Non-Seizure"
        shap.force_plot(
            explainer.expected_value, 
            shap_values[i], 
            sample_df.iloc[i],
            matplotlib=True,
            show=False
        )
        plt.title(f"SHAP Force Plot - Sample {i} (Class: {class_label})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'shap_force_plot_{i}_{class_label}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Generate dependence plots for top features
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(-feature_importance)[:3]
    top_features = [selected_features[i] for i in top_features_idx]
    
    for feature in top_features:
        plt.figure(figsize=(10, 8))
        shap.dependence_plot(
            feature, 
            shap_values, 
            sample_df,
            interaction_index=None,
            show=False
        )
        plt.title(f"SHAP Dependence Plot - {feature}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'shap_dependence_{feature}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("SHAP tabular analysis completed and saved to 'shap_explanations' directory")

if __name__ == "__main__":
    base_dir = 'D:/DSNproject/DSN2/EEG dataset + code/dataset' 
    model_path = 'model.h5'  
    print("Processing dataset...")
    images, labels, all_features_list, loaded_model = process_dataset(base_dir, model_path)
    print(f"Processed {len(images)} images")
    generate_tabular_shap_explanations(images, labels, loaded_model, all_features_list, num_to_analyze=30)