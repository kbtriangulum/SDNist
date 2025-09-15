import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

from sdnist.utils import *
import sdnist.strs as strs


class PredictiveUtility:
    NAME = 'Predictive Utility'
    
    # SC features to use as targets
    SC_TARGETS = [
        'SCSAVINGS', 'SCASSETS',
        'SCEQUITY', 'SCCREDIT',
        'SCGOVTLOAN', 'SCGOVTGUAR', 'SCBANKLOAN', 'SCFAMLOAN',
        'SCVENTURE', 'SCGRANT', 'SCOTHER', 'SCDONTKNOW',
        'SCNONENEEDED', 'SCNOTREPORTED',
        'SCAMOUNT'
    ]
    
    def __init__(self, target, synthetic, output_directory, feature_types=None, data_dict=None):
        self.target = target
        self.synthetic = synthetic
        self.o_dir = output_directory
        self.o_path = Path(self.o_dir, 'predictive_utility')
        self.feature_types = feature_types or {}
        self.data_dict = data_dict or {}
        
        # Results storage
        self.results = {}
        self.report_data = {}
        
        self._setup()
    
    def _get_feature_description(self, feature):
        """Get feature description from data dictionary"""
        if feature in self.data_dict and 'description' in self.data_dict[feature]:
            return self.data_dict[feature]['description']
        else:
            # Return a default description if not available
            return f"Startup Capital: {feature}"
    
    def _setup(self):
        if not self.o_dir.exists():
            raise Exception(f'Path {self.o_dir} does not exist')
        if not self.o_path.exists():
            os.mkdir(self.o_path)
    
    def _create_preprocessor(self, feature_columns):
        """Create preprocessor for selective scaling and one-hot encoding"""
        continuous_features = []
        categorical_features = []
        
        for feature in feature_columns:
            feature_type = self.feature_types.get(feature, strs.CATEGORICAL)
            if feature_type == strs.CONTINUOUS:
                continuous_features.append(feature)
            else:
                categorical_features.append(feature)
        
        transformers = []
        
        if continuous_features:
            transformers.append(('scaler', StandardScaler(), continuous_features))
        
        if categorical_features:
            transformers.append(('onehot', OneHotEncoder(sparse_output=False, handle_unknown='error'), categorical_features))
        
        if not transformers:
            return None
        
        return ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    def compute_score(self):
        # Get predictor columns (all non-SC columns)
        X_columns = [col for col in self.target.columns 
                    if col not in self.SC_TARGETS]
        
        # Get X features once for all models
        X_target = self.target[X_columns]
        X_synthetic = self.synthetic[X_columns]
        
        # Create preprocessors once
        preprocessor_target = self._create_preprocessor(X_columns)
        preprocessor_synthetic = self._create_preprocessor(X_columns)
        
        # Process each SC target
        for sc_target in self.SC_TARGETS:
            print('processing: ', sc_target)
            if sc_target not in self.target.columns:
                continue
            
            # Get target variables
            y_target = self.target[sc_target]
            y_synthetic = self.synthetic[sc_target]
            
            # Split target data (X and y together)
            X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(
                X_target, y_target, test_size=0.4, random_state=42
            )
            
            # Split synthetic data (X and y together)
            X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
                X_synthetic, y_synthetic, test_size=0.4, random_state=42
            )
            
            # Apply preprocessing for each split
            if preprocessor_target is not None:
                X_t_train_processed = preprocessor_target.fit_transform(X_t_train)
                X_t_test_processed = preprocessor_target.transform(X_t_test)
                
                X_s_train_processed = preprocessor_synthetic.fit_transform(X_s_train)
                X_s_test_processed = preprocessor_synthetic.transform(X_s_test)
            else:
                # No preprocessing needed (all features same type or no transformers)
                X_t_train_processed = X_t_train.values
                X_t_test_processed = X_t_test.values
                X_s_train_processed = X_s_train.values
                X_s_test_processed = X_s_test.values

            hyper_params = {
                'max_iter': 100,
                'solver': 'lbfgs',
                'n_jobs': -1,
                'random_state': 42
            }
            # Train models
            model_target = LogisticRegression(**hyper_params)
            model_target.fit(X_t_train_processed, y_t_train)
            
            model_synthetic = LogisticRegression(**hyper_params)
            model_synthetic.fit(X_s_train_processed, y_s_train)
            
            # Get predictions on their respective test sets
            pred_target_on_target = model_target.predict(X_t_test_processed)
            pred_synthetic_on_synthetic = model_synthetic.predict(X_s_test_processed)
            
            # Also get cross predictions for better comparison
            pred_synthetic_on_target = model_synthetic.predict(X_t_test_processed)
            
            # Calculate metrics - both models evaluated on target test set
            acc_target = accuracy_score(y_t_test, pred_target_on_target)
            acc_synthetic_on_target = accuracy_score(y_t_test, pred_synthetic_on_target)
            
            # Store results
            self.results[sc_target] = {
                'actual_target': y_t_test,
                'pred_target_on_target': pred_target_on_target,
                'pred_synthetic_on_target': pred_synthetic_on_target,  # Synthetic model on target test
                'acc_target': acc_target,
                'acc_synthetic_on_target': acc_synthetic_on_target,
                'degradation': acc_target - acc_synthetic_on_target  # Compare on same test set
            }
        
        # Create visualizations
        self._create_accuracy_grid()
        self._create_plot()
        
        # Prepare report data
        self._prepare_report_data()
        
        return self.results
    
    def _create_accuracy_grid(self):
        """Create accuracy grid chart with target features on y-axis and accuracy comparisons on x-axis"""
        if not self.results:
            return
        
        # Prepare data for grid
        features = list(self.results.keys())
        n_features = len(features)
        
        # Create accuracy matrix: [target_acc, deid_acc, abs_diff]
        accuracy_matrix = np.zeros((n_features, 3))
        
        for i, feature in enumerate(features):
            target_acc = self.results[feature]['acc_target']
            deid_acc = self.results[feature]['acc_synthetic_on_target']
            abs_diff = abs(target_acc - deid_acc)
            
            accuracy_matrix[i, 0] = target_acc
            accuracy_matrix[i, 1] = deid_acc
            accuracy_matrix[i, 2] = abs_diff
        
        # Create figure with subplots for different colorbars
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, max(3, n_features * 0.5)), gridspec_kw={'width_ratios': [2, 1]})
        
        # First subplot: Target and Deid accuracies
        acc_grid = accuracy_matrix[:, :2]
        im1 = ax1.imshow(acc_grid, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels for first subplot
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Target', 'Deid'])
        ax1.set_yticks(range(n_features))
        ax1.set_yticklabels(features)
        ax1.set_title('Model Accuracies', fontsize=12)
        
        # Add text annotations for accuracies
        for i in range(n_features):
            for j in range(2):
                text = f'{acc_grid[i, j]:.3f}'
                ax1.text(j, i, text, ha='center', va='center', fontsize=9, 
                        color='white' if acc_grid[i, j] > 0.5 else 'black')
        
        # Add colorbar for first subplot
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Accuracy', rotation=270, labelpad=15)
        
        # Second subplot: Absolute differences
        diff_grid = accuracy_matrix[:, 2:3]
        im2 = ax2.imshow(diff_grid, cmap='Reds', aspect='auto', vmin=0, vmax=max(0.1, np.max(diff_grid)))
        
        # Set ticks and labels for second subplot
        ax2.set_xticks([0])
        ax2.set_xticklabels(['Diff'])
        ax2.set_yticks(range(n_features))
        ax2.set_yticklabels([])  # No labels since they're on the left subplot
        ax2.set_title('Absolute Difference', fontsize=12)
        
        # Add text annotations for differences
        for i in range(n_features):
            text = f'{diff_grid[i, 0]:.3f}'
            ax2.text(0, i, text, ha='center', va='center', fontsize=9,
                    color='white' if diff_grid[i, 0] > 0.05 else 'black')
        
        # Add colorbar for second subplot
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('|Target - Deid|', rotation=270, labelpad=15)
        
        # Overall title and layout
        fig.suptitle('Target and Deid Models Accuracies', fontsize=14)
        plt.tight_layout()
        
        # Save the plot
        grid_path = Path(self.o_path, 'predictive_utility_accuracy_grid.png')
        fig.savefig(grid_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return grid_path
    
    def _create_plot(self):
        # Calculate grid dimensions for combined plot
        n_targets = len(self.results)
        n_cols = 2  # 2 columns: counts and percentage difference
        n_rows = n_targets
        
        # Create combined plot with 2 columns per target (extra height for headers)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 3.5))
        if n_targets == 1:
            axes = axes.reshape(1, -1)  # Ensure 2D array for single target
        
        for idx, (sc_target, data) in enumerate(self.results.items()):
            # Add feature header above the chart pair
            feature_desc = self._get_feature_description(sc_target)
            avl_figh = 0.973
            h_per_row = avl_figh / n_rows
            # Calculate y position for header - position above each row of subplots
            header_y = avl_figh - (idx * h_per_row)
            fig.text(0.05, header_y, f"{sc_target}: {feature_desc}",
                    ha='left', va='top', fontsize=11, fontweight='bold',
                    transform=fig.transFigure)
            
            # Column 1: Distribution comparison
            ax1 = axes[idx, 0]
            
            # Compare on target test set
            actual_target = data['actual_target']
            pred_target_model = data['pred_target_on_target']
            pred_deid_model = data['pred_synthetic_on_target']  # Both models on same test set!
            
            # Get unique labels
            labels = sorted(set(actual_target) | set(pred_target_model) | set(pred_deid_model))
            
            # Count occurrences - all on target test set
            actual_counts = [np.sum(actual_target == label) for label in labels]
            target_model_counts = [np.sum(pred_target_model == label) for label in labels]
            deid_model_counts = [np.sum(pred_deid_model == label) for label in labels]
            
            # Create bars for distribution comparison
            x = np.arange(len(labels))
            width = 0.25
            
            ax1.bar(x - width, actual_counts, width, label='Actual', color='#ff9500', alpha=0.8)
            ax1.bar(x, target_model_counts, width, label='Target Model', color='#5373d8', alpha=0.8)
            ax1.bar(x + width, deid_model_counts, width, label='Deidentified Model', color='#4eb07a', alpha=0.8)
            
            ax1.set_xlabel('Label')
            ax1.set_ylabel('Count')
            ax1.set_title(f'{sc_target} - Counts\nTar Acc={data["acc_target"]:.3f}, Deid Acc={data["acc_synthetic_on_target"]:.3f}', fontsize=10)
            ax1.set_xticks(x)
            ax1.set_xticklabels([str(l) for l in labels])
            ax1.legend(fontsize=8, framealpha=0.7)
            ax1.grid(axis='y', alpha=0.3)
            
            # Column 2: Percentage differences with 3 bars per label
            ax2 = axes[idx, 1]
            
            # Calculate three different percentage differences
            actual_target_diffs = []  # (actual - target) / actual * 100
            actual_deid_diffs = []  # (actual - deidentified) / actual * 100
            target_deid_diffs = []  # (target - deidentified) / actual * 100
            
            for i, label in enumerate(labels):
                actual_count = actual_counts[i]
                target_count = target_model_counts[i]
                deid_count = deid_model_counts[i]
                
                if actual_count > 0:
                    actual_target_diff = ((actual_count - target_count) / actual_count) * 100
                    actual_deid_diff = ((actual_count - deid_count) / actual_count) * 100
                    target_deid_diff = ((target_count - deid_count) / actual_count) * 100
                else:
                    # If actual count is 0, use simple differences
                    actual_target_diff = -target_count * 100 if target_count > 0 else 0
                    actual_deid_diff = -deid_count * 100 if deid_count > 0 else 0
                    target_deid_diff = (target_count - deid_count) * 100 if (target_count + deid_count) > 0 else 0
                
                actual_target_diffs.append(actual_target_diff)
                actual_deid_diffs.append(actual_deid_diff)
                target_deid_diffs.append(target_deid_diff)
            
            # Create bars for percentage differences - 3 bars per label
            width = 0.25
            ax2.bar(x - width, actual_target_diffs, width, label='Act - Tar', color='#5373d8', alpha=0.8)
            ax2.bar(x, actual_deid_diffs, width, label='Act - Deid', color='#4eb07a', alpha=0.8)
            ax2.bar(x + width, target_deid_diffs, width, label='Tar - Deid', color='#e11d48', alpha=0.8)
            
            # Add horizontal line at 0
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            # Calculate y-axis limits with padding for text
            all_diffs = actual_target_diffs + actual_deid_diffs + target_deid_diffs
            if all_diffs:
                y_min = min(all_diffs) * 1.2 if min(all_diffs) < 0 else min(all_diffs) - 10
                y_max = max(all_diffs) * 1.2 if max(all_diffs) > 0 else max(all_diffs) + 10
                ax2.set_ylim(y_min, y_max)
            
            ax2.set_xlabel('Label')
            ax2.set_ylabel('Difference (%)')
            ax2.set_title(f'{sc_target} - Differences', fontsize=10)
            ax2.set_xticks(x)
            ax2.set_xticklabels([str(l) for l in labels])
            ax2.legend(fontsize=7, loc='upper right', framealpha=0.7)
            ax2.grid(axis='y', alpha=0.3)
        
        # Save combined plot with proper spacing
        fig.suptitle('Predictive Utility Analysis', fontsize=14)
        fig.tight_layout(pad=1.5, h_pad=3.0, rect=(0, 0, 1, 0.98))  # Minimal space for headers
        plot_path = Path(self.o_path, 'predictive_utility_combined.png')
        fig.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return [plot_path]
    
    def _prepare_report_data(self):
        # Summary statistics
        accuracies_target = [r['acc_target'] for r in self.results.values()]
        accuracies_synthetic = [r['acc_synthetic_on_target'] for r in self.results.values()]
        degradations = [r['degradation'] for r in self.results.values()]
        
        self.report_data = {
            'features_evaluated': list(self.results.keys()),
            'mean_target_accuracy': np.mean(accuracies_target),
            'mean_synthetic_accuracy': np.mean(accuracies_synthetic),
            'mean_degradation': np.mean(degradations),
            'individual_results': {
                k: {
                    'accuracy_target': v['acc_target'],
                    'accuracy_synthetic': v['acc_synthetic_on_target'],
                    'degradation': v['degradation']
                }
                for k, v in self.results.items()
            }
        }