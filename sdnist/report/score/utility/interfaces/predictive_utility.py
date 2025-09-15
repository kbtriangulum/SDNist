from pathlib import Path

from sdnist.report.dataset import Dataset
from sdnist.report.dataset.data_dict import get_feature_type
from sdnist.report.report_data import (
    ReportData, ReportUIData, UtilityScorePacket, Attachment, AttachmentType
)
from sdnist.metrics.predictive_utility import PredictiveUtility
import sdnist.strs as strs


class PredictiveUtilityReport:
    def __init__(self, dataset: Dataset, ui_data: ReportUIData, report_data: ReportData):
        self.ds = dataset
        self.r_ui_d = ui_data
        self.rd = report_data
        
        # Extract feature types for all columns
        feature_types = {}
        for feature in self.ds.t_target_data.columns:
            if feature in self.ds.data_dict:
                feature_types[feature] = get_feature_type(self.ds.data_dict, feature)
            else:
                # Default to categorical if not in data dict
                feature_types[feature] = strs.CATEGORICAL
        
        # Use transformed data with feature types
        self.metric = PredictiveUtility(
            self.ds.t_target_data,
            self.ds.t_synthetic_data,
            self.r_ui_d.output_directory,
            feature_types=feature_types,
            data_dict=self.ds.data_dict
        )
        
        self.results = None
    
    def compute(self):
        # Run the metric computation
        self.results = self.metric.compute_score()
    
    def add_to_ui(self):
        if not self.results:
            return
        
        # Add to report data
        self.rd.add('predictive_utility', self.metric.report_data)
        
        # Create attachments for UI
        attachments = []
        
        # Add description paragraph
        desc_para = (
            "Predictive Utility measures how well machine learning models trained on deidentified data "
            "perform compared to models trained on real data. For each startup capital feature, "
            "we train a logistic regression model to predict that feature from all other non-SC features. "
            "Models are trained on 60% of the data and tested on 40%. "
            "The charts show the distribution of actual labels vs predictions from models trained on target and deidentified data."
        )
        
        attachments.append(
            Attachment(
                name=None,
                _data=desc_para,
                _type=AttachmentType.String
            )
        )
        
        # Add summary statistics
        summary_text = (
            f"Mean Target Accuracy: {self.metric.report_data['mean_target_accuracy']:.3f}<br>"
            f"Mean Deidentified Accuracy: {self.metric.report_data['mean_synthetic_accuracy']:.3f}<br>"
            f"Mean Degradation: {self.metric.report_data['mean_degradation']:.3f}"
        )
        
        attachments.append(
            Attachment(
                name=None,
                _data=f"Highlight-Score: {summary_text}",
                _type=AttachmentType.String
            )
        )
        
        # Add accuracy grid plot first
        grid_path = Path(self.metric.o_path, 'predictive_utility_accuracy_grid.png')
        if grid_path.exists():
            rel_path = "/".join(list(grid_path.parts)[-2:])
            attachments.append(
                Attachment(
                    name='Target and Deid Models Accuracies',
                    _data=[{strs.IMAGE_NAME: 'predictive_utility_accuracy_grid', strs.PATH: rel_path}],
                    _type=AttachmentType.ImageLinks
                )
            )
        
        # Add combined plot
        plot_path = Path(self.metric.o_path, 'predictive_utility_combined.png')
        if plot_path.exists():
            rel_path = "/".join(list(plot_path.parts)[-2:])
            attachments.append(
                Attachment(
                    name='Predictive Utility Analysis',
                    _data=[{strs.IMAGE_NAME: 'predictive_utility_combined', strs.PATH: rel_path}],
                    _type=AttachmentType.ImageLinks
                )
            )
        
        # Create packet and add to UI
        packet = UtilityScorePacket(
            self.metric.NAME,
            None,
            attachments
        )
        self.r_ui_d.add(packet)