"""
Intelligent Anomaly Detection - ML-powered anomaly detection for training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

from .io.schemas import AnomalyReport, TrainingMetrics

@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection"""
    models: List[str] = None
    adaptive_thresholds: bool = True
    context_aware: bool = True
    sensitivity: float = 0.1  # Lower = more sensitive
    window_size: int = 100
    min_anomalies: int = 3
    confidence_threshold: float = 0.8
    
    def __post_init__(self):
        if self.models is None:
            self.models = ['isolation_forest', 'autoencoder', 'statistical']

class Autoencoder(nn.Module):
    """Simple autoencoder for anomaly detection"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class StatisticalDetector:
    """Statistical anomaly detection methods"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = []
        
    def detect_anomalies(self, data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect anomalies using statistical methods"""
        anomalies = np.zeros(len(data), dtype=bool)
        details = {}
        
        if len(data) < self.window_size:
            return anomalies, details
        
        # Z-score based detection
        z_scores = self._calculate_z_scores(data)
        z_anomalies = np.abs(z_scores) > 3.0
        anomalies |= z_anomalies
        
        # Moving average based detection
        ma_anomalies = self._moving_average_detection(data)
        anomalies |= ma_anomalies
        
        # Variance based detection
        var_anomalies = self._variance_detection(data)
        anomalies |= var_anomalies
        
        details = {
            'z_scores': z_scores.tolist(),
            'z_anomalies': z_anomalies.sum(),
            'ma_anomalies': ma_anomalies.sum(),
            'var_anomalies': var_anomalies.sum(),
            'total_anomalies': anomalies.sum()
        }
        
        return anomalies, details
    
    def _calculate_z_scores(self, data: np.ndarray) -> np.ndarray:
        """Calculate rolling z-scores"""
        z_scores = np.zeros_like(data)
        
        for i in range(self.window_size, len(data)):
            window = data[i-self.window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            if std > 0:
                z_scores[i] = (data[i] - mean) / std
        
        return z_scores
    
    def _moving_average_detection(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies using moving average"""
        anomalies = np.zeros_like(data, dtype=bool)
        
        for i in range(self.window_size, len(data)):
            window = data[i-self.window_size:i]
            ma = np.mean(window)
            threshold = 2.0 * np.std(window)
            
            if abs(data[i] - ma) > threshold:
                anomalies[i] = True
        
        return anomalies
    
    def _variance_detection(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies using variance changes"""
        anomalies = np.zeros_like(data, dtype=bool)
        
        for i in range(self.window_size, len(data)):
            window = data[i-self.window_size:i]
            current_var = np.var(window)
            prev_var = np.var(data[max(0, i-2*self.window_size):i-self.window_size])
            
            if prev_var > 0 and current_var / prev_var > 3.0:
                anomalies[i] = True
        
        return anomalies

class AnomalyDetector:
    """
    Intelligent Anomaly Detection - ML-powered anomaly detection for training
    
    Features:
    - Multiple ML models (Isolation Forest, Autoencoder, Statistical)
    - Adaptive thresholds based on training context
    - Context-aware detection
    - Confidence scoring
    - Real-time detection
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        self.config = config or AnomalyConfig()
        self.scaler = StandardScaler()
        self.models = {}
        self.statistical_detector = StatisticalDetector(self.config.window_size)
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize ML models"""
        if 'isolation_forest' in self.config.models:
            self.models['isolation_forest'] = IsolationForest(
                contamination=self.config.sensitivity,
                random_state=42
            )
        
        if 'local_outlier_factor' in self.config.models:
            self.models['lof'] = LocalOutlierFactor(
                contamination=self.config.sensitivity,
                novelty=True
            )
    
    def detect_anomalies(self, data: Union[np.ndarray, pd.DataFrame, List[Dict]], 
                        features: Optional[List[str]] = None) -> AnomalyReport:
        """
        Detect anomalies in training data
        
        Args:
            data: Training data (numpy array, pandas DataFrame, or list of dicts)
            features: Feature names (if not provided, will be auto-detected)
            
        Returns:
            AnomalyReport with detected anomalies and details
        """
        # Convert data to numpy array
        if isinstance(data, list):
            data = self._convert_dict_list_to_array(data, features)
        elif isinstance(data, pd.DataFrame):
            data = data.values
            if features is None:
                features = list(data.columns)
        
        if features is None:
            features = [f"feature_{i}" for i in range(data.shape[1])]
        
        # Preprocess data
        data_clean = self._preprocess_data(data)
        
        # Run different detection methods
        results = {}
        
        # Statistical detection
        if 'statistical' in self.config.models:
            stat_anomalies, stat_details = self.statistical_detector.detect_anomalies(data_clean)
            results['statistical'] = {
                'anomalies': stat_anomalies,
                'details': stat_details,
                'confidence': self._calculate_confidence(stat_anomalies, data_clean)
            }
        
        # Isolation Forest
        if 'isolation_forest' in self.models:
            if_anomalies = self._detect_with_isolation_forest(data_clean)
            results['isolation_forest'] = {
                'anomalies': if_anomalies,
                'confidence': self._calculate_confidence(if_anomalies, data_clean)
            }
        
        # Autoencoder
        if 'autoencoder' in self.config.models:
            ae_anomalies = self._detect_with_autoencoder(data_clean)
            results['autoencoder'] = {
                'anomalies': ae_anomalies,
                'confidence': self._calculate_confidence(ae_anomalies, data_clean)
            }
        
        # Combine results
        combined_anomalies = self._combine_results(results)
        
        # Generate report
        report = self._generate_report(combined_anomalies, results, features)
        
        return report
    
    def _convert_dict_list_to_array(self, data: List[Dict], features: Optional[List[str]] = None) -> np.ndarray:
        """Convert list of dictionaries to numpy array"""
        if not data:
            return np.array([])
        
        if features is None:
            # Auto-detect features from first item
            features = list(data[0].keys())
        
        # Extract numeric features
        numeric_features = []
        for feature in features:
            if all(isinstance(item.get(feature), (int, float)) for item in data):
                numeric_features.append(feature)
        
        # Convert to array
        array_data = []
        for item in data:
            row = [item.get(feature, 0.0) for feature in numeric_features]
            array_data.append(row)
        
        return np.array(array_data)
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess data for anomaly detection"""
        # Remove infinite values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Standardize if needed
        if len(data) > 0:
            data = self.scaler.fit_transform(data)
        
        return data
    
    def _detect_with_isolation_forest(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies using Isolation Forest"""
        if len(data) == 0:
            return np.array([])
        
        # Fit model on first part of data
        split_point = min(len(data) // 2, 1000)
        if split_point > 0:
            self.models['isolation_forest'].fit(data[:split_point])
        
        # Predict on all data
        predictions = self.models['isolation_forest'].predict(data)
        return predictions == -1  # -1 indicates anomaly
    
    def _detect_with_autoencoder(self, data: np.ndarray) -> np.ndarray:
        """Detect anomalies using autoencoder"""
        if len(data) < 10:
            return np.zeros(len(data), dtype=bool)
        
        # Create autoencoder
        input_dim = data.shape[1]
        autoencoder = Autoencoder(input_dim, hidden_dim=min(32, input_dim))
        
        # Convert to torch tensors
        data_tensor = torch.FloatTensor(data)
        dataset = TensorDataset(data_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Train autoencoder
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        
        autoencoder.train()
        for epoch in range(10):  # Quick training
            for batch in dataloader:
                x = batch[0]
                optimizer.zero_grad()
                output = autoencoder(x)
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()
        
        # Calculate reconstruction errors
        autoencoder.eval()
        reconstruction_errors = []
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                output = autoencoder(x)
                error = torch.mean((x - output) ** 2, dim=1)
                reconstruction_errors.extend(error.numpy())
        
        reconstruction_errors = np.array(reconstruction_errors)
        
        # Detect anomalies based on reconstruction error
        threshold = np.percentile(reconstruction_errors, 95)
        anomalies = reconstruction_errors > threshold
        
        return anomalies
    
    def _calculate_confidence(self, anomalies: np.ndarray, data: np.ndarray) -> float:
        """Calculate confidence in anomaly detection"""
        if len(anomalies) == 0:
            return 0.0
        
        # Confidence based on anomaly ratio and data quality
        anomaly_ratio = np.mean(anomalies)
        
        # Higher confidence for moderate anomaly ratios
        if 0.01 <= anomaly_ratio <= 0.1:
            confidence = 0.9
        elif 0.001 <= anomaly_ratio <= 0.05:
            confidence = 0.8
        elif anomaly_ratio > 0.2:
            confidence = 0.3  # Too many anomalies might indicate poor detection
        else:
            confidence = 0.5
        
        return confidence
    
    def _combine_results(self, results: Dict[str, Dict]) -> np.ndarray:
        """Combine results from different detection methods"""
        if not results:
            return np.array([])
        
        # Get all anomaly arrays
        anomaly_arrays = []
        weights = []
        
        for method, result in results.items():
            if 'anomalies' in result:
                anomaly_arrays.append(result['anomalies'])
                confidence = result.get('confidence', 0.5)
                weights.append(confidence)
        
        if not anomaly_arrays:
            return np.array([])
        
        # Weighted voting
        combined = np.zeros_like(anomaly_arrays[0], dtype=float)
        total_weight = sum(weights)
        
        for anomalies, weight in zip(anomaly_arrays, weights):
            combined += (anomalies.astype(float) * weight / total_weight)
        
        # Apply threshold
        final_anomalies = combined > self.config.confidence_threshold
        
        return final_anomalies
    
    def _generate_report(self, anomalies: np.ndarray, results: Dict, 
                        features: List[str]) -> AnomalyReport:
        """Generate comprehensive anomaly report"""
        # Find anomaly indices
        anomaly_indices = np.where(anomalies)[0]
        
        # Calculate statistics
        total_points = len(anomalies)
        anomaly_count = len(anomaly_indices)
        anomaly_ratio = anomaly_count / total_points if total_points > 0 else 0
        
        # Generate details
        details = {
            'total_points': total_points,
            'anomaly_count': anomaly_count,
            'anomaly_ratio': anomaly_ratio,
            'anomaly_indices': anomaly_indices.tolist(),
            'methods_used': list(results.keys()),
            'method_details': results
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(anomaly_ratio, results)
        
        # Create report
        report = AnomalyReport(
            anomalies_detected=anomaly_count > 0,
            anomaly_count=anomaly_count,
            anomaly_ratio=anomaly_ratio,
            confidence=self._calculate_overall_confidence(results),
            details=details,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        return report
    
    def _calculate_overall_confidence(self, results: Dict) -> float:
        """Calculate overall confidence in detection"""
        if not results:
            return 0.0
        
        confidences = [result.get('confidence', 0.5) for result in results.values()]
        return np.mean(confidences)
    
    def _generate_recommendations(self, anomaly_ratio: float, results: Dict) -> List[str]:
        """Generate recommendations based on anomaly detection results"""
        recommendations = []
        
        if anomaly_ratio > 0.2:
            recommendations.append("High anomaly rate detected. Consider checking data quality and preprocessing.")
        elif anomaly_ratio > 0.1:
            recommendations.append("Moderate anomaly rate. Monitor training closely and investigate specific anomalies.")
        elif anomaly_ratio > 0.05:
            recommendations.append("Some anomalies detected. Review training parameters and data.")
        else:
            recommendations.append("Low anomaly rate. Training appears normal.")
        
        # Method-specific recommendations
        if 'isolation_forest' in results:
            if_anomalies = results['isolation_forest']['anomalies'].sum()
            if if_anomalies > 0:
                recommendations.append(f"Isolation Forest detected {if_anomalies} anomalies - check for data distribution shifts.")
        
        if 'autoencoder' in results:
            ae_anomalies = results['autoencoder']['anomalies'].sum()
            if ae_anomalies > 0:
                recommendations.append(f"Autoencoder detected {ae_anomalies} anomalies - check for unusual patterns in training data.")
        
        return recommendations
    
    def detect_training_anomalies(self, training_logs: Union[str, Path, List[Dict]]) -> AnomalyReport:
        """
        Specialized method for detecting training-specific anomalies
        
        Args:
            training_logs: Path to training logs or list of log entries
            
        Returns:
            AnomalyReport with training-specific anomalies
        """
        # Load training logs
        if isinstance(training_logs, (str, Path)):
            logs = self._load_training_logs(training_logs)
        else:
            logs = training_logs
        
        # Extract relevant features for training anomaly detection
        features = ['step', 'loss', 'reward', 'kl_divergence', 'entropy', 'value_loss']
        
        # Detect anomalies
        report = self.detect_anomalies(logs, features)
        
        # Add training-specific analysis
        report = self._add_training_analysis(report, logs)
        
        return report
    
    def _load_training_logs(self, log_path: Union[str, Path]) -> List[Dict]:
        """Load training logs from file"""
        log_path = Path(log_path)
        logs = []
        
        if log_path.is_file():
            with open(log_path, 'r') as f:
                for line in f:
                    try:
                        logs.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        elif log_path.is_dir():
            for file_path in log_path.rglob("*.jsonl"):
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            logs.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
        
        return logs
    
    def _add_training_analysis(self, report: AnomalyReport, logs: List[Dict]) -> AnomalyReport:
        """Add training-specific analysis to report"""
        if not logs:
            return report
        
        # Extract training metrics
        steps = [log.get('step', i) for i, log in enumerate(logs)]
        losses = [log.get('loss', 0.0) for log in logs]
        rewards = [log.get('reward', 0.0) for log in logs]
        kl_divs = [log.get('kl_divergence', 0.0) for log in logs]
        
        # Training-specific anomaly detection
        training_anomalies = []
        
        # Check for loss spikes
        if len(losses) > 10:
            loss_std = np.std(losses)
            loss_mean = np.mean(losses)
            for i, loss in enumerate(losses):
                if abs(loss - loss_mean) > 3 * loss_std:
                    training_anomalies.append(f"Loss spike at step {steps[i]}: {loss}")
        
        # Check for reward drops
        if len(rewards) > 10:
            for i in range(1, len(rewards)):
                if rewards[i] < rewards[i-1] * 0.5:  # 50% drop
                    training_anomalies.append(f"Reward drop at step {steps[i]}: {rewards[i-1]} -> {rewards[i]}")
        
        # Check for KL divergence spikes
        if len(kl_divs) > 10:
            kl_std = np.std(kl_divs)
            kl_mean = np.mean(kl_divs)
            for i, kl in enumerate(kl_divs):
                if kl > kl_mean + 3 * kl_std:
                    training_anomalies.append(f"KL divergence spike at step {steps[i]}: {kl}")
        
        # Add to report
        report.details['training_anomalies'] = training_anomalies
        report.recommendations.extend(self._generate_training_recommendations(training_anomalies))
        
        return report
    
    def _generate_training_recommendations(self, training_anomalies: List[str]) -> List[str]:
        """Generate training-specific recommendations"""
        recommendations = []
        
        loss_spikes = [a for a in training_anomalies if 'Loss spike' in a]
        reward_drops = [a for a in training_anomalies if 'Reward drop' in a]
        kl_spikes = [a for a in training_anomalies if 'KL divergence spike' in a]
        
        if loss_spikes:
            recommendations.append(f"Detected {len(loss_spikes)} loss spikes. Consider reducing learning rate or checking data quality.")
        
        if reward_drops:
            recommendations.append(f"Detected {len(reward_drops)} reward drops. Check reward model stability and training data.")
        
        if kl_spikes:
            recommendations.append(f"Detected {len(kl_spikes)} KL divergence spikes. Consider adjusting KL penalty coefficient.")
        
        return recommendations

# Convenience functions
def detect_anomalies(data: Union[np.ndarray, pd.DataFrame, List[Dict]], **kwargs) -> AnomalyReport:
    """Quick anomaly detection with default settings"""
    detector = AnomalyDetector()
    return detector.detect_anomalies(data, **kwargs)

def detect_training_anomalies(training_logs: Union[str, Path, List[Dict]]) -> AnomalyReport:
    """Quick training anomaly detection"""
    detector = AnomalyDetector()
    return detector.detect_training_anomalies(training_logs)