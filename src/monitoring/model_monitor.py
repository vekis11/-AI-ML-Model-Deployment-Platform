"""
Model monitoring for performance tracking and health checks.
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.metrics_exporter import AzureMetricsExporter
from opencensus.stats import aggregation, measure, view
from opencensus.stats.aggregation import CountAggregation, LastValueAggregation
from opencensus.stats.measure import MeasureFloat, MeasureInt
from opencensus.stats.view import View
from opencensus.tags import tag_key, tag_map, tag_value

from ..training.model_config import ModelConfig


class ModelMonitor:
    """Model monitoring for performance tracking and health checks."""
    
    def __init__(self, model_name: str, config: ModelConfig = None):
        """Initialize the model monitor."""
        self.model_name = model_name
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup Azure monitoring
        self._setup_azure_monitoring()
        
        # Initialize metrics
        self._setup_metrics()
        
        # Performance tracking
        self.performance_history = []
        self.error_count = 0
        self.total_requests = 0
        
    def _setup_azure_monitoring(self):
        """Setup Azure Application Insights monitoring."""
        try:
            connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
            if connection_string:
                # Setup logging
                self.logger.addHandler(AzureLogHandler(connection_string=connection_string))
                
                # Setup metrics exporter
                self.metrics_exporter = AzureMetricsExporter(connection_string=connection_string)
                self.logger.info("Azure monitoring enabled")
            else:
                self.metrics_exporter = None
                self.logger.warning("Azure Application Insights connection string not found")
        except Exception as e:
            self.logger.warning(f"Failed to setup Azure monitoring: {e}")
            self.metrics_exporter = None
    
    def _setup_metrics(self):
        """Setup custom metrics."""
        # Define measures
        self.prediction_latency_measure = MeasureFloat(
            "prediction_latency", 
            "Time taken for prediction", 
            "milliseconds"
        )
        
        self.prediction_accuracy_measure = MeasureFloat(
            "prediction_accuracy", 
            "Model prediction accuracy", 
            "percentage"
        )
        
        self.request_count_measure = MeasureInt(
            "request_count", 
            "Number of prediction requests", 
            "requests"
        )
        
        self.error_count_measure = MeasureInt(
            "error_count", 
            "Number of prediction errors", 
            "errors"
        )
        
        # Define views
        self.prediction_latency_view = View(
            "prediction_latency",
            "Distribution of prediction latency",
            [tag_key("model_name"), tag_key("endpoint")],
            self.prediction_latency_measure,
            aggregation.DistributionAggregation(0, 10000, 10)
        )
        
        self.prediction_accuracy_view = View(
            "prediction_accuracy",
            "Last value of prediction accuracy",
            [tag_key("model_name")],
            self.prediction_accuracy_measure,
            LastValueAggregation()
        )
        
        self.request_count_view = View(
            "request_count",
            "Total number of requests",
            [tag_key("model_name")],
            self.request_count_measure,
            CountAggregation()
        )
        
        self.error_count_view = View(
            "error_count",
            "Total number of errors",
            [tag_key("model_name")],
            self.error_count_measure,
            CountAggregation()
        )
    
    def record_prediction(self, latency: float, accuracy: float = None, 
                         confidence: float = None, error: bool = False):
        """Record a prediction event."""
        timestamp = datetime.now()
        
        # Update counters
        self.total_requests += 1
        if error:
            self.error_count += 1
        
        # Store performance data
        performance_data = {
            'timestamp': timestamp.isoformat(),
            'latency': latency,
            'accuracy': accuracy,
            'confidence': confidence,
            'error': error
        }
        self.performance_history.append(performance_data)
        
        # Keep only last 1000 records
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        # Log to Azure if available
        if self.metrics_exporter:
            try:
                # Create tag map
                tmap = tag_map.TagMap()
                tmap.insert(tag_key.TagKey("model_name"), tag_value.TagValue(self.model_name))
                
                # Record metrics
                self.metrics_exporter.export_metrics([
                    self.prediction_latency_measure.create_measurement(latency, tmap),
                    self.request_count_measure.create_measurement(1, tmap)
                ])
                
                if accuracy is not None:
                    self.metrics_exporter.export_metrics([
                        self.prediction_accuracy_measure.create_measurement(accuracy, tmap)
                    ])
                
                if error:
                    self.metrics_exporter.export_metrics([
                        self.error_count_measure.create_measurement(1, tmap)
                    ])
                    
            except Exception as e:
                self.logger.error(f"Failed to export metrics: {e}")
        
        # Log locally
        if error:
            self.logger.error(f"Prediction error - Latency: {latency}ms")
        else:
            self.logger.info(f"Prediction recorded - Latency: {latency}ms, Accuracy: {accuracy}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter recent data
        recent_data = [
            data for data in self.performance_history
            if datetime.fromisoformat(data['timestamp']) > cutoff_time
        ]
        
        if not recent_data:
            return {
                'total_requests': 0,
                'error_rate': 0.0,
                'avg_latency': 0.0,
                'avg_accuracy': 0.0,
                'avg_confidence': 0.0
            }
        
        # Calculate metrics
        total_requests = len(recent_data)
        error_count = sum(1 for data in recent_data if data['error'])
        error_rate = (error_count / total_requests) * 100 if total_requests > 0 else 0
        
        latencies = [data['latency'] for data in recent_data if not data['error']]
        avg_latency = np.mean(latencies) if latencies else 0
        
        accuracies = [data['accuracy'] for data in recent_data if data['accuracy'] is not None]
        avg_accuracy = np.mean(accuracies) if accuracies else 0
        
        confidences = [data['confidence'] for data in recent_data if data['confidence'] is not None]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'total_requests': total_requests,
            'error_rate': error_rate,
            'avg_latency': avg_latency,
            'avg_accuracy': avg_accuracy,
            'avg_confidence': avg_confidence,
            'min_latency': min(latencies) if latencies else 0,
            'max_latency': max(latencies) if latencies else 0,
            'p95_latency': np.percentile(latencies, 95) if latencies else 0,
            'p99_latency': np.percentile(latencies, 99) if latencies else 0
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        summary = self.get_performance_summary(hours=1)  # Last hour
        
        # Define health thresholds
        error_rate_threshold = 5.0  # 5%
        latency_threshold = 1000.0  # 1 second
        
        # Determine health status
        is_healthy = (
            summary['error_rate'] < error_rate_threshold and
            summary['avg_latency'] < latency_threshold
        )
        
        health_status = "healthy" if is_healthy else "unhealthy"
        
        # Identify issues
        issues = []
        if summary['error_rate'] >= error_rate_threshold:
            issues.append(f"High error rate: {summary['error_rate']:.2f}%")
        if summary['avg_latency'] >= latency_threshold:
            issues.append(f"High latency: {summary['avg_latency']:.2f}ms")
        
        return {
            'status': health_status,
            'is_healthy': is_healthy,
            'issues': issues,
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_metrics(self, output_path: str):
        """Export metrics to file."""
        data = {
            'model_name': self.model_name,
            'export_timestamp': datetime.now().isoformat(),
            'performance_history': self.performance_history,
            'current_summary': self.get_performance_summary(),
            'health_status': self.get_health_status()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Metrics exported to {output_path}")
    
    def clear_history(self):
        """Clear performance history."""
        self.performance_history = []
        self.error_count = 0
        self.total_requests = 0
        self.logger.info("Performance history cleared")
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get alerts based on performance thresholds."""
        alerts = []
        summary = self.get_performance_summary(hours=1)
        
        # Error rate alert
        if summary['error_rate'] > 5.0:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'high',
                'message': f"Error rate is {summary['error_rate']:.2f}% (threshold: 5%)",
                'timestamp': datetime.now().isoformat()
            })
        
        # Latency alert
        if summary['avg_latency'] > 1000.0:
            alerts.append({
                'type': 'high_latency',
                'severity': 'medium',
                'message': f"Average latency is {summary['avg_latency']:.2f}ms (threshold: 1000ms)",
                'timestamp': datetime.now().isoformat()
            })
        
        # P99 latency alert
        if summary['p99_latency'] > 2000.0:
            alerts.append({
                'type': 'high_p99_latency',
                'severity': 'high',
                'message': f"P99 latency is {summary['p99_latency']:.2f}ms (threshold: 2000ms)",
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts 