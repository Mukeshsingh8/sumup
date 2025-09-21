"""
Unit tests for model loading and prediction functionality.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import load_artifacts, predict_proba


class TestModelLoading:
    """Test model loading functionality."""
    
    def test_load_artifacts_success(self):
        """Test successful loading of all artifacts."""
        def mock_open_side_effect(filename, mode='r', **kwargs):
            if 'feature_order.json' in filename:
                return mock_open(read_data='["feature1", "feature2"]')()
            elif 'version.txt' in filename:
                return mock_open(read_data="model=logreg@123\nthreshold=0.081\n")()
            else:
                return mock_open(read_data='{"feature1": 1, "feature2": 2}')()
        
        with patch('joblib.load') as mock_load, \
             patch('builtins.open', side_effect=mock_open_side_effect), \
             patch('os.path.exists', return_value=True), \
             patch('yaml.safe_load', return_value={'rules': {'test': True}}):
            
            model, features, tau, policy = load_artifacts('test_artifacts')
            
            assert tau == 0.081
            assert features == ["feature1", "feature2"]
            assert policy['rules']['test'] is True
            mock_load.assert_called_once()
    
    def test_load_artifacts_missing_files(self):
        """Test handling of missing artifact files."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                load_artifacts('nonexistent_artifacts')
    
    def test_load_artifacts_fallback_policy(self):
        """Test fallback to default policy when YAML loading fails."""
        def mock_open_side_effect(filename, mode='r', **kwargs):
            if 'feature_order.json' in filename:
                return mock_open(read_data='["feature1"]')()
            elif 'version.txt' in filename:
                return mock_open(read_data="threshold=0.5\n")()
            else:
                return mock_open(read_data='{"feature1": 1}')()
        
        with patch('joblib.load'), \
             patch('builtins.open', side_effect=mock_open_side_effect), \
             patch('os.path.exists', return_value=True), \
             patch('yaml.safe_load', side_effect=Exception("YAML error")):
            
            model, features, tau, policy = load_artifacts('test_artifacts')
            
            assert tau == 0.5
            assert features == ["feature1"]
            assert policy == {}


class TestModelPrediction:
    """Test model prediction functionality."""
    
    def test_predict_proba(self):
        """Test probability prediction."""
        # Mock model
        mock_model = type('MockModel', (), {})()
        mock_model.predict_proba = lambda X: np.array([[0.2, 0.8]])
        
        # Test data
        X = pd.DataFrame([[1, 2, 3]], columns=['f1', 'f2', 'f3'])
        
        result = predict_proba(mock_model, X)
        assert result == 0.8
    
    def test_predict_proba_single_class(self):
        """Test prediction with single class output."""
        mock_model = type('MockModel', (), {})()
        mock_model.predict_proba = lambda X: np.array([[0.3]])
        
        X = pd.DataFrame([[1, 2]], columns=['f1', 'f2'])
        
        with pytest.raises(IndexError):
            predict_proba(mock_model, X)


if __name__ == '__main__':
    pytest.main([__file__])
