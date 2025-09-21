"""
Integration tests for the FastAPI service.
"""
import os
import sys
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import with proper module path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.service import app


class TestService:
    """Test FastAPI service endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_artifacts(self):
        """Mock artifacts for testing."""
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        
        return {
            'model': mock_model,
            'feature_order': ['turn_idx', 'user_caps_ratio', 'exclam_count', 'msg_len', 
                             'bot_unhelpful', 'user_requests_human', 'risk_terms', 
                             'no_progress_count', 'bot_repeat_count'],
            'tau': 0.081,
            'policy': {
                'rules': {
                    'explicit_human_request': {
                        'enabled': True,
                        'patterns': [r"\\bhuman\\b"]
                    }
                }
            }
        }
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "ok" in data
        assert "model_loaded" in data
    
    @patch('src.service.ARTIFACTS')
    def test_score_endpoint_success(self, mock_artifacts, client):
        """Test successful scoring endpoint."""
        mock_artifacts.__getitem__.side_effect = lambda key: {
            'model': MagicMock(),
            'feature_order': ['turn_idx', 'user_caps_ratio'],
            'tau': 0.081,
            'policy': {'rules': {}}
        }[key]
        
        with patch('src.service.decide') as mock_decide:
            mock_decide.return_value = ({
                'conversation_id': 'test_conv',
                'turn_id': None,
                'escalate': True,
                'where': 'model',
                'score': 0.85,
                'threshold': 0.081,
                'fired_rules': [],
                'reason': 'model score >= tau',
                'latency_ms': 10,
                'model_version': 'model.joblib',
                'policy_version': 'policy@assess',
                'state': {'user_turn_idx': 1}
            }, {})
            
            response = client.post("/score", json={
                "conversation_id": "test_conv",
                "role": "user",
                "message": "I need help",
                "prev_bot_text": "Hello"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data['escalate'] == True
            assert data['score'] == 0.85
            assert data['where'] == 'model'
    
    def test_score_endpoint_validation_error(self, client):
        """Test validation error handling."""
        response = client.post("/score", json={
            "conversation_id": "test_conv",
            "role": "invalid_role",  # Invalid role
            "message": "I need help"
        })
        
        assert response.status_code == 422  # Validation error
    
    def test_score_endpoint_missing_fields(self, client):
        """Test missing required fields."""
        response = client.post("/score", json={
            "conversation_id": "test_conv"
            # Missing required fields
        })
        
        assert response.status_code == 422
    
    @patch('src.service.ARTIFACTS')
    def test_score_endpoint_rule_escalation(self, mock_artifacts, client):
        """Test rule-based escalation."""
        mock_artifacts.__getitem__.side_effect = lambda key: {
            'model': MagicMock(),
            'feature_order': ['turn_idx'],
            'tau': 0.081,
            'policy': {
                'rules': {
                    'explicit_human_request': {
                        'enabled': True,
                        'patterns': [r"\\bhuman\\b"]
                    }
                }
            }
        }[key]
        
        with patch('src.service.decide') as mock_decide:
            mock_decide.return_value = ({
                'conversation_id': 'test_conv',
                'turn_id': None,
                'escalate': True,
                'where': 'rules',
                'score': 1.0,
                'threshold': 0.081,
                'fired_rules': ['explicit_human_request'],
                'reason': 'user explicitly requested human',
                'latency_ms': 5,
                'model_version': 'model.joblib',
                'policy_version': 'policy@assess',
                'state': {'user_turn_idx': 1}
            }, {})
            
            response = client.post("/score", json={
                "conversation_id": "test_conv",
                "role": "user",
                "message": "I want to speak to a human",
                "prev_bot_text": "Hello"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data['escalate'] == True
            assert data['where'] == 'rules'
            assert 'explicit_human_request' in data['fired_rules']


if __name__ == '__main__':
    pytest.main([__file__])
