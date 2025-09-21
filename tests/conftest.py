"""
Pytest configuration and shared fixtures.
"""
import os
import sys
import pytest
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_conversation():
    """Sample conversation data for testing."""
    return {
        "conversation_id": "test_conv",
        "conversation_history": [
            {"role": "bot", "message": "Hello! How can I help you?"},
            {"role": "user", "message": "I need help with my account"},
            {"role": "bot", "message": "Could you provide more details?"},
            {"role": "user", "message": "I want to speak to a human agent"}
        ],
        "is_escalation_needed": True,
        "reasoning": "User explicitly requested human agent"
    }


@pytest.fixture
def sample_policy():
    """Sample policy configuration for testing."""
    return {
        "version": "policy@test",
        "thresholds": {"tau_low": 0.45, "tau_high": 0.70},
        "guards": {"min_turn_before_model": 1},
        "rules": {
            "explicit_human_request": {
                "enabled": True,
                "patterns": [r"\\b(human|agent)\\b"]
            },
            "risk_terms": {
                "enabled": True,
                "patterns": ["kyc", "chargeback", "legal"]
            },
            "bot_unhelpful_templates": {
                "enabled": True,
                "patterns": ["could you provide more details"]
            }
        }
    }


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    from unittest.mock import MagicMock
    model = MagicMock()
    model.predict_proba.return_value = [[0.2, 0.8]]
    return model
