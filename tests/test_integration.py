"""
End-to-end integration tests for the escalation detection system.
"""
import os
import sys
import pytest
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import load_artifacts
from policy import decide
from state import ConvState


class TestIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary artifacts directory for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create mock artifacts
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        
        # Save mock model
        import joblib
        joblib.dump(mock_model, os.path.join(temp_dir, 'model.joblib'))
        
        # Save feature order
        with open(os.path.join(temp_dir, 'feature_order.json'), 'w') as f:
            json.dump(['turn_idx', 'user_caps_ratio', 'exclam_count', 'msg_len', 
                      'bot_unhelpful', 'user_requests_human', 'risk_terms', 
                      'no_progress_count', 'bot_repeat_count'], f)
        
        # Save version info
        with open(os.path.join(temp_dir, 'version.txt'), 'w') as f:
            f.write('model=logreg@123\\nthreshold=0.081\\n')
        
        # Save policy
        policy = {
            'rules': {
                'explicit_human_request': {
                    'enabled': True,
                    'patterns': [r"\\b(human|agent)\\b"]
                },
                'risk_terms': {
                    'enabled': True,
                    'patterns': ['kyc', 'chargeback']
                }
            }
        }
        
        import yaml
        with open(os.path.join(temp_dir, 'policy.yaml'), 'w') as f:
            yaml.dump(policy, f)
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_pipeline_rule_escalation(self, temp_artifacts_dir):
        """Test full pipeline with rule-based escalation."""
        # Load artifacts
        model, feature_order, tau, policy = load_artifacts(temp_artifacts_dir)
        
        # Create artifacts dict
        artifacts = {
            'model': model,
            'feature_order': feature_order,
            'tau': tau,
            'policy': policy
        }
        
        # Test event
        event = {
            'conversation_id': 'test_conv_1',
            'role': 'user',
            'message': 'I want to speak to a human agent',
            'prev_bot_text': 'Hello, how can I help?'
        }
        
        # Initial state
        conv_state = {
            'user_turn_idx': 0,
            'no_progress_count': 0.0,
            'bot_repeat_count': 0.0,
            'prev_bot_text': ''
        }
        
        # Make decision
        decision, new_state = decide(event, conv_state, artifacts)
        
        # Verify rule-based escalation
        assert decision['escalate'] == True
        assert decision['where'] == 'rules'
        assert 'explicit_human_request' in decision['fired_rules']
        assert decision['score'] == 1.0
        assert new_state['user_turn_idx'] == 1
    
    def test_full_pipeline_model_escalation(self, temp_artifacts_dir):
        """Test full pipeline with model-based escalation."""
        # Load artifacts
        model, feature_order, tau, policy = load_artifacts(temp_artifacts_dir)
        
        # Create artifacts dict
        artifacts = {
            'model': model,
            'feature_order': feature_order,
            'tau': tau,
            'policy': policy
        }
        
        # Test event (should trigger model, not rules)
        event = {
            'conversation_id': 'test_conv_2',
            'role': 'user',
            'message': 'This is frustrating! I need help!',
            'prev_bot_text': 'Could you provide more details?'
        }
        
        # Initial state
        conv_state = {
            'user_turn_idx': 1,  # Past guard threshold
            'no_progress_count': 0.0,
            'bot_repeat_count': 0.0,
            'prev_bot_text': ''
        }
        
        # Make decision
        decision, new_state = decide(event, conv_state, artifacts)
        
        # Verify model-based escalation
        assert decision['escalate'] == True
        assert decision['where'] == 'model'
        assert decision['score'] == 0.8  # From mock model
        assert new_state['user_turn_idx'] == 2
    
    def test_full_pipeline_no_escalation(self, temp_artifacts_dir):
        """Test full pipeline with no escalation."""
        # Load artifacts
        model, feature_order, tau, policy = load_artifacts(temp_artifacts_dir)
        
        # Create artifacts dict
        artifacts = {
            'model': model,
            'feature_order': feature_order,
            'tau': tau,
            'policy': policy
        }
        
        # Test event (should not escalate)
        event = {
            'conversation_id': 'test_conv_3',
            'role': 'user',
            'message': 'Hello, I have a question',
            'prev_bot_text': 'Hi! How can I help you?'
        }
        
        # Initial state
        conv_state = {
            'user_turn_idx': 1,
            'no_progress_count': 0.0,
            'bot_repeat_count': 0.0,
            'prev_bot_text': ''
        }
        
        # Make decision
        decision, new_state = decide(event, conv_state, artifacts)
        
        # Verify no escalation
        assert decision['escalate'] == False
        assert decision['where'] == 'model'
        assert decision['score'] == 0.2  # From mock model
        assert new_state['user_turn_idx'] == 2
    
    def test_state_management(self):
        """Test conversation state management."""
        state_manager = ConvState()
        
        # Test saving and loading state
        conv_id = 'test_conv_state'
        test_state = {
            'user_turn_idx': 5,
            'no_progress_count': 2.0,
            'bot_repeat_count': 1.0,
            'prev_bot_text': 'test message'
        }
        
        # Save state
        state_manager.save(conv_id, test_state)
        
        # Load state
        loaded_state = state_manager.load(conv_id)
        
        # Verify state persistence
        assert loaded_state['user_turn_idx'] == 5
        assert loaded_state['no_progress_count'] == 2.0
        assert loaded_state['bot_repeat_count'] == 1.0
        assert loaded_state['prev_bot_text'] == 'test message'
    
    def test_conversation_flow(self, temp_artifacts_dir):
        """Test complete conversation flow."""
        # Load artifacts
        model, feature_order, tau, policy = load_artifacts(temp_artifacts_dir)
        
        artifacts = {
            'model': model,
            'feature_order': feature_order,
            'tau': tau,
            'policy': policy
        }
        
        # Simulate conversation
        conv_state = {
            'user_turn_idx': 0,
            'no_progress_count': 0.0,
            'bot_repeat_count': 0.0,
            'prev_bot_text': ''
        }
        
        # Turn 1: Bot greeting
        bot_event = {
            'conversation_id': 'conv_flow',
            'role': 'bot',
            'message': 'Hello! How can I help you?'
        }
        decision, conv_state = decide(bot_event, conv_state, artifacts)
        assert decision['escalate'] == False  # Bot messages don't escalate
        
        # Turn 2: User question
        user_event = {
            'conversation_id': 'conv_flow',
            'role': 'user',
            'message': 'I need help with my account'
        }
        decision, conv_state = decide(user_event, conv_state, artifacts)
        assert decision['escalate'] == False  # Normal question
        assert conv_state['user_turn_idx'] == 1
        
        # Turn 3: Bot unhelpful response
        bot_event = {
            'conversation_id': 'conv_flow',
            'role': 'bot',
            'message': 'Could you provide more details?'
        }
        decision, conv_state = decide(bot_event, conv_state, artifacts)
        assert decision['escalate'] == False  # Bot messages don't escalate
        
        # Turn 4: User frustrated
        user_event = {
            'conversation_id': 'conv_flow',
            'role': 'user',
            'message': 'I already told you! I want to speak to a human!'
        }
        decision, conv_state = decide(user_event, conv_state, artifacts)
        assert decision['escalate'] == True  # Should escalate
        assert 'explicit_human_request' in decision['fired_rules']


if __name__ == '__main__':
    pytest.main([__file__])
