"""
Unit tests for feature engineering functionality.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from features import featurize_one, _has_any, _caps_ratio


class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    def test_has_any_patterns(self):
        """Test pattern matching functionality."""
        patterns = [r"\\bhuman\\b", r"\\bagent\\b"]
        
        assert _has_any(patterns, "I want to speak to a human") == 1
        assert _has_any(patterns, "I need an agent") == 1
        assert _has_any(patterns, "Hello there") == 0
        assert _has_any(patterns, "") == 0
        assert _has_any(patterns, None) == 0
    
    def test_caps_ratio(self):
        """Test capitalization ratio calculation."""
        assert _caps_ratio("HELLO WORLD") == 1.0
        assert _caps_ratio("hello world") == 0.0
        assert _caps_ratio("Hello World") == 0.5
        assert _caps_ratio("") == 0.0
        assert _caps_ratio("123!@#") == 0.0
        assert _caps_ratio("H") == 1.0
    
    def test_featurize_one_basic(self):
        """Test basic feature extraction."""
        policy = {
            'rules': {
                'explicit_human_request': {'patterns': [r"\\bhuman\\b"]},
                'risk_terms': {'patterns': ['kyc']},
                'bot_unhelpful_templates': {'patterns': ['could you provide']}
            }
        }
        feature_order = ['turn_idx', 'user_caps_ratio', 'exclam_count', 'msg_len', 
                        'bot_unhelpful', 'user_requests_human', 'risk_terms', 
                        'no_progress_count', 'bot_repeat_count']
        
        conv_state = {'no_progress_count': 0.0, 'bot_repeat_count': 0.0, 'prev_bot_text': ''}
        
        X, new_state = featurize_one(
            user_turn_idx=1,
            user_text="I need a HUMAN agent!",
            prev_bot_text="Could you provide more details?",
            conv_state=conv_state,
            policy=policy,
            feature_order=feature_order
        )
        
        # Check feature values
        assert X.iloc[0]['turn_idx'] == 1.0
        assert X.iloc[0]['user_caps_ratio'] > 0.0  # Has caps
        assert X.iloc[0]['exclam_count'] == 1.0  # Has exclamation
        assert X.iloc[0]['msg_len'] == len("I need a HUMAN agent!")
        assert X.iloc[0]['bot_unhelpful'] == 1.0  # Bot text matches pattern
        assert X.iloc[0]['user_requests_human'] == 1.0  # User text matches pattern
        assert X.iloc[0]['risk_terms'] == 0.0  # No risk terms
        assert X.iloc[0]['no_progress_count'] == 1.0  # Incremented
        assert X.iloc[0]['bot_repeat_count'] == 0.0  # No repeat
        
        # Check state update
        assert new_state['no_progress_count'] == 1.0
        assert new_state['prev_bot_text'] == "could you provide more details?"
    
    def test_featurize_one_state_tracking(self):
        """Test conversation state tracking."""
        policy = {'rules': {}}
        feature_order = ['turn_idx', 'user_caps_ratio', 'exclam_count', 'msg_len', 
                        'bot_unhelpful', 'user_requests_human', 'risk_terms', 
                        'no_progress_count', 'bot_repeat_count']
        
        conv_state = {'no_progress_count': 0.0, 'bot_repeat_count': 0.0, 'prev_bot_text': ''}
        
        # First message
        X1, state1 = featurize_one(1, "Hello", "Hi there", conv_state, policy, feature_order)
        
        # Second message with same bot text (should increment repeat count)
        X2, state2 = featurize_one(2, "Help me", "Hi there", state1, policy, feature_order)
        
        assert state2['bot_repeat_count'] == 1.0
        assert X2.iloc[0]['bot_repeat_count'] == 1.0
    
    def test_featurize_one_empty_inputs(self):
        """Test feature extraction with empty inputs."""
        policy = {'rules': {}}
        feature_order = ['turn_idx', 'user_caps_ratio', 'exclam_count', 'msg_len', 
                        'bot_unhelpful', 'user_requests_human', 'risk_terms', 
                        'no_progress_count', 'bot_repeat_count']
        
        conv_state = {'no_progress_count': 0.0, 'bot_repeat_count': 0.0, 'prev_bot_text': ''}
        
        X, new_state = featurize_one(0, "", "", conv_state, policy, feature_order)
        
        assert X.iloc[0]['turn_idx'] == 0.0
        assert X.iloc[0]['user_caps_ratio'] == 0.0
        assert X.iloc[0]['exclam_count'] == 0.0
        assert X.iloc[0]['msg_len'] == 0.0
        assert X.iloc[0]['bot_unhelpful'] == 0.0
        assert X.iloc[0]['user_requests_human'] == 0.0
        assert X.iloc[0]['risk_terms'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__])
