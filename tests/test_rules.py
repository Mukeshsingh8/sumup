"""
Unit tests for rule-based detection functionality.
"""
import os
import sys
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rules import check_rules, _has_any


class TestRuleDetection:
    """Test rule-based escalation detection."""
    
    def test_has_any_function(self):
        """Test the _has_any helper function."""
        patterns = [r"\bhuman\b", r"\bagent\b", "customer service"]
        
        # Positive cases
        assert _has_any(patterns, "I want to speak to a human") == True
        assert _has_any(patterns, "I need an agent") == True
        assert _has_any(patterns, "customer service is terrible") == True
        assert _has_any(patterns, "HUMAN AGENT") == True  # Case insensitive
        
        # Negative cases
        assert _has_any(patterns, "Hello there") == False
        assert _has_any(patterns, "") == False
        assert _has_any(patterns, None) == False
        assert _has_any(patterns, "humanoid") == False  # Word boundary
    
    def test_check_rules_explicit_human_request(self):
        """Test explicit human request detection."""
        policy = {
            'rules': {
                'explicit_human_request': {
                    'enabled': True,
                    'patterns': [r"\b(human|agent|real person)\b"]
                },
                'risk_terms': {
                    'enabled': True,
                    'patterns': []
                },
                'bot_unhelpful_templates': {
                    'enabled': True,
                    'patterns': []
                },
                'frustration_patterns': {
                    'enabled': False,
                    'patterns': []
                }
            }
        }
        
        # Should trigger
        fired = check_rules("I want to speak to a human", "Bot response", policy)
        assert "explicit_human_request" in fired
        
        fired = check_rules("I need an agent", "Bot response", policy)
        assert "explicit_human_request" in fired
        
        # Should not trigger
        fired = check_rules("Hello there", "Bot response", policy)
        assert "explicit_human_request" not in fired
    
    def test_check_rules_risk_terms(self):
        """Test risk term detection."""
        policy = {
            'rules': {
                'explicit_human_request': {
                    'enabled': True,
                    'patterns': []
                },
                'risk_terms': {
                    'enabled': True,
                    'patterns': ['kyc', 'chargeback', 'legal']
                },
                'bot_unhelpful_templates': {
                    'enabled': True,
                    'patterns': []
                },
                'frustration_patterns': {
                    'enabled': False,
                    'patterns': []
                }
            }
        }
        
        # Should trigger
        fired = check_rules("My account is blocked due to KYC", "Bot response", policy)
        assert "risk_terms" in fired
        
        fired = check_rules("I have a chargeback issue", "Bot response", policy)
        assert "risk_terms" in fired
        
        # Should not trigger
        fired = check_rules("Hello there", "Bot response", policy)
        assert "risk_terms" not in fired
    
    def test_check_rules_bot_unhelpful(self):
        """Test bot unhelpfulness detection."""
        policy = {
            'rules': {
                'explicit_human_request': {
                    'enabled': True,
                    'patterns': []
                },
                'risk_terms': {
                    'enabled': True,
                    'patterns': []
                },
                'bot_unhelpful_templates': {
                    'enabled': True,
                    'patterns': ['could you provide more details', 'we could not find']
                },
                'frustration_patterns': {
                    'enabled': False,
                    'patterns': []
                }
            }
        }
        
        # Should trigger
        fired = check_rules("User message", "Could you provide more details?", policy)
        assert "bot_unhelpful_template_seen" in fired
        
        fired = check_rules("User message", "We could not find the information", policy)
        assert "bot_unhelpful_template_seen" in fired
        
        # Should not trigger
        fired = check_rules("User message", "Here's the information you need", policy)
        assert "bot_unhelpful_template_seen" not in fired
    
    def test_check_rules_disabled(self):
        """Test disabled rules."""
        policy = {
            'rules': {
                'explicit_human_request': {
                    'enabled': False,
                    'patterns': [r"\bhuman\b"]
                },
                'risk_terms': {
                    'enabled': True,
                    'patterns': []
                },
                'bot_unhelpful_templates': {
                    'enabled': True,
                    'patterns': []
                },
                'frustration_patterns': {
                    'enabled': False,
                    'patterns': []
                }
            }
        }
        
        fired = check_rules("I want to speak to a human", "Bot response", policy)
        assert "explicit_human_request" not in fired
    
    def test_check_rules_multiple_triggers(self):
        """Test multiple rule triggers."""
        policy = {
            'rules': {
                'explicit_human_request': {
                    'enabled': True,
                    'patterns': [r"\bhuman\b"]
                },
                'risk_terms': {
                    'enabled': True,
                    'patterns': ['kyc']
                },
                'bot_unhelpful_templates': {
                    'enabled': True,
                    'patterns': []
                },
                'frustration_patterns': {
                    'enabled': False,
                    'patterns': []
                }
            }
        }
        
        fired = check_rules("I need a human for KYC issues", "Bot response", policy)
        assert "explicit_human_request" in fired
        assert "risk_terms" in fired
        assert len(fired) == 2
    
    def test_check_rules_empty_policy(self):
        """Test with empty policy."""
        policy = {
            'rules': {
                'explicit_human_request': {
                    'enabled': True,
                    'patterns': []
                },
                'risk_terms': {
                    'enabled': True,
                    'patterns': []
                },
                'bot_unhelpful_templates': {
                    'enabled': True,
                    'patterns': []
                },
                'frustration_patterns': {
                    'enabled': False,
                    'patterns': []
                }
            }
        }
        fired = check_rules("I want a human", "Bot response", policy)
        assert fired == []
    
    def test_check_rules_missing_patterns(self):
        """Test with missing patterns."""
        policy = {
            'rules': {
                'explicit_human_request': {
                    'enabled': True
                    # Missing patterns
                },
                'risk_terms': {
                    'enabled': True,
                    'patterns': []
                },
                'bot_unhelpful_templates': {
                    'enabled': True,
                    'patterns': []
                },
                'frustration_patterns': {
                    'enabled': False,
                    'patterns': []
                }
            }
        }
        
        fired = check_rules("I want a human", "Bot response", policy)
        assert fired == []


if __name__ == '__main__':
    pytest.main([__file__])
