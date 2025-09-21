"""
Logging configuration for the escalation detection system.
"""
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any


class EscalationFormatter(logging.Formatter):
    """Custom formatter for escalation detection logs."""
    
    def format(self, record):
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'conversation_id'):
            log_entry['conversation_id'] = record.conversation_id
        if hasattr(record, 'escalate'):
            log_entry['escalate'] = record.escalate
        if hasattr(record, 'score'):
            log_entry['score'] = record.score
        if hasattr(record, 'latency_ms'):
            log_entry['latency_ms'] = record.latency_ms
        if hasattr(record, 'fired_rules'):
            log_entry['fired_rules'] = record.fired_rules
        
        return json.dumps(log_entry)


def setup_logging(log_level: str = None, log_file: str = None) -> logging.Logger:
    """
    Setup logging configuration for the escalation detection system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Get log level from environment or use default
    level = log_level or os.getenv('LOG_LEVEL', 'INFO')
    
    # Create logger
    logger = logging.getLogger('escalation_detector')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    if os.getenv('LOG_FORMAT', 'json') == 'json':
        formatter = EscalationFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def log_escalation_decision(logger: logging.Logger, 
                          conversation_id: str,
                          escalate: bool,
                          score: float,
                          latency_ms: int,
                          fired_rules: list,
                          reason: str,
                          **kwargs):
    """
    Log escalation decision with structured data.
    
    Args:
        logger: Logger instance
        conversation_id: Conversation identifier
        escalate: Whether escalation was triggered
        score: Model score or rule score
        latency_ms: Processing latency in milliseconds
        fired_rules: List of fired rules
        reason: Human-readable reason for decision
        **kwargs: Additional fields to log
    """
    extra = {
        'conversation_id': conversation_id,
        'escalate': escalate,
        'score': score,
        'latency_ms': latency_ms,
        'fired_rules': fired_rules,
        'reason': reason,
        **kwargs
    }
    
    if escalate:
        logger.warning(f"ESCALATION TRIGGERED: {reason}", extra=extra)
    else:
        logger.info(f"No escalation: {reason}", extra=extra)


def log_model_performance(logger: logging.Logger,
                         model_name: str,
                         roc_auc: float,
                         pr_auc: float,
                         threshold: float,
                         **kwargs):
    """
    Log model performance metrics.
    
    Args:
        logger: Logger instance
        model_name: Name of the model
        roc_auc: ROC-AUC score
        pr_auc: PR-AUC score
        threshold: Decision threshold
        **kwargs: Additional metrics
    """
    extra = {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'threshold': threshold,
        **kwargs
    }
    
    logger.info(f"Model performance: {model_name}", extra=extra)


def log_system_health(logger: logging.Logger,
                     component: str,
                     status: str,
                     details: Dict[str, Any] = None):
    """
    Log system health status.
    
    Args:
        logger: Logger instance
        component: System component name
        status: Health status (healthy, degraded, unhealthy)
        details: Additional health details
    """
    extra = {
        'component': component,
        'status': status,
        'details': details or {}
    }
    
    if status == 'healthy':
        logger.info(f"System health: {component} is {status}", extra=extra)
    elif status == 'degraded':
        logger.warning(f"System health: {component} is {status}", extra=extra)
    else:
        logger.error(f"System health: {component} is {status}", extra=extra)
