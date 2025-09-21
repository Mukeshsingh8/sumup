#!/usr/bin/env python3
"""
AI-Powered Customer Support Chatbot with Escalation Detection

Uses Google Gemini to provide intelligent customer support responses and
detect when conversations should be escalated to human agents.
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

# Try to import Gemini, fail if not available
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Try to import Redis for caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    role: str  # 'user' or 'bot'
    message: str
    timestamp: float
    turn_id: int

@dataclass
class ChatbotResponse:
    """Response from the AI chatbot."""
    message: str
    should_escalate: bool
    escalation_reason: Optional[str]
    confidence: float
    cached: bool = False

class CustomerSupportChatbot:
    """
    AI-powered customer support chatbot using Google Gemini.
    
    This chatbot provides intelligent responses to customer queries and
    automatically detects when conversations should be escalated to humans.
    """
    
    def __init__(self, api_key: Optional[str] = None, redis_url: str = "redis://localhost:6379"):
        """
        Initialize the customer support chatbot.
        
        Args:
            api_key: Google Gemini API key (if None, uses environment variable)
            redis_url: Redis URL for caching responses
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.redis_url = redis_url
        self.conversation_history: List[ConversationTurn] = []
        
        # Initialize Gemini
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI not available. Install with: pip install google-generativeai")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set. Please set your Google Gemini API key.")
        
        genai.configure(api_key=self.api_key)
        # Use the correct model name for the current API version
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize Redis for caching
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()  # Test connection
                logger.info("Redis connection established for caching")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.redis_client = None
        else:
            logger.warning("Redis not available. Install with: pip install redis")
    
    def add_turn(self, role: str, message: str) -> None:
        """Add a conversation turn to the history."""
        turn = ConversationTurn(
            role=role,
            message=message,
            timestamp=time.time(),
            turn_id=len(self.conversation_history) + 1
        )
        self.conversation_history.append(turn)
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
    
    def _get_cache_key(self, user_message: str, context: str) -> str:
        """Generate cache key for user message and context."""
        content = f"{user_message}|{context}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[ChatbotResponse]:
        """Get cached response if available."""
        if not self.redis_client:
            return None
        
        try:
            cached = self.redis_client.get(f"chatbot:{cache_key}")
            if cached:
                data = json.loads(cached)
                return ChatbotResponse(
                    message=data["message"],
                    should_escalate=data["should_escalate"],
                    escalation_reason=data.get("escalation_reason"),
                    confidence=data["confidence"],
                    cached=True
                )
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _cache_response(self, cache_key: str, response: ChatbotResponse) -> None:
        """Cache the response for future use."""
        if not self.redis_client:
            return
        
        try:
            data = {
                "message": response.message,
                "should_escalate": response.should_escalate,
                "escalation_reason": response.escalation_reason,
                "confidence": response.confidence
            }
            # Cache for 1 hour
            self.redis_client.setex(f"chatbot:{cache_key}", 3600, json.dumps(data))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _get_conversation_context(self, max_turns: int = 10) -> str:
        """Get formatted conversation context for AI analysis."""
        recent_turns = self.conversation_history[-max_turns:] if len(self.conversation_history) > max_turns else self.conversation_history
        
        context = "CONVERSATION HISTORY:\n"
        for turn in recent_turns:
            role_emoji = "👤" if turn.role == "user" else "🤖"
            context += f"{role_emoji} {turn.role.upper()}: {turn.message}\n"
        
        return context
    
    def respond_to_customer(self, user_message: str) -> ChatbotResponse:
        """
        Generate a customer support response and check for escalation.
        
        Args:
            user_message: The customer's message
            
        Returns:
            ChatbotResponse with the bot's response and escalation decision
        """
        # Add user message to history
        self.add_turn("user", user_message)
        
        # Get conversation context
        context = self._get_conversation_context()
        
        # Check cache first
        cache_key = self._get_cache_key(user_message, context)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            logger.info("Using cached response")
            return cached_response
        
        try:
            # Create the prompt for Gemini
            prompt = f"""
You are a professional customer support agent for SumUp, a financial technology company that provides payment solutions for small businesses.

{context}

CURRENT CUSTOMER MESSAGE: {user_message}

Your task:
1. Provide a helpful, professional response to the customer
2. Determine if this conversation should be escalated to a human agent
3. Be empathetic and solution-oriented

ESCALATION CRITERIA - Escalate to human if:
- Customer explicitly requests to speak to a human/agent/manager
- Customer is extremely frustrated, angry, or threatening
- Complex technical issues that require human expertise
- Account security concerns, fraud, or legal issues
- Payment disputes, chargebacks, or financial problems
- KYC/verification issues or account blocks
- Customer has been going in circles with bot responses

RESPONSE GUIDELINES:
- Be helpful, professional, and empathetic
- Provide clear, actionable solutions when possible
- Acknowledge customer concerns
- Use SumUp's tone: friendly, professional, solution-focused
- Keep responses concise but complete

IMPORTANT: You must respond with ONLY a valid JSON object. No other text before or after.

Example response format:
{{
    "response": "Hello! I'd be happy to help you with your account. What specific issue are you experiencing?",
    "should_escalate": false,
    "escalation_reason": null,
    "confidence": 0.8
}}

Now respond with your JSON:
"""

            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Clean the response text and try to parse JSON
            response_text = response.text.strip()
            
            # Try to extract JSON from the response if it's wrapped in markdown
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove ```
            response_text = response_text.strip()
            
            result = json.loads(response_text)
            
            chatbot_response = ChatbotResponse(
                message=result["response"],
                should_escalate=result["should_escalate"],
                escalation_reason=result.get("escalation_reason"),
                confidence=result["confidence"],
                cached=False
            )
            
            # Cache the response
            self._cache_response(cache_key, chatbot_response)
            
            # Add bot response to history
            self.add_turn("bot", chatbot_response.message)
            
            return chatbot_response
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            # Fallback response
            fallback_response = ChatbotResponse(
                message="I apologize, but I'm experiencing technical difficulties. Let me connect you with a human agent who can help you better.",
                should_escalate=True,
                escalation_reason="Technical error in AI response",
                confidence=1.0,
                cached=False
            )
            self.add_turn("bot", fallback_response.message)
            return fallback_response
    
    def generate_response_only(self, user_message: str) -> str:
        """
        Generate a response to a customer message WITHOUT escalation detection.
        This is used in ML mode where the ML model handles escalation decisions.
        
        Args:
            user_message: The customer's message
            
        Returns:
            Just the response message as a string
        """
        # Add user message to history
        self.add_turn("user", user_message)
        
        # Get conversation context
        context = self._get_conversation_context()
        
        # Check cache first
        cache_key = f"response_only:{hash(user_message + context)}"
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    logger.info(f"Cache hit for response-only: {user_message[:50]}...")
                    return cached
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        try:
            # Create the prompt for Gemini - RESPONSE ONLY, NO ESCALATION
            prompt = f"""
You are a professional customer support agent for SumUp, a financial technology company that provides payment solutions for small businesses.

{context}

CURRENT CUSTOMER MESSAGE: {user_message}

Your task:
Provide a helpful, professional response to the customer. Focus ONLY on being helpful and solution-oriented.

RESPONSE GUIDELINES:
- Be helpful, professional, and empathetic
- Provide clear, actionable solutions when possible
- Acknowledge customer concerns
- Use SumUp's tone: friendly, professional, solution-focused
- Keep responses concise but complete
- Do NOT make any escalation decisions - that's handled by another system

IMPORTANT: Respond with ONLY the response message. No JSON, no escalation decisions, just the response text.

Example:
Customer: "I need help with my account"
Response: "Hi there! I'd be happy to help you with your SumUp account. What specific issue are you experiencing?"

Now provide your response:
"""

            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Clean the response text
            response_text = response.text.strip()
            
            # Remove any markdown formatting if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            
            # Cache the response
            if self.redis_client:
                try:
                    self.redis_client.setex(cache_key, 3600, response_text)
                except Exception as e:
                    logger.warning(f"Cache storage failed: {e}")
            
            # Add bot response to history
            self.add_turn("bot", response_text)
            
            return response_text
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or contact our support team directly."
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation."""
        if not self.conversation_history:
            return {
                "total_turns": 0, 
                "user_turns": 0, 
                "bot_turns": 0, 
                "duration": 0,
                "redis_available": self.redis_client is not None,
                "model": "gemini-pro"
            }
        
        user_turns = len([t for t in self.conversation_history if t.role == "user"])
        bot_turns = len([t for t in self.conversation_history if t.role == "bot"])
        duration = self.conversation_history[-1].timestamp - self.conversation_history[0].timestamp if len(self.conversation_history) > 1 else 0
        
        return {
            "total_turns": len(self.conversation_history),
            "user_turns": user_turns,
            "bot_turns": bot_turns,
            "duration": duration,
            "redis_available": self.redis_client is not None,
            "model": "gemini-1.5-flash"
        }

def create_customer_support_chatbot(api_key: Optional[str] = None, redis_url: str = "redis://localhost:6379") -> CustomerSupportChatbot:
    """Factory function to create a customer support chatbot."""
    return CustomerSupportChatbot(api_key=api_key, redis_url=redis_url)