# 🚀 SumUp Escalation Detection System

A production-ready hybrid system that combines machine learning and AI to automatically detect when customer support conversations need to be escalated to human agents, while providing intelligent responses.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🎯 Overview

This system offers two powerful modes for customer support escalation detection:

### 🤖 AI Mode (Pure AI)
- **AI handles both**: response generation + escalation detection
- **Google Gemini integration** for intelligent conversations
- **Context-aware responses** with conversation history
- **Automatic escalation detection** using AI reasoning

### 📊 ML Mode (Hybrid)
- **ML model handles**: escalation detection (decision only)
- **AI handles**: response generation (response only)
- **Best of both worlds**: ML reliability + AI intelligence
- **Separate concerns**: clear separation between decision and response

### Key Capabilities

- **Dual-mode operation** with AI and ML approaches
- **Real-time escalation detection** during live conversations
- **Intelligent response generation** using Google Gemini
- **Rule-based fallbacks** for critical scenarios
- **Conversation state tracking** with rolling metrics
- **Production-ready API** with FastAPI
- **Comprehensive CLI** for testing and simulation
- **Redis caching** for faster AI responses
- **MLflow integration** for experiment tracking

## ✨ Features

### 🤖 AI Integration
- **Google Gemini 2.5 Flash**: State-of-the-art language model
- **Context-aware responses**: Maintains conversation history
- **Intelligent escalation**: AI-powered escalation detection
- **Redis caching**: Faster response times with intelligent caching
- **Professional tone**: SumUp customer support style

### 📊 Machine Learning
- **Multiple algorithms**: Logistic Regression, Random Forest, XGBoost
- **Feature engineering**: 9 engineered features including conversation dynamics
- **Model calibration**: Sigmoid calibration for better probability estimates
- **Cross-validation**: Group-aware CV to prevent data leakage
- **Performance metrics**: ROC-AUC, PR-AUC, early escalation detection

### 🛡️ Rule-Based Safety
- **Explicit human requests**: Immediate escalation for "speak to human" patterns
- **Risk term detection**: Escalation for legal/security terms (KYC, chargeback, etc.)
- **Bot unhelpfulness**: Detection of repetitive or unhelpful bot responses

### 🏗️ Production Features
- **Dual-mode CLI**: Choose between AI and ML approaches
- **FastAPI service**: RESTful API with automatic documentation
- **State management**: Redis with in-memory fallback
- **PII redaction**: Automatic redaction of sensitive information
- **Health checks**: System monitoring and status endpoints
- **Configurable policies**: YAML-based configuration management

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda
- Google Gemini API key (for AI mode)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd sumup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key (for AI Mode)
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Get your API key from: https://makersuite.google.com/app/apikey
```

### 3. Test the System
```bash
# Test CLI interface (choose between AI and ML modes)
python cli/main.py

# Start API server (automatically uses notebooks/artifacts/)
python -m src.service
```

**Note**: The system automatically uses `notebooks/artifacts/` as the default artifacts directory. No manual setup required!

### 4. Example CLI Usage

**AI Mode (Pure AI):**
```bash
$ python cli/main.py
🎯 ESCALATION DETECTION MODE SELECTION
Choose your escalation detection method:
1. 🤖 AI Customer Support Chatbot
2. 📊 ML Model Detection
Enter your choice (1 or 2): 1

🤖 AI Customer Support Chatbot Mode
> hi
🤖 Bot: Hi there! Thanks for contacting SumUp support. How can I help you today?
✅ NO ESCALATION (AI: Normal conversation)
   Confidence: 0.90

> I want to speak to a human agent
🤖 Bot: Hi there! I understand you'd like to speak with a human agent...
🚨 ESCALATE ✅ (AI: Customer explicitly requested to speak with a human agent)
   Confidence: 1.00
```

**ML Mode (Hybrid):**
```bash
$ python cli/main.py
Enter your choice (1 or 2): 2

📊 ML Model Detection Mode
> hi
🤖 Bot: Hi there! How can I help you today?
✅ NO ESCALATION (ML: model score < tau)
   ML Score: 0.125 | Threshold: 0.300

> I want to speak to a human agent
🤖 Bot: Hi there! I understand you'd like to speak with a human agent...
🚨 ESCALATE ✅ (ML: user explicitly requested human)
   ML Score: 1.000 | Threshold: 0.300
   Where: rules | Rules: ['explicit_human_request']
```

## 📦 Installation

### Option 1: Using pip
```bash
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
conda create -n sumup python=3.11
conda activate sumup
pip install -r requirements.txt
```

### Dependencies
- **Core ML**: scikit-learn, pandas, numpy
- **AI Integration**: google-generativeai, python-dotenv
- **API**: fastapi, uvicorn
- **Storage**: redis (optional), pyarrow
- **MLOps**: mlflow, xgboost
- **Utilities**: pyyaml, joblib

## 💻 Usage

### CLI Interface

The CLI provides an interactive way to test both AI and ML escalation systems:

```bash
python cli/main.py [--artifacts notebooks/artifacts/] [--mode ai|ml] [--verbose]
```

**Mode Selection:**
- **AI Mode**: Pure AI chatbot with intelligent responses and escalation detection
- **ML Mode**: Hybrid system with ML escalation decisions and AI responses

**Commands:**
- `help` - Show help message
- `examples` - Show example conversations
- `stats` - Show conversation statistics
- `reset` - Reset conversation state
- `quit/exit` - Exit the program

**Example Session (AI Mode):**
```bash
$ python cli/main.py --mode ai
🤖 AI Customer Support Chatbot Mode
> hi
🤖 Bot: Hi there! Thanks for contacting SumUp support. How can I help you today?
✅ NO ESCALATION (AI: Normal conversation)
   Confidence: 0.90

> I need help with my payment
🤖 Bot: I'd be happy to help you with your payment issue. Could you please provide more details about what's happening?
✅ NO ESCALATION (AI: Normal conversation)
   Confidence: 0.85

> I want to speak to a human agent
🤖 Bot: I understand you'd like to speak with a human agent. Let me connect you with our support team.
🚨 ESCALATE ✅ (AI: Customer explicitly requested to speak with a human agent)
   Confidence: 1.00
```

**Example Session (ML Mode):**
```bash
$ python cli/main.py --mode ml
📊 ML Model Detection Mode
> hi
🤖 Bot: Hi there! How can I help you today?
✅ NO ESCALATION (ML: model score < tau)
   ML Score: 0.125 | Threshold: 0.300

> I want to speak to a human agent
🤖 Bot: I understand you'd like to speak with a human agent. How can I help you?
🚨 ESCALATE ✅ (ML: user explicitly requested human)
   ML Score: 1.000 | Threshold: 0.300
   Where: rules | Rules: ['explicit_human_request']
```

### API Service

Start the FastAPI service:

```bash
python -m src.service
```

The API will be available at `http://localhost:8080`

**Health Check:**
```bash
curl http://localhost:8080/health
```

**Score a Message:**
```bash
curl -X POST "http://localhost:8080/score" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "test_conv_1",
    "role": "user",
    "message": "I need to speak to a human agent",
    "prev_bot_text": "Could you provide more details?"
  }'
```

## 📚 API Documentation

### Endpoints

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "ok": true,
  "model_loaded": true
}
```

#### `POST /score`
Score a conversation turn for escalation.

**Request:**
```json
{
  "conversation_id": "string",
  "turn_id": "string (optional)",
  "role": "user|bot",
  "message": "string",
  "prev_bot_text": "string",
  "ts": "string (optional)",
  "lang": "string (optional, default: en)"
}
```

**Response:**
```json
{
  "conversation_id": "string",
  "turn_id": "string",
  "escalate": true,
  "where": "rules|model|guard",
  "score": 0.85,
  "threshold": 0.081,
  "fired_rules": ["explicit_human_request"],
  "reason": "user explicitly requested human",
  "latency_ms": 12,
  "model_version": "model.joblib",
  "policy_version": "policy@assess",
  "state": {
    "user_turn_idx": 3,
    "no_progress_count": 1.0,
    "bot_repeat_count": 0.0
  }
}
```

### Interactive API Documentation

Visit `http://localhost:8080/docs` for Swagger UI documentation.

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Client    │    │   API Service   │    │   Web Client    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     Mode Selection        │
                    │  ┌─────────────────────┐  │
                    │  │   AI Mode           │  │
                    │  │   ML Mode           │  │
                    │  └─────────────────────┘  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     AI Integration        │
                    │  ┌─────────────────────┐  │
                    │  │   Google Gemini     │  │
                    │  │   - Response Gen    │  │
                    │  │   - Escalation      │  │
                    │  │   - Context Aware   │  │
                    │  └─────────────────────┘  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │     ML Engine             │
                    │  ┌─────────────────────┐  │
                    │  │   Rule Engine       │  │
                    │  │  - Human requests   │  │
                    │  │  - Risk terms       │  │
                    │  │  - Bot unhelpful    │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │   ML Model          │  │
                    │  │  - Feature eng.     │  │
                    │  │  - Probability      │  │
                    │  │  - Threshold        │  │
                    │  └─────────────────────┘  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    State Management       │
                    │  ┌─────────────────────┐  │
                    │  │   Redis (primary)   │  │
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │   In-Memory (fallback)│  │
                    │  └─────────────────────┘  │
                    └───────────────────────────┘
```

### Data Flow

#### AI Mode (Pure AI)
1. **Input**: User message + conversation context
2. **AI Processing**: Google Gemini analyzes message and context
3. **AI Decision**: AI determines escalation and generates response
4. **State Update**: Update conversation state
5. **Response**: Return AI response + escalation decision

#### ML Mode (Hybrid)
1. **Input**: User message + conversation context
2. **Rule Check**: Immediate escalation for critical patterns
3. **Feature Engineering**: Extract 9 conversation features
4. **ML Prediction**: Get escalation probability
5. **ML Decision**: Compare probability to threshold
6. **AI Response**: Generate response using AI (no escalation logic)
7. **State Update**: Update conversation state
8. **Response**: Return ML escalation decision + AI response

### Feature Engineering

The system uses 9 engineered features:

| Feature | Description | Type |
|---------|-------------|------|
| `turn_idx` | User turn index in conversation | Numeric |
| `user_caps_ratio` | Ratio of capital letters in user message | Numeric |
| `exclam_count` | Number of exclamation marks | Numeric |
| `msg_len` | Length of user message | Numeric |
| `bot_unhelpful` | Bot response contains unhelpful patterns | Binary |
| `user_requests_human` | User explicitly requests human | Binary |
| `risk_terms` | User message contains risk terms | Binary |
| `no_progress_count` | Rolling count of unhelpful bot responses | Numeric |
| `bot_repeat_count` | Rolling count of repeated bot responses | Numeric |

## 📊 Model Performance

### Training Results

| Model | ROC-AUC | PR-AUC | Threshold | Early Escalation | Premature Rate |
|-------|---------|--------|-----------|------------------|----------------|
| Logistic Regression | 0.633 | 0.461 | 0.081 | 100% | 90.9% |
| Random Forest | 0.705 | 0.416 | 0.094 | 100% | 90.9% |
| XGBoost | 0.463 | 0.259 | 0.107 | 100% | 90.9% |

### Test Set Performance
- **ROC-AUC**: 0.950
- **PR-AUC**: 0.833
- **Early Escalation Rate**: 100%
- **Premature Escalation Rate**: 100%
- **Average Time to Escalation**: -1.5 turns (early detection)

### Dataset Statistics
- **Total Conversations**: 20
- **Total User Turns**: 61
- **Positive Examples**: 12 (19.7%)
- **Negative Examples**: 49 (80.3%)

## 🤖 AI Integration

### Google Gemini Integration

The system integrates with Google Gemini 1.5 Flash for intelligent conversation handling:

#### AI Mode Features
- **Context-aware responses**: Maintains conversation history
- **Intelligent escalation**: AI-powered escalation detection
- **Professional tone**: SumUp customer support style
- **Redis caching**: Faster response times
- **Error handling**: Graceful fallbacks

#### ML Mode Features
- **Response-only generation**: AI focuses purely on responses
- **No escalation logic**: ML model handles all escalation decisions
- **Separate prompts**: Different prompts for different modes
- **Caching support**: Redis caching for performance

### API Key Setup

1. **Get API Key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **Create .env file**:
   ```bash
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```
3. **Test AI Mode**:
   ```bash
   python cli/main.py --mode ai
   ```

### AI Prompt Engineering

#### AI Mode Prompt (Full Functionality)
```
You are a professional customer support agent for SumUp...

Your task:
1. Provide a helpful, professional response to the customer
2. Determine if this conversation should be escalated to a human agent
3. Be empathetic and solution-oriented

ESCALATION CRITERIA - Escalate to human if:
- Customer explicitly requests to speak to a human/agent/manager
- Customer is extremely frustrated, angry, or threatening
...
```

#### ML Mode Prompt (Response Only)
```
You are a professional customer support agent for SumUp...

Your task:
Provide a helpful, professional response to the customer. Focus ONLY on being helpful and solution-oriented.

RESPONSE GUIDELINES:
- Be helpful, professional, and empathetic
- Do NOT make any escalation decisions - that's handled by another system
...
```

## 🛠️ Development

### Project Structure
```
sumup/
├── src/                    # Source code
│   ├── service.py         # FastAPI service
│   ├── model.py           # Model loading utilities
│   ├── features.py        # Feature engineering
│   ├── policy.py          # Decision logic
│   ├── state.py           # State management
│   ├── rules.py           # Rule-based detection
│   ├── ai_detector.py     # AI integration (Google Gemini)
│   └── logging_config.py  # Logging configuration
├── cli/                   # Command line interface
│   └── main.py           # CLI implementation with dual modes
├── notebooks/             # Jupyter notebooks
│   └── escalation_detector.ipynb  # Training notebook
├── notebooks/artifacts/  # Trained models and configs
│   ├── model.joblib      # Trained model
│   ├── feature_order.json # Feature ordering
│   ├── policy.yaml       # Policy configuration
│   └── version.txt       # Model metadata
├── data/                  # Training data
│   └── escalation_dataset.json
├── tests/                 # Test suite
├── requirements.txt       # Dependencies
├── policy.yaml           # Main policy config
├── .env                  # Environment variables (GEMINI_API_KEY)
└── README.md             # This file
```

### Configuration

The system uses YAML configuration files:

**policy.yaml:**
```yaml
version: "policy@assess"
thresholds:
  tau_low: 0.45
  tau_high: 0.70
guards:
  min_turn_before_model: 1
rules:
  explicit_human_request:
    enabled: true
    patterns:
      - "\\b(human|agent|real person|talk to (?:a )?human|speak to (?:a )?human|customer service|support agent)\\b"
  risk_terms:
    enabled: true
    patterns: ["kyc","blocked","chargeback","legal","id verification"]
  bot_unhelpful_templates:
    enabled: true
    patterns:
      - "could you provide more details"
      - "we could not find the information"
      - "check your spam folder"
      - "ensure your documents are clear and valid"
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key (required for AI mode) | None |
| `ARTIFACTS_DIR` | Path to model artifacts | `notebooks/artifacts` |
| `PORT` | API server port | `8080` |
| `REDIS_URL` | Redis connection URL | None |
| `REDIS_HOST` | Redis host | `localhost` |
| `REDIS_PORT` | Redis port | `6379` |
| `SEED` | Random seed for reproducibility | `42` |

## 🧪 Testing

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test
python -m pytest tests/test_model.py -v
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **End-to-End Tests**: Full pipeline testing
- **Performance Tests**: Latency and throughput testing

### Manual Testing

**Test CLI (AI Mode):**
```bash
python cli/main.py --mode ai
# Test AI chatbot with intelligent responses and escalation
```

**Test CLI (ML Mode):**
```bash
python cli/main.py --mode ml
# Test hybrid ML + AI system
```

**Test API:**
```bash
# Start server
python -m src.service

# Test health
curl http://localhost:8080/health

# Test scoring
curl -X POST "http://localhost:8080/score" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "test", "role": "user", "message": "I need help"}'
```

## 🚀 Deployment

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "-m", "src.service"]
```

**Build and Run:**
```bash
docker build -t sumup-escalation .
docker run -p 8080:8080 sumup-escalation
```

### Production Deployment

**Using Docker Compose:**
```yaml
version: '3.8'
services:
  escalation-api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

**Kubernetes Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: escalation-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: escalation-api
  template:
    metadata:
      labels:
        app: escalation-api
    spec:
      containers:
      - name: escalation-api
        image: sumup-escalation:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

### Monitoring

**Health Checks:**
- `/health` endpoint for liveness probes
- Model loading status
- Redis connectivity

**Metrics:**
- Request latency
- Escalation rate
- Model prediction confidence
- Error rates



## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


---

## 🎯 Summary

This system provides two powerful approaches to customer support escalation detection:

### 🤖 AI Mode (Pure AI)
- **Best for**: Intelligent, context-aware conversations
- **Features**: AI handles both responses and escalation decisions
- **Use case**: When you want full AI control and intelligent reasoning

### 📊 ML Mode (Hybrid)
- **Best for**: Reliable escalation decisions with intelligent responses
- **Features**: ML handles escalation, AI handles responses
- **Use case**: When you want ML reliability combined with AI intelligence

Both modes are production-ready and can be used interchangeably based on your specific needs.

---

**Built with ❤️ for SumUp's Assessment**
