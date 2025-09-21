# src/service.py
import os, json, uvicorn, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import time

from .state import ConvState
from .model import load_artifacts
from .policy import decide
from .logging_config import setup_logging, log_escalation_decision, log_system_health

ART_DIR = os.getenv("ARTIFACTS_DIR", "notebooks/artifacts")

# Setup logging
logger = setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_file=os.getenv("LOG_FILE")
)

# Load artifacts once
try:
    model, feature_order, tau, policy = load_artifacts(ART_DIR)
    ARTIFACTS = {"model": model, "feature_order": feature_order, "tau": tau, "policy": policy}
    logger.info(f"Successfully loaded model with threshold {tau}")
    log_system_health(logger, "model_loading", "healthy", {"threshold": tau})
except Exception as e:
    logger.error(f"Failed to load artifacts: {e}")
    log_system_health(logger, "model_loading", "unhealthy", {"error": str(e)})
    raise

state = ConvState(ttl_seconds=int(policy.get("redis",{}).get("ttl_seconds", 86400)))

app = FastAPI(
    title="SumUp Escalation Detection API",
    description="Real-time escalation detection for customer support conversations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

class ScoreRequest(BaseModel):
    conversation_id: str
    turn_id: Optional[str] = None
    role: str = Field(..., pattern="^(user|bot)$")
    message: str = ""
    prev_bot_text: str = ""
    ts: Optional[str] = None
    lang: Optional[str] = "en"

class ScoreResponse(BaseModel):
    conversation_id: str
    turn_id: Optional[str]
    escalate: bool
    where: str
    score: float
    threshold: float
    fired_rules: list[str]
    reason: str
    latency_ms: int
    model_version: str
    policy_version: str
    state: Dict[str, Any]

@app.get("/health")
def health():
    """Health check endpoint with detailed system status."""
    try:
        # Check model availability
        model_loaded = ARTIFACTS.get("model") is not None
        
        # Check Redis connectivity
        redis_healthy = True
        try:
            test_state = state.load("health_check")
            state.save("health_check", {"test": "value"})
        except Exception as e:
            redis_healthy = False
            logger.warning(f"Redis health check failed: {e}")
        
        # Overall health
        overall_healthy = model_loaded and redis_healthy
        
        health_status = {
            "ok": overall_healthy,
            "model_loaded": model_loaded,
            "redis_healthy": redis_healthy,
            "timestamp": time.time(),
            "version": "1.0.0"
        }
        
        if overall_healthy:
            log_system_health(logger, "health_check", "healthy", health_status)
        else:
            log_system_health(logger, "health_check", "degraded", health_status)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        log_system_health(logger, "health_check", "unhealthy", {"error": str(e)})
        return {"ok": False, "error": str(e), "timestamp": time.time()}

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    """Score a conversation turn for escalation."""
    start_time = time.time()
    
    try:
        cid = req.conversation_id
        st = state.load(cid)
        event = req.dict()
        
        # Make escalation decision
        decision, new_state = decide(event, st, ARTIFACTS)
        state.save(cid, new_state)
        
        # Log the decision
        log_escalation_decision(
            logger=logger,
            conversation_id=cid,
            escalate=decision["escalate"],
            score=decision["score"],
            latency_ms=decision["latency_ms"],
            fired_rules=decision["fired_rules"],
            reason=decision["reason"],
            turn_id=req.turn_id,
            role=req.role
        )
        
        # Return clean response (remove redacted texts)
        decision.pop("redacted_user_text", None)
        decision.pop("redacted_bot_text", None)
        
        return decision
        
    except Exception as e:
        logger.error(f"Scoring failed for conversation {req.conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/metrics")
def metrics():
    """Basic metrics endpoint for monitoring."""
    try:
        # This would typically connect to a metrics store
        # For now, return basic system info
        return {
            "model_threshold": ARTIFACTS.get("tau", 0.0),
            "feature_count": len(ARTIFACTS.get("feature_order", [])),
            "policy_version": ARTIFACTS.get("policy", {}).get("version", "unknown"),
            "uptime_seconds": time.time() - start_time if 'start_time' in globals() else 0
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Metrics unavailable")

if __name__ == "__main__":
    uvicorn.run("src.service:app", host="0.0.0.0", port=int(os.getenv("PORT","8080")), reload=False)
