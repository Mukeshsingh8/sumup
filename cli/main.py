#!/usr/bin/env python3
"""
SumUp Escalation Detection CLI

Interactive command-line interface for testing the escalation detection system.
Simulates conversations between users and bots to test escalation decisions.

Usage:
    python cli/main.py [--artifacts artifacts/] [--help]

Commands:
    bot: <message>     - Bot response message
    user: <message>    - User message (triggers escalation check)
    help               - Show this help message
    examples           - Show example conversations
    stats              - Show conversation statistics
    reset              - Reset conversation state
    quit/exit          - Exit the program
    Ctrl+C             - Exit the program

Examples:
    bot: Hello! How can I help you today?
    user: I need help with my account
    user: I want to speak to a human agent
"""
import os, sys, json, re, joblib, argparse, time
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, continue without it

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ART = "artifacts"

# Import AI detector
try:
    from src.ai_detector import CustomerSupportChatbot, create_customer_support_chatbot
    AI_DETECTOR_AVAILABLE = True
except ImportError:
    AI_DETECTOR_AVAILABLE = False
    CustomerSupportChatbot = None
    create_customer_support_chatbot = None

def load_artifacts(art_dir: str) -> Tuple[object, List[str], float, Dict[str, Any]]:
    """Load trained model and configuration artifacts."""
    try:
        model = joblib.load(os.path.join(art_dir, "model.joblib"))
        with open(os.path.join(art_dir, "feature_order.json"), "r", encoding="utf-8") as f:
            feature_order = json.load(f)
        
        # Read threshold from version.txt
        tau = 0.5
        with open(os.path.join(art_dir, "version.txt"), "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("threshold="):
                    tau = float(line.strip().split("=", 1)[1])
                    break
        
        # Load policy configuration
        policy_path = os.path.join(art_dir, "policy.yaml")
        try:
            import yaml
            with open(policy_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}
        except Exception:
            policy = {}
        
        return model, feature_order, tau, policy
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required artifacts in {art_dir}")
        print(f"   Missing file: {e}")
        print(f"   Make sure you have run the training notebook first.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        sys.exit(1)

def _has_any(patterns, s: str) -> int:
    s = (s or "").lower()
    return int(any(re.search(p, s) for p in patterns))

def featurize_one(user_turn_idx, user_text, prev_bot_text, conv_state, policy):
    # patterns
    rules = (policy.get("rules") or {})
    unhelp = (rules.get("bot_unhelpful_templates") or {}).get("patterns", [
        "could you provide more details","we could not find the information",
        "check your spam folder","ensure your documents are clear and valid"
    ])
    ask_human = (rules.get("explicit_human_request") or {}).get("patterns", [
        r"\b(human|agent|real person|talk to (?:a )?human|speak to (?:a )?human|customer service|support agent)\b"
    ])
    risk_terms = (rules.get("risk_terms") or {}).get("patterns", ["kyc","blocked","chargeback","legal","id verification"])

    def caps_ratio(s): 
        if not s: return 0.0
        caps = sum(1 for c in s if c.isupper())
        letters = sum(1 for c in s if c.isalpha())
        return (caps / letters) if letters else 0.0

    X = {
        "turn_idx": float(user_turn_idx),
        "user_caps_ratio": float(caps_ratio(user_text)),
        "exclam_count": float((user_text or "").count("!")),
        "msg_len": float(len(user_text or "")),
        "bot_unhelpful": float(_has_any(unhelp, prev_bot_text)),
        "user_requests_human": float(_has_any(ask_human, user_text)),
        "risk_terms": float(_has_any(risk_terms, user_text)),
        "no_progress_count": float(conv_state.get("no_progress_count", 0.0)),
        "bot_repeat_count": float(conv_state.get("bot_repeat_count", 0.0)),
    }
    # update rolling counts
    this_bot = (prev_bot_text or "").strip().lower()
    prev_bot = conv_state.get("prev_bot_text", "")
    if prev_bot and this_bot and this_bot == prev_bot:
        conv_state["bot_repeat_count"] = conv_state.get("bot_repeat_count", 0.0) + 1.0
    else:
        conv_state["bot_repeat_count"] = max(conv_state.get("bot_repeat_count", 0.0) - 1.0, 0.0)
    if _has_any(unhelp, this_bot):
        conv_state["no_progress_count"] = conv_state.get("no_progress_count", 0.0) + 1.0
    else:
        conv_state["no_progress_count"] = max(conv_state.get("no_progress_count", 0.0) - 1.0, 0.0)
    conv_state["prev_bot_text"] = this_bot
    return X, conv_state

def select_detection_mode() -> str:
    """Allow user to select between AI Detection and ML Model Detection modes."""
    print("🎯 ESCALATION DETECTION MODE SELECTION")
    print("=" * 50)
    print()
    print("Choose your escalation detection method:")
    print()
    print("1. 🤖 AI Customer Support Chatbot")
    print("   • Real customer support chatbot using Google Gemini")
    print("   • Responds to customer queries intelligently")
    print("   • Context-aware conversation analysis")
    print("   • Automatic escalation detection")
    print("   • Redis caching for faster responses")
    print("   • Requires GEMINI_API_KEY environment variable")
    print()
    print("2. 📊 ML Model Detection")
    print("   • Uses trained machine learning model for escalation decisions")
    print("   • AI generates intelligent responses to customer queries")
    print("   • Hybrid approach: ML for decisions, AI for conversations")
    print("   • Best of both worlds: reliable escalation + intelligent responses")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == "1":
                if AI_DETECTOR_AVAILABLE:
                    print("✅ Selected: AI-Powered Detection")
                    return "ai"
                else:
                    print("❌ AI detector not available. Please select ML Model Detection.")
                    continue
            elif choice == "2":
                print("✅ Selected: ML Model Detection")
                return "ml"
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)

def show_help():
    """Display help information."""
    print("""
🤖 SumUp Escalation Detection CLI

DETECTION MODES:
  🤖 AI Customer Support Chatbot  - Real chatbot using Google Gemini
  📊 ML Model Detection           - Hybrid ML (decisions) + AI (responses) system

COMMANDS:
  <message>          - Type any message (auto-detected as user or bot)
  bot: <message>     - Explicitly mark as bot response
  user: <message>    - Explicitly mark as user message
  help               - Show this help message
  examples           - Show example conversations
  stats              - Show conversation statistics
  reset              - Reset conversation state
  quit/exit          - Exit the program
  Ctrl+C             - Exit the program

EXAMPLES:
  Hello! How can I help you today?  (auto-detected as bot)
  I need help with my account       (auto-detected as user)
  I want to speak to a human agent  (auto-detected as user)
  bot: Sure, I can help with that   (explicitly marked as bot)

AI CHATBOT MODE:
  • Responds to customer queries intelligently
  • Context-aware conversation analysis
  • Automatic escalation detection
  • Redis caching for faster responses

ML MODEL MODE:
  • Rule-based + ML model with engineered features
  • Fast and deterministic predictions
""")

def show_examples():
    """Display example conversations."""
    print("""
📚 EXAMPLE CONVERSATIONS:

Example 1 - Normal conversation:
  bot: Hello! How can I help you today?
  user: I need help with my payment
  → not escalate (p=0.045 < tau=0.081)

Example 2 - Explicit human request:
  bot: Could you provide more details?
  user: I want to speak to a human agent
  → ESCALATE ✅ (rule=explicit_human_request)

Example 3 - Risk terms:
  user: My account is blocked due to KYC issues
  → ESCALATE ✅ (rule=risk_terms)

Example 4 - Frustrated user:
  bot: Could you provide more details about your issue?
  user: I already explained many times! This is ridiculous!
  → ESCALATE ✅ (p=0.892 >= tau=0.081)
""")

def show_stats(conv_state: Dict[str, Any], user_turn_idx: int):
    """Display conversation statistics."""
    print(f"""
📊 CONVERSATION STATISTICS:
  User turns: {user_turn_idx}
  No progress count: {conv_state.get('no_progress_count', 0.0):.1f}
  Bot repeat count: {conv_state.get('bot_repeat_count', 0.0):.1f}
  Previous bot text: "{conv_state.get('prev_bot_text', 'None')[:50]}..."
""")

def reset_conversation() -> Dict[str, Any]:
    """Reset conversation state."""
    return {"no_progress_count": 0.0, "bot_repeat_count": 0.0, "prev_bot_text": ""}

def main():
    ap = argparse.ArgumentParser(
        description="SumUp Escalation Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/main.py
  python cli/main.py --artifacts /path/to/artifacts
  python cli/main.py --mode ai
  python cli/main.py --mode ml
        """
    )
    ap.add_argument("--artifacts", default=ART, help="Artifacts directory (default: artifacts/)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    ap.add_argument("--mode", choices=["ai", "ml"], help="Force detection mode (ai or ml)")
    args = ap.parse_args()

    print("🚀 Loading SumUp Escalation Detection System...")
    
    # Check AI detector availability
    if not AI_DETECTOR_AVAILABLE:
        print("⚠️  AI detector not available. Only ML model mode will be available.")
    
    # Select detection mode
    if args.mode:
        detection_mode = args.mode
        if detection_mode == "ai" and not AI_DETECTOR_AVAILABLE:
            print("❌ AI detector not available. Switching to ML Model Detection.")
            detection_mode = "ml"
        print(f"✅ Mode forced to: {'AI-Powered' if detection_mode == 'ai' else 'ML Model'} Detection")
    else:
        detection_mode = select_detection_mode()
    
    print()
    
    # Initialize based on selected mode
    if detection_mode == "ai":
        # AI Mode - Customer Support Chatbot
        try:
            if not AI_DETECTOR_AVAILABLE:
                raise ImportError("AI detector not available")
            chatbot = create_customer_support_chatbot()
            print("🤖 AI Customer Support Chatbot Mode")
            print("   • Real customer support responses")
            print("   • Context-aware conversation analysis")
            print("   • Automatic escalation detection")
            print(f"   • Redis caching: {chatbot.redis_client is not None}")
            print("   • Model: Google Gemini 1.5 Flash")
            print()
        except ValueError as e:
            print(f"❌ Error: {e}")
            print("Please set your GEMINI_API_KEY environment variable.")
            print("Get your API key from: https://makersuite.google.com/app/apikey")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error initializing AI chatbot: {e}")
            sys.exit(1)
    else:
        # ML Mode
        model, feat_order, tau, policy = load_artifacts(args.artifacts)
        
        guards = (policy.get("guards") or {})
        min_turn_before_model = int(guards.get("min_turn_before_model", 0))

        # Rule patterns
        rules = (policy.get("rules") or {})
        ask_human = (rules.get("explicit_human_request") or {}).get("patterns", [
            r"\b(human|agent|real person|talk to (?:a )?human|speak to (?:a )?human|customer service|support agent)\b"
        ])
        risk_terms = (rules.get("risk_terms") or {}).get("patterns", ["kyc","blocked","chargeback","legal","id verification"])

        print("📊 ML Model Detection Mode")
        print(f"   • Threshold (tau): {tau:.3f}")
        print(f"   • Features: {len(feat_order)}")
        print(f"   • Min turns before model: {min_turn_before_model}")
        print(f"   • Rule patterns: {len(ask_human + risk_terms)}")
        print()

    print("💡 Type 'help' for commands, 'examples' for sample conversations")
    print("   Type 'quit' or Ctrl+C to exit")
    print()

    # Initialize conversation state
    conv_state = reset_conversation()
    prev_bot_text = ""
    user_turn_idx = 0

    while True:
        try:
            line = input("> ").strip()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        
        if not line:
            continue
            
        # Handle special commands
        if line.lower() in ["help", "h"]:
            show_help()
            continue
        elif line.lower() in ["examples", "ex"]:
            show_examples()
            continue
        elif line.lower() in ["stats", "s"]:
            if detection_mode == "ai":
                stats = chatbot.get_conversation_stats()
                print("📊 CONVERSATION STATISTICS (AI Chatbot Mode):")
                print(f"  Total turns: {stats['total_turns']}")
                print(f"  User turns: {stats['user_turns']}")
                print(f"  Bot turns: {stats['bot_turns']}")
                print(f"  Duration: {stats['duration']:.1f}s")
                print(f"  Redis caching: {stats['redis_available']}")
                print(f"  Model: {stats['model']}")
            else:
                show_stats(conv_state, user_turn_idx)
            continue
        elif line.lower() in ["reset", "r"]:
            conv_state = reset_conversation()
            prev_bot_text = ""
            user_turn_idx = 0
            if detection_mode == "ai":
                chatbot.reset_conversation()
            print("🔄 Conversation state reset!")
            continue
        elif line.lower() in ["quit", "exit", "q"]:
            print("👋 Goodbye!")
            break
        
        # Handle bot messages
        if line.lower().startswith("bot:"):
            bot_text = line[4:].strip()
            prev_bot_text = bot_text
            if detection_mode == "ai":
                chatbot.add_turn("bot", bot_text)
            if args.verbose:
                print(f"🤖 Bot: {bot_text}")
            continue
            
        # Handle user messages
        if line.lower().startswith("user:"):
            user_text = line[5:].strip()
        else:
            # Treat as user message if no prefix
            user_text = line.strip()
        
        if user_text:
            if args.verbose:
                print(f"👤 User: {user_text}")

            if detection_mode == "ai":
                # AI Mode: Use customer support chatbot
                response = chatbot.respond_to_customer(user_text)
                
                # Display the chatbot's response
                print(f"🤖 Bot: {response.message}")
                if response.cached:
                    print("   (cached response)")
                
                # Check for escalation
                if response.should_escalate:
                    print(f"🚨 ESCALATE ✅ (AI: {response.escalation_reason})")
                    print(f"   Confidence: {response.confidence:.2f}")
                else:
                    print(f"✅ NO ESCALATION (AI: Normal conversation)")
                    print(f"   Confidence: {response.confidence:.2f}")
                
                user_turn_idx += 1
                continue
            else:
                # ML Mode: Hybrid ML + AI system
                # ML model decides escalation, AI generates response
                try:
                    # Step 1: Use ML model for escalation decision
                    from src.policy import decide
                    
                    # Create event in the format expected by the ML pipeline
                    event = {
                        "conversation_id": "cli_session",
                        "role": "user", 
                        "message": user_text,
                        "prev_bot_text": prev_bot_text,
                        "ts": str(int(time.time())),
                        "lang": "en"
                    }
                    
                    # Make escalation decision using ML model
                    improved_tau = 0.3  # Better threshold for this small dataset
                    decision, conv_state = decide(event, conv_state, {
                        "model": model,
                        "feature_order": feat_order, 
                        "tau": improved_tau,
                        "policy": policy
                    })
                    
                    # Step 2: Use AI to generate response ONLY (no escalation detection)
                    try:
                        if not AI_DETECTOR_AVAILABLE:
                            raise ImportError("AI detector not available")
                        chatbot = create_customer_support_chatbot()
                        
                        # Generate AI response ONLY (no escalation detection)
                        ai_response_text = chatbot.generate_response_only(user_text)
                        
                        # Display AI response
                        print(f"🤖 Bot: {ai_response_text}")
                        
                        # Display ML escalation decision (separate from AI)
                        if decision["escalate"]:
                            print(f"🚨 ESCALATE ✅ (ML: {decision['reason']})")
                            print(f"   ML Score: {decision['score']:.3f} | Threshold: {decision['threshold']:.3f}")
                            print(f"   Where: {decision['where']} | Rules: {decision['fired_rules']}")
                        else:
                            print(f"✅ NO ESCALATION (ML: {decision['reason']})")
                            print(f"   ML Score: {decision['score']:.3f} | Threshold: {decision['threshold']:.3f}")
                            print(f"   Where: {decision['where']}")
                        
                        if args.verbose:
                            print(f"   ML Features: turn={conv_state.get('user_turn_idx', 0)}, "
                                  f"no_progress={conv_state.get('no_progress_count', 0):.1f}, "
                                  f"bot_repeat={conv_state.get('bot_repeat_count', 0):.1f}")
                            print(f"   ML Latency: {decision['latency_ms']}ms")
                            print(f"   AI: Response-only mode (no escalation detection)")
                        
                    except Exception as ai_error:
                        # Fallback: AI not available, just show ML decision
                        print(f"⚠️  AI response generation failed: {ai_error}")
                        print("   Using ML decision only...")
                        
                        if decision["escalate"]:
                            print(f"🚨 ESCALATE ✅ (ML: {decision['reason']})")
                            print(f"   ML Score: {decision['score']:.3f} | Threshold: {decision['threshold']:.3f}")
                        else:
                            print(f"✅ NO ESCALATION (ML: {decision['reason']})")
                            print(f"   ML Score: {decision['score']:.3f} | Threshold: {decision['threshold']:.3f}")
                        
                except Exception as e:
                    print(f"❌ Error during ML prediction: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()

                user_turn_idx += 1
                continue

        # Invalid command
        print("🤔 I'm not sure what you meant. Try typing a message or 'help' for commands!")

if __name__ == "__main__":
    main()
