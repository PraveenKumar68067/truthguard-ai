import streamlit as st
import pandas as pd
import json
import re
import os
import joblib
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TruthGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# BASIC UI
# ============================================================

st.title("🛡️ TruthGuard AI")
st.subheader("Fake News & Spam Email Detection Chatbot")

st.write(
    "TruthGuard AI analyzes news articles and emails using Gemini AI, "
    "trained ensemble ML fallback models, and optional long-term memory "
    "to provide verdicts, suspicion scores, reasons, and safety advice."
)


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "sample_text" not in st.session_state:
    st.session_state.sample_text = ""

if "content_type" not in st.session_state:
    st.session_state.content_type = "News Article"


# ============================================================
# LOAD GEMINI SETTINGS
# ============================================================

try:
    API_KEY = st.secrets.get("GEMINI_API_KEY", None)
    MODEL_NAME = st.secrets.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
except Exception:
    API_KEY = None
    MODEL_NAME = "gemini-2.5-flash-lite"


# ============================================================
# LOAD HINDSIGHT MEMORY SETTINGS
# ============================================================

try:
    HINDSIGHT_API_KEY = st.secrets.get("HINDSIGHT_API_KEY", None)
    HINDSIGHT_BASE_URL = st.secrets.get("HINDSIGHT_BASE_URL", None)
    HINDSIGHT_BANK_ID = st.secrets.get("HINDSIGHT_BANK_ID", "truthguard-ai-final")
except Exception:
    HINDSIGHT_API_KEY = None
    HINDSIGHT_BASE_URL = None
    HINDSIGHT_BANK_ID = "truthguard-ai-final"


def hindsight_available():
    return bool(HINDSIGHT_API_KEY and HINDSIGHT_BASE_URL and HINDSIGHT_BANK_ID)


# ============================================================
# LOAD LOCAL ML MODELS
# ============================================================

@st.cache_resource
def load_ml_models():
    models = {
        "news": None,
        "email": None
    }

    if os.path.exists("fake_news_model.pkl"):
        try:
            models["news"] = joblib.load("fake_news_model.pkl")
        except Exception as e:
            st.warning(f"Could not load fake_news_model.pkl: {e}")

    if os.path.exists("spam_model.pkl"):
        try:
            models["email"] = joblib.load("spam_model.pkl")
        except Exception as e:
            st.warning(f"Could not load spam_model.pkl: {e}")

    return models


ml_models = load_ml_models()


# ============================================================
# JSON EXTRACTOR
# ============================================================

def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except Exception:
            return None

    return None


# ============================================================
# WARNING WORD DETECTOR
# ============================================================

def find_warning_words(text):
    text_lower = text.lower()

    suspicious_patterns = [
        "urgent",
        "click here",
        "free money",
        "lottery",
        "winner",
        "limited time",
        "verify your account",
        "password",
        "bank account",
        "miracle cure",
        "shocking",
        "breaking",
        "guaranteed",
        "act now",
        "claim reward",
        "congratulations",
        "risk-free",
        "share this message",
        "before it gets deleted",
        "otp",
        "prize",
        "reward",
        "account verification",
        "verify",
        "claim",
        "deleted",
        "secret remedy",
        "government is hiding",
        "immediately",
        "bank details",
        "personal information",
        "limited offer",
        "you have won",
        "cure all diseases",
        "hidden truth",
        "forward this message",
        "verify your identity",
        "account suspended",
        "unusual login",
        "confirm your details"
    ]

    matched = []

    for pattern in suspicious_patterns:
        if pattern in text_lower:
            matched.append(pattern)

    return list(dict.fromkeys(matched))


# ============================================================
# INPUT TYPE DETECTOR
# ============================================================

def detect_possible_input_type(text):
    text_lower = text.lower()

    email_indicators = [
        "dear",
        "regards",
        "subject",
        "click here",
        "verify your account",
        "password",
        "otp",
        "bank account",
        "claim your reward",
        "email",
        "sender",
        "inbox",
        "department",
        "please carry",
        "room",
        "congratulations",
        "account suspended",
        "verify your identity"
    ]

    news_indicators = [
        "breaking",
        "news",
        "officials",
        "government",
        "report",
        "announced",
        "article",
        "source",
        "published",
        "doctors",
        "reserve bank",
        "policy",
        "committee",
        "inflation",
        "growth projections",
        "official statement",
        "rbi"
    ]

    email_score = sum(1 for word in email_indicators if word in text_lower)
    news_score = sum(1 for word in news_indicators if word in text_lower)

    if email_score > news_score:
        return "Email Message"
    elif news_score > email_score:
        return "News Article"
    else:
        return "Unclear"


# ============================================================
# SAMPLE INPUTS
# ============================================================

def get_sample_text(sample_type):
    samples = {
        "Spam Email": """Congratulations! You have won ₹5,00,000 in our lucky draw. Click here immediately to claim your reward. You must verify your bank account within 2 hours or the prize will expire.""",

        "Non-Spam Email": """Dear Student,

This is to inform you that your AI Essentials class test will be conducted tomorrow at 10:00 AM in Room 204. Please carry your ID card and reach the classroom 10 minutes before the test begins.

Regards,
Department of Computer Science""",

        "Fake News": """Breaking shocking news! Doctors have discovered one secret home remedy that cures all diseases in 24 hours. Government officials are hiding this truth from the public. Share this message before it gets deleted.""",

        "Real News": """The Reserve Bank of India announced its monetary policy decision after the scheduled meeting of the Monetary Policy Committee. According to the official statement published on the RBI website, the committee reviewed inflation trends, growth projections, liquidity conditions, and global economic developments before announcing the policy decision. The report also mentioned that future policy actions will depend on incoming economic data and inflation outlook."""
    }

    return samples.get(sample_type, "")


# ============================================================
# HINDSIGHT MEMORY FUNCTIONS
# ============================================================

def make_safe_memory_summary(user_text, content_type, result, analysis_mode):
    """
    Store only summarized detection patterns.
    Do not store full private email/news text.
    """
    verdict = result.get("verdict", "Unknown")
    score = result.get("suspicion_score", 0)
    risk = result.get("risk_level", "Unknown")
    reasons = result.get("key_reasons", [])

    warning_words = find_warning_words(user_text)

    summary = f"""
TruthGuard AI analyzed a {content_type}.
Verdict: {verdict}.
Suspicion score: {score}/100.
Risk level: {risk}.
Analysis mode: {analysis_mode}.
Important pattern indicators: {", ".join(warning_words[:8]) if warning_words else "No major suspicious keyword pattern found"}.
Main reasons: {"; ".join(reasons[:3]) if reasons else "No reason available"}.
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
"""

    return summary.strip()


def get_hindsight_client():
    try:
        from hindsight_client import Hindsight

        try:
            client = Hindsight(
                base_url=HINDSIGHT_BASE_URL,
                api_key=HINDSIGHT_API_KEY
            )
        except TypeError:
            client = Hindsight(
                base_url=HINDSIGHT_BASE_URL
            )

        return client, None

    except Exception as e:
        return None, str(e)


def close_hindsight_client(client):
    try:
        if hasattr(client, "close"):
            result = client.close()

            if hasattr(result, "__await__"):
                pass

    except Exception:
        pass


def recall_hindsight_memory(user_text, content_type):
    if not hindsight_available():
        return "Hindsight memory is not configured."

    client, error = get_hindsight_client()

    if error:
        return f"Hindsight client creation failed: {error}"

    try:
        query = f"""
Find previous TruthGuard AI memories similar to this {content_type}.
Focus on suspicious patterns, verdicts, risk levels, phishing indicators, spam signs, and fake news indicators.
Do not return private full text.
Current text pattern preview: {user_text[:500]}
"""

        memories = client.recall(
            bank_id=HINDSIGHT_BANK_ID,
            query=query
        )

        close_hindsight_client(client)

        if not memories:
            return "No similar previous memory found."

        cleaned_memories = []
        raw_memory_text = str(memories)

        extracted_texts = re.findall(
            r"text='(.*?)'(?:,|\))",
            raw_memory_text,
            flags=re.DOTALL
        )

        if not extracted_texts:
            extracted_texts = re.findall(
                r'text="(.*?)"(?:,|\))',
                raw_memory_text,
                flags=re.DOTALL
            )

        if not extracted_texts:
            if hasattr(memories, "results"):
                memory_items = memories.results
            else:
                memory_items = memories if isinstance(memories, list) else []

            for memory in memory_items:
                if hasattr(memory, "text"):
                    extracted_texts.append(memory.text)
                elif hasattr(memory, "content"):
                    extracted_texts.append(memory.content)

        for memory_text in extracted_texts[:5]:
            memory_text = memory_text.replace("\\n", " ")
            memory_text = memory_text.replace("\\'", "'")
            memory_text = memory_text.replace('\\"', '"')
            memory_text = re.sub(r"\s+", " ", memory_text).strip()

            unwanted_phrases = [
                "team rule",
                "api route handlers",
                "business logic moved to services",
                "route handlers should be thin"
            ]

            if any(phrase in memory_text.lower() for phrase in unwanted_phrases):
                continue

            if len(memory_text) > 450:
                memory_text = memory_text[:450] + "..."

            if memory_text:
                cleaned_memories.append(memory_text)

        if not cleaned_memories:
            return "No directly relevant previous TruthGuard memory found."

        final_output = "The system found similar previous analysis patterns:\n\n"

        for index, memory_text in enumerate(cleaned_memories[:3], start=1):
            final_output += f"- **Case {index}:** {memory_text}\n\n"

        final_output += (
            "\nThese memories are used only for comparison. "
            "The final decision is still based on the current input analysis."
        )

        return final_output.strip()

    except Exception as e:
        close_hindsight_client(client)
        return f"Hindsight recall failed: {e}"


def retain_hindsight_memory(user_text, content_type, result, analysis_mode):
    if not hindsight_available():
        return "Hindsight memory is not configured."

    client, error = get_hindsight_client()

    if error:
        return f"Hindsight client creation failed: {error}"

    try:
        safe_summary = make_safe_memory_summary(
            user_text=user_text,
            content_type=content_type,
            result=result,
            analysis_mode=analysis_mode
        )

        try:
            client.retain(
                bank_id=HINDSIGHT_BANK_ID,
                content=safe_summary
            )
        except TypeError:
            client.retain(
                bank_id=HINDSIGHT_BANK_ID,
                text=safe_summary
            )

        close_hindsight_client(client)

        return "Memory saved successfully."

    except Exception as e:
        close_hindsight_client(client)
        return f"Hindsight retain failed: {e}"


# ============================================================
# GEMINI ANALYSIS FUNCTION
# ============================================================

def analyze_with_gemini(user_text, content_type, temperature, top_p):
    if API_KEY is None:
        return {
            "error": True,
            "message": "Gemini API key not found."
        }

    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        return {
            "error": True,
            "message": f"Gemini library import failed: {e}"
        }

    try:
        client = genai.Client(api_key=API_KEY)

        system_instruction = """
You are TruthGuard AI, a domain-specific AI assistant for detecting fake news and spam emails.

Your task:
1. Analyze the given text carefully.
2. If it is a news article, detect whether it is likely fake or likely real.
3. If it is an email, detect whether it is likely spam, phishing, or safe.
4. Give a suspicion score from 0 to 100.
5. Give risk level as Low, Medium, or High.
6. Give simple professional reasons.
7. Give safety advice.
8. Do not claim 100% certainty.
9. Return ONLY valid JSON.
10. Do not use markdown.
11. Do not wrap the response in ```json.

Required JSON format:
{
  "verdict": "Likely Fake News / Likely Real News / Likely Spam / Possible Phishing / Safe Email / Unclear",
  "suspicion_score": 0,
  "risk_level": "Low / Medium / High",
  "key_reasons": ["reason 1", "reason 2", "reason 3"],
  "safety_advice": ["advice 1", "advice 2"],
  "explanation": "short professional explanation"
}
"""

        prompt = f"""
Content Type: {content_type}

Analyze this text:

{user_text}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=1200,
                response_mime_type="application/json"
            )
        )

        return {
            "error": False,
            "text": response.text
        }

    except Exception as e:
        return {
            "error": True,
            "message": str(e)
        }


# ============================================================
# ENSEMBLE ML HELPER FUNCTIONS
# ============================================================

def get_positive_class_probability_from_model(model, user_text, positive_class=1):
    """
    Returns probability of positive/suspicious class from a single model.
    Works with sklearn pipelines that support predict_proba().
    """
    try:
        probabilities = model.predict_proba([user_text])[0]
        classes = list(model.classes_)

        if positive_class in classes:
            positive_index = classes.index(positive_class)
            return float(probabilities[positive_index])

        if str(positive_class) in classes:
            positive_index = classes.index(str(positive_class))
            return float(probabilities[positive_index])

        return float(max(probabilities))

    except Exception:
        return None


def get_ensemble_probability(model_object, user_text, positive_class=1):
    """
    Supports:
    1. New ensemble format:
       {
           "model_format": "ensemble_v1",
           "models": {...},
           "threshold": ...
       }

    2. Old direct sklearn Pipeline format.
    """
    if isinstance(model_object, dict) and model_object.get("model_format") == "ensemble_v1":
        models = model_object.get("models", {})
        probs = []

        for model_name, model in models.items():
            prob = get_positive_class_probability_from_model(
                model=model,
                user_text=user_text,
                positive_class=positive_class
            )

            if prob is not None:
                probs.append(prob)

        if len(probs) == 0:
            return None, model_object.get("threshold", 0.5), "Ensemble model probability failed"

        final_prob = sum(probs) / len(probs)
        threshold = model_object.get("threshold", 0.5)
        model_type = model_object.get(
            "model_type",
            "TF-IDF Ensemble: Logistic Regression + Linear SVM + Naive Bayes"
        )

        return final_prob, threshold, model_type

    # Old model format
    prob = get_positive_class_probability_from_model(
        model=model_object,
        user_text=user_text,
        positive_class=positive_class
    )

    return prob, 0.5, "TF-IDF Logistic Regression"


def adjust_ml_suspicion_score(raw_score, user_text, content_type):
    """
    Calibration layer for local ML fallback.
    Reduces false positives for normal business/academic emails and trusted news.
    Avoids unrealistic 0/100 confidence.
    """
    text_lower = user_text.lower()
    adjusted_score = raw_score

    trusted_news_indicators = [
        "official statement",
        "published on",
        "reserve bank of india",
        "rbi website",
        "monetary policy committee",
        "according to",
        "scheduled meeting",
        "policy decision",
        "inflation outlook",
        "growth projections",
        "official website",
        "microsoft",
        "quarterly earnings",
        "revenue",
        "cloud services",
        "financial performance",
        "company reported",
        "earnings report",
        "publicly traded company",
        "official filing",
        "sec filing",
        "press release",
        "reported revenue",
        "increase in revenue",
        "business update",
        "financial results"
    ]

    safe_email_indicators = [
        "dear team",
        "dear student",
        "hello",
        "regards",
        "sincerely",
        "thank you",
        "department",
        "computer science",
        "university",
        "class test",
        "room",
        "please carry your id card",
        "conducted tomorrow",
        "10:00 am",
        "meeting reminder",
        "meeting agenda",
        "scheduled meeting",
        "project discussion",
        "invoice attached",
        "please find attached",
        "attached invoice",
        "payment receipt",
        "application received",
        "thank you for applying",
        "job application",
        "interview schedule",
        "event invitation",
        "university event",
        "workshop",
        "seminar",
        "registered successfully"
    ]

    fake_news_indicators = [
        "shocking",
        "breaking shocking",
        "secret remedy",
        "cure all diseases",
        "government is hiding",
        "share this message",
        "before it gets deleted",
        "miracle cure",
        "hidden truth",
        "rumors spread",
        "without evidence",
        "viral claim",
        "no official confirmation",
        "secret plan",
        "doctors hate this",
        "they don't want you to know"
    ]

    strong_spam_indicators = [
        "click here",
        "verify your account",
        "bank account",
        "credit card",
        "password",
        "otp",
        "claim your reward",
        "you have won",
        "winner",
        "lottery",
        "limited time",
        "act fast",
        "prize will expire",
        "within 2 hours",
        "account suspended",
        "urgent account suspension",
        "confirm your identity",
        "donate now",
        "fake charity",
        "investment",
        "500% returns",
        "financial freedom"
    ]

    medium_spam_indicators = [
        "password reset",
        "reset your password",
        "invoice",
        "attached",
        "account",
        "application",
        "payment",
        "limited offer"
    ]

    if content_type == "News Article":
        trusted_hits = sum(1 for phrase in trusted_news_indicators if phrase in text_lower)
        fake_hits = sum(1 for phrase in fake_news_indicators if phrase in text_lower)

        adjusted_score -= trusted_hits * 8
        adjusted_score += fake_hits * 8

    elif content_type == "Email Message":
        safe_hits = sum(1 for phrase in safe_email_indicators if phrase in text_lower)
        strong_spam_hits = sum(1 for phrase in strong_spam_indicators if phrase in text_lower)
        medium_spam_hits = sum(1 for phrase in medium_spam_indicators if phrase in text_lower)

        # Strong false-positive reduction for normal professional/academic emails
        adjusted_score -= safe_hits * 10

        # Strong spam indicators increase score
        adjusted_score += strong_spam_hits * 8

        # Medium indicators only increase slightly because they can appear in real emails also
        adjusted_score += medium_spam_hits * 2

    adjusted_score = max(0, min(100, int(adjusted_score)))

    # Avoid unrealistic extremes
    if adjusted_score <= 3:
        adjusted_score = 5
    elif adjusted_score >= 97:
        adjusted_score = 95

    return adjusted_score
# ============================================================
# LOCAL ML FALLBACK ANALYSIS
# ============================================================

def analyze_with_ml_model(user_text, content_type):
    warning_words = find_warning_words(user_text)

    # ========================================================
    # NEWS ARTICLE FALLBACK
    # ========================================================
    if content_type == "News Article":
        model_object = ml_models["news"]

        if model_object is None:
            return {
                "verdict": "Fake News ML model not found",
                "suspicion_score": 0,
                "risk_level": "Unknown",
                "key_reasons": [
                    "Gemini API failed or was turned off.",
                    "fake_news_model.pkl is missing from the project folder.",
                    "The system could not perform local fake news analysis."
                ],
                "safety_advice": [
                    "Add fake_news_model.pkl to the project folder.",
                    "Restart the Streamlit app."
                ],
                "explanation": "The local fake news model is not available."
            }

        fake_probability, threshold, model_type = get_ensemble_probability(
            model_object=model_object,
            user_text=user_text,
            positive_class=1
        )

        if fake_probability is None:
            fake_probability = 0.50

        raw_score = int(fake_probability * 100)

        suspicion_score = adjust_ml_suspicion_score(
            raw_score=raw_score,
            user_text=user_text,
            content_type=content_type
        )

        if suspicion_score >= 70:
            verdict = "Likely Fake News"
            risk_level = "High"
        elif suspicion_score >= 40:
            verdict = "Uncertain / Needs Verification"
            risk_level = "Medium"
        else:
            verdict = "Likely Real News"
            risk_level = "Low"

        key_reasons = [
            "Gemini API was unavailable or turned off, so the trained local ML fallback was used.",
            f"The fallback uses {model_type}.",
            "The final suspicion score is calculated using soft-voting probability from local models when ensemble format is available.",
            "A small calibration layer reduces false positives for trusted-source indicators and increases score for suspicious fake-news patterns."
        ]

        if warning_words:
            key_reasons.append(
                "Suspicious terms found: " + ", ".join(warning_words[:8])
            )

        return {
            "verdict": verdict,
            "suspicion_score": suspicion_score,
            "risk_level": risk_level,
            "key_reasons": key_reasons,
            "safety_advice": [
                "Verify the article from trusted news sources.",
                "Check the author, date, source, and supporting evidence.",
                "Medium-risk fallback results should be treated as uncertain and manually verified."
            ],
            "explanation": (
                "The local fallback analyzed the article using TF-IDF text features. "
                "If an ensemble model is available, Logistic Regression, Linear SVM, and Naive Bayes "
                "are combined through soft voting. The score is then lightly calibrated using trusted-source "
                "and suspicious-pattern indicators to reduce false positives."
            )
        }

    # ========================================================
    # EMAIL FALLBACK
    # ========================================================
    elif content_type == "Email Message":
        model_object = ml_models["email"]

        if model_object is None:
            return {
                "verdict": "Spam Email ML model not found",
                "suspicion_score": 0,
                "risk_level": "Unknown",
                "key_reasons": [
                    "Gemini API failed or was turned off.",
                    "spam_model.pkl is missing from the project folder.",
                    "The system could not perform local spam email analysis."
                ],
                "safety_advice": [
                    "Add spam_model.pkl to the project folder.",
                    "Restart the Streamlit app."
                ],
                "explanation": "The local spam email model is not available."
            }

        spam_probability, threshold, model_type = get_ensemble_probability(
            model_object=model_object,
            user_text=user_text,
            positive_class=1
        )

        if spam_probability is None:
            spam_probability = 0.50

        raw_score = int(spam_probability * 100)

        suspicion_score = adjust_ml_suspicion_score(
            raw_score=raw_score,
            user_text=user_text,
            content_type=content_type
        )

        if suspicion_score >= 80:
            verdict = "Likely Spam / Possible Phishing"
            risk_level = "High"
        elif suspicion_score >= 55:
            verdict = "Uncertain / Review Recommended"
            risk_level = "Medium"
        else:
            verdict = "Safe Email"
            risk_level = "Low"

        key_reasons = [
            "Gemini API was unavailable or turned off, so the trained local ML fallback was used.",
            f"The fallback uses {model_type}.",
            "The final suspicion score is calculated using soft-voting probability from local models when ensemble format is available.",
            "A small calibration layer reduces false positives for safe-email indicators and increases score for suspicious spam/phishing patterns."
        ]

        if warning_words:
            key_reasons.append(
                "Suspicious terms found: " + ", ".join(warning_words[:8])
            )

        return {
            "verdict": verdict,
            "suspicion_score": suspicion_score,
            "risk_level": risk_level,
            "key_reasons": key_reasons,
            "safety_advice": [
                "Do not click suspicious links.",
                "Do not share passwords, OTP, or bank details.",
                "Medium-risk fallback results should be reviewed manually before action."
            ],
            "explanation": (
                "The local fallback analyzed the email using TF-IDF text features. "
                "If an ensemble model is available, Logistic Regression, Linear SVM, and Naive Bayes "
                "are combined through soft voting. The score is then lightly calibrated using safe-email "
                "and suspicious-pattern indicators to reduce false positives."
            )
        }

    return {
        "verdict": "Unknown content type",
        "suspicion_score": 0,
        "risk_level": "Unknown",
        "key_reasons": ["Invalid content type selected."],
        "safety_advice": ["Select News Article or Email Message."],
        "explanation": "The system could not analyze the input."
    }


# ============================================================
# DISPLAY RESULT FUNCTION
# ============================================================

def display_result(result, analysis_mode, user_text, content_type):
    final_score = int(result.get("suspicion_score", 0))
    final_score = max(0, min(final_score, 100))

    final_risk = result.get("risk_level", "Unknown")
    verdict = result.get("verdict", "Unknown")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Verdict", verdict)

    with col2:
        st.metric("Suspicion Score", f"{final_score}/100")

    with col3:
        st.metric("Risk Level", final_risk)

    with col4:
        st.metric("Analysis Mode", analysis_mode)

    st.progress(final_score / 100)

    # -----------------------------
    # RESULT SUMMARY BOX
    # -----------------------------
    if final_risk == "High":
        st.error(
            f"High-risk content detected. The system classified this input as **{verdict}** "
            f"with a suspicion score of **{final_score}/100**."
        )
    elif final_risk == "Medium":
        st.warning(
            f"Moderate-risk or uncertain content detected. The system classified this input as **{verdict}** "
            f"with a suspicion score of **{final_score}/100**."
        )
    elif final_risk == "Low":
        st.success(
            f"Low-risk content detected. The system classified this input as **{verdict}** "
            f"with a suspicion score of **{final_score}/100**."
        )
    else:
        st.info(
            f"The system classified this input as **{verdict}** "
            f"with a suspicion score of **{final_score}/100**."
        )

    if analysis_mode == "Local ML Fallback":
        st.caption(
            "Note: Local ML fallback is based on TF-IDF word patterns and may be less context-aware than Gemini. "
            "Medium-risk fallback results are treated as uncertain and should be verified manually."
        )

    # -----------------------------
    # KEY REASONS
    # -----------------------------
    st.markdown("### Key Reasons")

    reasons = result.get("key_reasons", [])

    if reasons:
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("- No specific reason provided.")

    # -----------------------------
    # SAFETY ADVICE
    # -----------------------------
    st.markdown("### Safety Advice")

    advice_list = result.get("safety_advice", [])

    if advice_list:
        for advice in advice_list:
            st.write(f"- {advice}")
    else:
        st.write("- No safety advice provided.")

    # -----------------------------
    # PROFESSIONAL EXPLANATION
    # -----------------------------
    st.markdown("### Professional Explanation")

    if final_score >= 70:
        confidence_note = (
            "The system detected strong suspicious signals. "
            "This content should be treated as high risk until it is verified from trusted sources."
        )
    elif final_score >= 40:
        confidence_note = (
            "The system detected moderate suspicious signals. "
            "This result should be interpreted as uncertain and verified before taking action."
        )
    else:
        confidence_note = (
            "The system detected limited suspicious signals. "
            "However, important news claims or emails should still be verified from trusted sources."
        )

    if analysis_mode == "Gemini API":
        method_note = (
            "This result was generated using Gemini API. The model performed contextual language analysis "
            "to identify suspicious wording, possible intent, risk indicators, credibility issues, and safety concerns."
        )
    elif analysis_mode == "Local ML Fallback":
        method_note = (
            "This result was generated using the trained local machine learning fallback. "
            "If the new ensemble model is available, the fallback combines Logistic Regression, Linear SVM, "
            "and Naive Bayes using soft voting."
        )
    else:
        method_note = "This result was generated using the available analysis pipeline."

    explanation_text = result.get("explanation", "No explanation provided.")

    st.info(
        f"""
**Decision Summary:**  
The system classified this input as **{verdict}** with a **{final_risk}** risk level and a suspicion score of **{final_score}/100**.

**Analysis Method:**  
{method_note}

**Reasoning:**  
{explanation_text}

**Confidence Interpretation:**  
{confidence_note}

**Recommended Action:**  
For news articles, verify the claim from trusted news sources, official websites, or multiple reliable references.  
For emails, avoid clicking unknown links and never share passwords, OTPs, bank details, or personal information.
"""
    )

    # -----------------------------
    # MEMORY INSIGHT BELOW RESULT
    # -----------------------------
    if hindsight_available():
        memory_insight = recall_hindsight_memory(
            user_text=user_text,
            content_type=content_type
        )

        st.markdown("### Memory Insight")
        with st.expander("View similar previous analysis patterns", expanded=False):
            st.markdown(memory_insight)

    # -----------------------------
    # SAVE TO HISTORY
    # -----------------------------
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Content Type": content_type,
        "Text Preview": user_text[:60] + "..." if len(user_text) > 60 else user_text,
        "Verdict": verdict,
        "Suspicion Score": final_score,
        "Risk Level": final_risk,
        "Analysis Mode": analysis_mode
    })

    # -----------------------------
    # SAVE MEMORY AFTER RESULT
    # -----------------------------
    if hindsight_available():
        memory_status = retain_hindsight_memory(
            user_text=user_text,
            content_type=content_type,
            result=result,
            analysis_mode=analysis_mode
        )

        st.caption(f"Hindsight Memory: {memory_status}")


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.title("⚙️ Model Settings")

content_type = st.sidebar.selectbox(
    "Select content type",
    ["News Article", "Email Message"],
    index=["News Article", "Email Message"].index(st.session_state.content_type),
    key="content_type_selector"
)

st.session_state.content_type = content_type

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.1
)

top_p = st.sidebar.slider(
    "Top-p",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.1
)

use_gemini = st.sidebar.toggle(
    "Use Gemini API",
    value=True,
    help="Turn off during testing to save Gemini quota. If off, the app uses local ML fallback directly."
)

st.sidebar.info(
    "Low temperature gives stable and factual answers. "
    "Higher temperature gives more creative but less predictable answers."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")

if API_KEY:
    st.sidebar.success("Gemini API key found")
else:
    st.sidebar.error("Gemini API key missing")

if ml_models["news"] is not None:
    if isinstance(ml_models["news"], dict) and ml_models["news"].get("model_format") == "ensemble_v1":
        st.sidebar.success("Fake News Ensemble model loaded")
    else:
        st.sidebar.success("Fake News ML model loaded")
else:
    st.sidebar.warning("Fake News ML model missing")

if ml_models["email"] is not None:
    if isinstance(ml_models["email"], dict) and ml_models["email"].get("model_format") == "ensemble_v1":
        st.sidebar.success("Spam Email Ensemble model loaded")
    else:
        st.sidebar.success("Spam Email ML model loaded")
else:
    st.sidebar.warning("Spam Email ML model missing")

if hindsight_available():
    st.sidebar.success("Hindsight memory configured")
else:
    st.sidebar.warning("Hindsight memory not configured")


# ============================================================
# USER INPUT
# ============================================================

st.markdown("---")
st.markdown("## Quick Test Samples")
st.caption(
    "Use these examples to quickly test spam, safe email, fake news, and real news detection."
)

sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)

with sample_col1:
    if st.button("Sample Spam Email"):
        st.session_state.sample_text = get_sample_text("Spam Email")
        st.session_state.content_type = "Email Message"
        st.rerun()

with sample_col2:
    if st.button("Sample Non-Spam Email"):
        st.session_state.sample_text = get_sample_text("Non-Spam Email")
        st.session_state.content_type = "Email Message"
        st.rerun()

with sample_col3:
    if st.button("Sample Fake News"):
        st.session_state.sample_text = get_sample_text("Fake News")
        st.session_state.content_type = "News Article"
        st.rerun()

with sample_col4:
    if st.button("Sample Real News"):
        st.session_state.sample_text = get_sample_text("Real News")
        st.session_state.content_type = "News Article"
        st.rerun()

st.caption("Tip: Sample buttons automatically select the correct content type in the sidebar.")

st.markdown("## Input Text")

user_text = st.text_area(
    "Paste your news article or email here:",
    value=st.session_state.sample_text,
    height=260,
    placeholder="Paste a news article or email message here..."
)

analyze_button = st.button("Analyze Text", type="primary")


# ============================================================
# ANALYSIS LOGIC
# ============================================================

if analyze_button:
    if not user_text.strip():
        st.warning("Please paste some text first.")

    else:
        detected_type = detect_possible_input_type(user_text)

        if detected_type != "Unclear" and detected_type != content_type:
            st.error(
                f"Content type mismatch detected. The text looks more like **{detected_type}**, "
                f"but selected type is **{content_type}**."
            )

            st.info(
                "Please select the correct content type from the sidebar before analysis. "
                "This prevents the news model from analyzing emails or the spam model from analyzing news articles."
            )

            st.stop()

        with st.spinner("Analyzing text..."):

            # ------------------------------------------------
            # STEP 1: TRY GEMINI FIRST IF ENABLED
            # ------------------------------------------------
            if use_gemini:
                gemini_result = analyze_with_gemini(
                    user_text=user_text,
                    content_type=content_type,
                    temperature=temperature,
                    top_p=top_p
                )
            else:
                gemini_result = {
                    "error": True,
                    "message": "Gemini API is turned off. Using local ML fallback to save quota."
                }

            # ------------------------------------------------
            # CASE 1: GEMINI FAILED OR OFF → USE LOCAL ML FALLBACK
            # ------------------------------------------------
            if gemini_result["error"]:
                st.error("Gemini analysis failed or is turned off.")
                st.write(gemini_result["message"])

                st.info(
                    "The system is using the trained local ML fallback. "
                    "If the new ensemble model files are available, it combines Logistic Regression, "
                    "Linear SVM, and Naive Bayes using soft voting."
                )

                result = analyze_with_ml_model(
                    user_text=user_text,
                    content_type=content_type
                )

                display_result(
                    result=result,
                    analysis_mode="Local ML Fallback",
                    user_text=user_text,
                    content_type=content_type
                )

            # ------------------------------------------------
            # CASE 2: GEMINI WORKED
            # ------------------------------------------------
            else:
                result = extract_json(gemini_result["text"])

                if result is None:
                    st.warning("Gemini response could not be parsed as valid JSON.")

                    with st.expander("View Raw Gemini Response"):
                        st.write(gemini_result["text"])

                    st.info(
                        "Because Gemini response was not in the expected JSON format, "
                        "the system is using the trained local ML fallback."
                    )

                    result = analyze_with_ml_model(
                        user_text=user_text,
                        content_type=content_type
                    )

                    display_result(
                        result=result,
                        analysis_mode="Local ML Fallback",
                        user_text=user_text,
                        content_type=content_type
                    )

                else:
                    warning_words = find_warning_words(user_text)

                    if warning_words:
                        existing_reasons = result.get("key_reasons", [])
                        existing_reasons.append(
                            "Suspicious terms found: " + ", ".join(warning_words[:8])
                        )
                        result["key_reasons"] = existing_reasons

                    display_result(
                        result=result,
                        analysis_mode="Gemini API",
                        user_text=user_text,
                        content_type=content_type
                    )


# ============================================================
# HISTORY SECTION
# ============================================================

if len(st.session_state.history) > 0:
    st.markdown("---")
    st.markdown("## Analysis History")

    if st.button("Clear Analysis History"):
        st.session_state.history = []
        st.rerun()

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df, width="stretch")

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Analysis History as CSV",
        data=csv,
        file_name="analysis_history.csv",
        mime="text/csv"
    )


# ============================================================
# SYSTEM WORKFLOW SECTION
# ============================================================

st.markdown("---")
st.markdown("## How TruthGuard AI Works")

flow_col1, flow_col2, flow_col3 = st.columns(3)

with flow_col1:
    st.markdown("### 1. GenAI Analysis")
    st.write(
        "Gemini 2.5 Flash-Lite analyzes the text and generates a verdict, "
        "suspicion score, reasons, and safety advice."
    )

with flow_col2:
    st.markdown("### 2. Ensemble ML Fallback")
    st.write(
        "If Gemini fails or is turned off, the local fallback can combine "
        "Logistic Regression, Linear SVM, and Naive Bayes using soft voting."
    )

with flow_col3:
    st.markdown("### 3. Memory Layer")
    st.write(
        "Hindsight recalls similar previous analysis patterns and stores only "
        "safe summarized memories."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "TruthGuard AI uses Gemini API for generative analysis, trained local ensemble ML models as fallback, "
    "and optional Hindsight memory for recalling previous analysis patterns. Results are assistive "
    "and should be verified from trusted sources."
)