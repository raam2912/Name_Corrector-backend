from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import os
from dotenv import load_dotenv
import datetime
import re
import json
import asyncio 
import logging
from functools import wraps
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time
import secrets
import hmac
import psutil
from collections import Counter
import sys 

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field 
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser

# ReportLab imports for PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('numerology_app.log'),
        logging.StreamHandler(sys.stdout) # Correctly use StreamHandler for sys.stdout
    ]
)

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__) 
logger.info("Flask app instance created.") # Log app creation
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'numerology-secret-key-2024')
app.config['CACHE_TYPE'] = 'simple'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300

# Initialize extensions (cache and limiter)
cache = None
limiter = None
try:
    cache = Cache(app)
    limiter = Limiter(
        app,
        default_limits=["200 per day", "50 per hour"],
        storage_uri=os.getenv('REDIS_URL', 'memory://')
    )
    logger.info("Flask-Caching and Flask-Limiter initialized successfully.")
except Exception as e:
    logger.warning(f"Could not initialize Redis features (Caching/Limiter might be unavailable): {e}")

# CORS Configuration
CORS(app, resources={r"/*": {"origins": [
    "https://namecorrectionsheelaa.netlify.app",
    "http://localhost:3000",
    "https://*.netlify.app"
], "supports_credentials": True}})

logger.info("CORS configured for the Flask app.")

# REFINED PROMPTS FOR NUMEROLOGY APPLICATION
# Updated to align with the PDF's detailed requirements for report generation and validation
NAME_SUGGESTION_SYSTEM_PROMPT = """You are Sheelaa's Elite AI Numerology Assistant and Master Name Strategist. Your expertise lies in creating numerologically aligned name variations that preserve cultural authenticity while optimizing energetic outcomes.

## YOUR MISSION:
Generate 3-5 strategically crafted full name variations that:
- Maintain maximum similarity to the original name (90%+ resemblance)
- Align precisely with target Expression Numbers
- Directly support the user's specific desired outcome
- Sound natural, culturally appropriate, and **highly practical for real-world use**

## MODIFICATION GUIDELINES:
- **Minimal Changes Only**: Single letter alterations, spelling variations, middle initial additions/removals
- **Preserve Core Identity**: Keep the essence and pronunciation as close as possible
- **Cultural Sensitivity**: Ensure variations respect the original name's cultural context
- **Practicality First**: Prioritize names that are easy to adopt and integrate into daily life. Avoid overly complex or unusual suggestions unless specifically requested.

## RATIONALE REQUIREMENTS:
For each suggestion, provide a comprehensive 2-3 sentence explanation that:
1. **Specifies the exact numerological advantage** of the new Expression Number
2. **Directly connects** this advantage to the user's desired outcome
3. **Explains the energetic transformation** this change creates
4. **Emphasizes positive impact** and specific benefits

**When crafting rationales, consider these additional factors from the client's profile:**
- **Lo Shu Grid Balance**: How the new name helps balance missing energies (e.g., if 5 is missing, does the new name's Expression 5 help?)
- **Planetary Compatibility**: How the new name's planetary ruler (from Chaldean mapping) aligns with their Ascendant/Lagna or avoids conflicts with malefic planetary lords.
- **Phonetic Harmony**: Confirm the suggested name maintains a pleasant and strong vibrational tone.
- **Edge Case Resolutions**: If the original profile had an exact edge case (e.g., Expression 8/Life Path 1 conflict), explain how the new name resolves or mitigates it.

## OUTPUT FORMAT:
Return a valid JSON object conforming to NameSuggestionsOutput schema with accurate expression_number calculations.

## QUALITY STANDARDS:
- Each rationale must be substantive, specific, and compelling
- Avoid generic explanations - make each one unique and targeted
- If a target number cannot be achieved while maintaining similarity, acknowledge this in reasoning""" + "{parser_instructions}"

NAME_SUGGESTION_HUMAN_PROMPT = """**NUMEROLOGICAL NAME OPTIMIZATION REQUEST**
**Original Name:** "{original_full_name}"
**Desired Life Outcome:** "{desired_outcome}"
**Target Expression Numbers:** {target_expression_numbers}

**TASK:** Create 3-5 name variations that are nearly identical to the original but numerologically optimized for the desired outcome. Each suggestion must include a detailed, specific rationale explaining its advantages.

**REQUIREMENTS:**
- Maintain 90%+ similarity to original name
- Target the specified Expression Numbers
- Provide compelling, detailed explanations for each suggestion
- Ensure cultural appropriateness, natural sound, and **high practicality for adoption**"""

ADVANCED_REPORT_SYSTEM_PROMPT = """You are Sheelaa's Elite AI Numerology Assistant, renowned for delivering transformative, deeply personalized numerological insights. You create comprehensive reports that combine ancient wisdom with modern psychological understanding, **integrating Chaldean Numerology, Lo Shu Grid analysis, Astro-Numerology principles (based on provided data), and Phonology considerations.**

## REPORT STRUCTURE (Follow Exactly):
### 1. **Introduction** (Warm & Personal)
- Acknowledge their name, birth date, and desired outcome personally
- Create immediate connection and establish trust
- Set positive, empowering tone for the entire report

### 2. **Core Numerological Blueprint** (Detailed Analysis)
- **Expression Number Deep Dive**: Core traits, psychological patterns, natural abilities, **and its Chaldean planetary association.**
- **Life Path Number Analysis**: Life purpose, karmic direction, soul mission.
- **Birth Day Number Significance**: Explain the day of birth's influence.
- **Compatibility Synthesis**: How these numbers work together or challenge each other.
- Draw from provided interpretations but add unique insights.

### 3. **Soul Architecture** (Soul Urge & Personality)
- **Soul Urge Number**: Inner motivations, heart's desires, what fulfills them.
- **Personality Number**: How others perceive them, outer persona, first impressions.
- **Integration Dynamics**: How inner and outer selves align or conflict.

### 4. **🌟 STRATEGIC NAME CORRECTIONS** (CRITICAL SECTION)
- Present each suggested name with its Expression Number.
- **Use the exact detailed rationales provided in the input data.**
- Explain how each suggestion specifically enhances their desired outcome.
- Compare energetic shifts from current to suggested numbers.
- Provide implementation guidance and timing recommendations.
- **Crucially, discuss how each suggested name addresses specific numerological or astrological imbalances identified in the client's profile (e.g., balancing missing Lo Shu numbers, harmonizing with planetary influences).**

### 5. **Karmic Lessons & Soul Work** (Transformational)
- Detail missing numbers from their **Lo Shu Grid** and their significance.
- Explain how addressing these lessons accelerates growth.
- Connect lessons to their desired outcome.
- **Specifically mention any karmic debt numbers identified in their birth date or original name.**

### 6. **Temporal Numerology** (Timing & Cycles)
- Current Personal Year energy and optimal activities.
- Best months for implementing changes (e.g., name changes, new ventures).
- Energy forecasts and strategic timing.

### 7. **Shadow Work & Growth Edge** (Honest Assessment)
- Potential challenges specific to their numbers, **including any identified edge cases (e.g., Expression 8/Life Path 1 conflict, Master Number challenges).**
- Practical mitigation strategies.
- Transform challenges into growth opportunities.

### 8. **Personalized Development Blueprint** (Actionable)
- Immediate focus areas with specific actions.
- Long-term vision aligned with their numbers.
- Monthly practices for sustained growth.
- **Suggest actions that help integrate energies from their Lo Shu Grid or align with beneficial planetary influences.**

### 9. **Future Cycles Forecast** (3-Year Vision)
- Year-by-year energy themes and focus areas.
- Strategic planning for optimal outcomes.
- Timeline for major life decisions.

### 10. **Numerical Uniqueness Profile** (Special Recognition)
- Highlight rare combinations or master numbers.
- Explain their unique gifts and cosmic significance.
- **Discuss how their unique numerical blueprint interacts with their conceptual astrological parameters (Ascendant/Lagna, Moon Sign).**

### 11. **Empowerment Conclusion** (Inspiring Close)
- Synthesize key insights.
- Reinforce their power to shape destiny.
- Leave them feeling inspired and capable.

## WRITING STANDARDS:
- **Depth Over Breadth**: Elaborate significantly on each section, providing rich, nuanced insights.
- **Personal Touch**: Use their name throughout, make it feel custom-crafted.
- **Markdown Formatting**: Use headers, bold text, bullet points for readability.
- **Empowering Tone**: Inspiring, supportive, never fatalistic.
- **Specific Details**: Avoid generalities, provide concrete insights, **referencing specific numbers, grid positions, or planetary influences where relevant.**
- **Professional Quality**: No AI disclaimers, no sensitive information.

## SUCCESS METRICS:
The report should feel like a personal consultation worth hundreds of dollars, providing actionable insights that directly address their desired outcome while honoring the sacred wisdom of numerology."""

ADVANCED_REPORT_HUMAN_PROMPT = """**COMPREHENSIVE NUMEROLOGY REPORT REQUEST**
Please generate a detailed, transformational numerology report using this complete profile data:
```json
{llm_input_data}
```
**REQUIREMENTS:**
- Follow the exact 11-section structure outlined in your instructions
- Elaborate extensively on each section - aim for comprehensive analysis
- Use the provided detailed rationales for name suggestions verbatim
- Directly address how everything connects to their desired outcome
- Create a report that feels personally crafted and professionally valuable"""

NAME_VALIDATION_SYSTEM_PROMPT = """You are Sheelaa's Elite AI Numerology Assistant, specializing in precise numerological name validation and strategic guidance through a conversational interface. Your goal is to help practitioners thoroughly evaluate potential names by asking clarifying questions and providing iterative insights.

## CONVERSATION GUIDELINES:
- **Start with an acknowledgement**: Confirm the name to be validated and the original profile context.
- **Iterative Questioning**: If you need more information to provide a comprehensive validation, ask specific, open-ended follow-up questions. Prioritize questions that will yield the most impactful numerological insights. Examples of practitioner-centric questions:
    - "Considering the client's Lo Shu Grid, how does the suggested name's Expression Number (and its associated Chaldean planetary influence) specifically address or balance any missing numbers or over-represented energies in their birth chart? For instance, if a '5' is missing for adaptability, does this name help to integrate that energy?"
    - "Given the client's conceptual astrological placements (Ascendant/Lagna ruler, Moon Sign, and any dominant benefic/malefic planetary lords from their birth date), how does the planetary ruler of this suggested name's Expression Number interact with these influences? Are there any potential harmonies or conflicts we should be aware of?"
    - "Does this suggested name help to mitigate or transform the energy of any identified karmic debt numbers (e.g., 13, 14, 16, 19) present in the client's original birth date or current name?"
    - "From a phonetic or 'pronology' perspective, what is the vibrational quality of this suggested name? Does it feel strong, harmonious, or are there any sounds that might create subtle dissonance or enhance specific qualities related to the client's desired outcome?"
    - "Beyond the general desired outcome of '{desired_outcome}', could you elaborate on the *specific* nuances or sub-goals the client has in mind for this name change? For example, is 'success' primarily financial, related to leadership, public recognition, or a more internal sense of achievement?"
    - "Are there any cultural or family traditions that might influence the acceptance or impact of this suggested name? How important is maintaining a close phonetic or structural resemblance to the original name for the client?"
    - "If the client has a known numerological 'edge case' (e.g., Expression 8 / Life Path 1 conflict, or challenges with a Master Number), how is this specific suggested name designed to address or mitigate that particular challenge?"
- **Provide Partial Insights**: Even if you need more information, offer initial observations or partial insights based on the data you have.
- **Comprehensive Validation on Sufficient Data**: Once you feel you have enough information OR the practitioner explicitly asks for a **'final validation report for [Suggested Name]'**, you MUST provide a full, structured validation report in Markdown format, following the "VALIDATION FRAMEWORK" below. State clearly when you are providing the final validation and why (e.g., 'Based on our discussion, here is the final validation report for [Suggested Name]:').
- **Maintain Context**: Remember the original profile and the suggested name throughout the conversation.
- **Empowering Tone**: Be supportive and informative.

## VALIDATION FRAMEWORK (Use this structure for the final comprehensive validation):
### **Analysis Structure:**
#### 1. **Suggested Name Numerological Profile**
- Expression Number and its core energetic signature, **including its Chaldean planetary association.**
- Psychological traits and natural tendencies.
- Career and relationship implications.
- Spiritual significance and growth potential.
- **Comment on its phonetic vibration and ease of adoption.**
#### 2. **Outcome Alignment Assessment** (Critical)
- **Direct Correlation**: How the suggested name's energy specifically supports their desired outcome.
- **Energy Shift Analysis**: Compare current vs. suggested name's energetic influence.
- **Manifestation Potential**: Rate the name's power to attract their desired results.
- **Obstacle Clearing**: How this name helps overcome current limitations.
- **How does it specifically address the client's desired profession/vocation?**
#### 3. **Life Path & Core Compatibility Matrix**
- **Synergy Score**: How well the new Expression Number harmonizes with their Life Path and Birth Day Number.
- **Challenge Areas**: Potential friction points and how to navigate them.
- **Amplification Effects**: Ways the combination enhances their natural gifts.
- **Balance Dynamics**: How this pairing creates equilibrium or tension.
- **Critically, assess its compatibility with their Ascendant/Lagna ruler and Moon Sign, and whether it avoids conflict with any planetary lords.**
- **Evaluate its impact on their Lo Shu Grid balance (e.g., does it introduce a missing number or reinforce an over-represented one?).**
#### 4. **Strategic Recommendation** (Clear Verdict)
- **Rating Scale**: "Exceptional Fit" | "Strong Alignment" | "Moderate Benefit" | "Requires Consideration" | "Not Recommended"
- **Confidence Level**: High/Medium/Low with reasoning.
- **Implementation Timeline**: When and how to make the change.
- **Expected Outcomes**: Realistic timeline for seeing results.
- **Provide specific guidance on how to maximize the benefits of this name, considering all numerological and conceptual astrological factors.**

## EVALUATION CRITERIA:
- **Numerical Harmony**: Mathematical compatibility between numbers.
- **Energetic Resonance**: How well the vibrations support their goals.
- **Practical Viability**: Real-world considerations for name change, **including phonetic ease and cultural fit.**
- **Timing Considerations**: Current life cycles and optimal implementation.
- **Holistic Alignment**: How the name integrates with their entire numerological and conceptual astrological profile.

## OUTPUT STANDARDS:
- **Markdown Formatting**: Clear headers and structured presentation for final reports.
- **Specific Evidence**: Reference exact numerical relationships, **Lo Shu grid observations, and conceptual planetary influences.**
- **Balanced Perspective**: Honest assessment including any concerns.
- **Actionable Guidance**: Concrete next steps and recommendations.
- **Encouraging Tone**: Supportive while being truthfully analytical.
- **NO AI DISCLAIMERS.**
"""
GENERAL_CHAT_SYSTEM_PROMPT = """You are Sheelaa's Elite AI Numerology Assistant - a wise, knowledgeable, and approachable guide in the sacred science of numbers.

## YOUR ROLE:
- **Numerology Expert**: Provide accurate information about numerological principles
- **Supportive Guide**: Offer gentle, encouraging guidance without being prescriptive
- **Educational Resource**: Explain concepts clearly for all knowledge levels
- **Boundary Keeper**: Redirect for personalized calculations requiring specific data

## RESPONSE GUIDELINES:
- **Concise & Helpful**: Keep responses focused and valuable
- **Warm Personality**: Maintain Sheelaa's caring, intuitive energy
- **Educational Focus**: Teach numerological principles and concepts
- **Ethical Boundaries**: No specific predictions, only guidance and insights

## WHAT YOU CAN HELP WITH:
- General numerology education and principles
- Explaining number meanings and significance
- Discussing numerological concepts and history
- Providing general guidance about name energy
- Answering questions about numerological practices

## WHAT REQUIRES SPECIALIZED SERVICE:
- Personal number calculations (need full name + birth date)
- Specific name suggestions or validations
- Comprehensive reports or detailed analysis
- PDF generation or advanced consultations

## COMMUNICATION STYLE:
- Professional yet warm and accessible
- Encouraging and empowering
- Clear explanations without overwhelming detail
- Respectful of numerological traditions
When users need personalized services, guide them toward providing complete information for proper analysis."""


# --- Data Models ---
class NumberType(Enum):
    EXPRESSION = "expression"
    LIFE_PATH = "life_path"
    SOUL_URGE = "soul_urge"
    PERSONALITY = "personality"

@dataclass
class NumerologyDetails:
    full_name: str
    birth_date: str
    expression_number: int
    life_path_number: int
    soul_urge_number: Optional[int] = None
    personality_number: Optional[int] = None

@dataclass
class ValidationRequest:
    original_full_name: str
    birth_date: str
    desired_outcome: str
    suggested_name: str

@dataclass
class ReportRequest:
    full_name: str
    birth_date: str
    birth_time: Optional[str] = None # Added birth_time
    birth_place: Optional[str] = None # Added birth_place
    desired_outcome: str
    current_expression_number: int
    current_life_path_number: int

class NameSuggestion(BaseModel):
    name: str = Field(description="Full name suggestion")
    rationale: str = Field(description="Detailed explanation (2-3 sentences minimum) about why this specific name change is advantageous based on its new numerological Expression Number and how it directly supports the user's desired outcome. Focus on the positive impact and the specific energy it brings.")
    expression_number: int = Field(description="The calculated Expression Number for the suggested name.")

class NameSuggestionsOutput(BaseModel):
    suggestions: List[NameSuggestion] = Field(description="List of name suggestions with their rationales and expression numbers.")
    reasoning: str = Field(description="Overall reasoning for the types of names suggested.")

# --- Advanced Numerology Calculator ---
class AdvancedNumerologyCalculator:
    # 1️⃣ CHALDEAN LETTER-TO-NUMBER MAPPING WITH PLANETS (from PDF)
    CHALDEAN_NUMEROLOGY_MAP = {
        'A': (1, 'Sun (☉)'), 'J': (1, 'Sun (☉)'), 'S': (1, 'Sun (☉)'),
        'B': (2, 'Moon (☽)'), 'K': (2, 'Moon (☽)'), 'T': (2, 'Moon (☽)'),
        'C': (3, 'Jupiter (♃)'), 'G': (3, 'Jupiter (♃)'), 'L': (3, 'Jupiter (♃)'), 'U': (3, 'Jupiter (♃)'),
        'D': (4, 'Rahu (☊)'), 'M': (4, 'Rahu (☊)'), 'V': (4, 'Rahu (☊)'),
        'E': (5, 'Mercury (☿)'), 'H': (5, 'Mercury (☿)'), 'N': (5, 'Mercury (☿)'), 'X': (5, 'Mercury (☿)'),
        'F': (6, 'Venus (♀)'), 'O': (6, 'Venus (♀)'), 'W': (6, 'Venus (♀)'),
        'P': (7, 'Ketu / Neptune (☋)'), 'Z': (7, 'Ketu / Neptune (☋)'),
        'I': (8, 'Saturn (♄)'), 'R': (8, 'Saturn (♄)'),
        # Critical Rule: Number 9 is NEVER used in letter assignments.
    }
    # Separate mapping for direct number-to-planet if needed for non-letter derived numbers
    # This maps 9 to Mars, as it's a common association in numerology for the number 9 itself.
    NUMBER_TO_PLANET_MAP = {
        1: 'Sun (☉)', 2: 'Moon (☽)', 3: 'Jupiter (♃)', 4: 'Rahu (☊)',
        5: 'Mercury (☿)', 6: 'Venus (♀)', 7: 'Ketu / Neptune (☋)', 8: 'Saturn (♄)',
        9: 'Mars (♂)', # 9 is often associated with Mars in some systems
        11: 'Higher Moon / Spiritual Insight', 22: 'Higher Rahu / Master Builder', 33: 'Higher Jupiter / Master Healer'
    }
    
    VOWELS = set('AEIOU')
    CONSONANTS = set('BCDFGHJKLMNPQRSTVWXYZ')
    KARMIC_DEBT_NUMBERS = {13, 14, 16, 19}
    MASTER_NUMBERS = {11, 22, 33} # PDF specifies 11, 22, 33

    # Enhanced interpretations with psychological depth
    NUMEROLOGY_INTERPRETATIONS = {
        1: {
            "core": "Leadership, independence, new beginnings, drive, and ambition. It empowers you to forge your own path and initiate new ventures with confidence.",
            "psychological": "Natural born leaders with strong individualistic tendencies. They thrive on challenges and pioneering new paths.",
            "career": "Excellent in executive roles, entrepreneurship, and any field requiring innovation and leadership.",
            "relationships": "Independent partners who need space but are fiercely loyal. They inspire others through example.",
            "challenges": "May struggle with collaboration and can be overly aggressive or impatient.",
            "spiritual": "Learning to balance self-reliance with healthy interdependence."
        },
        2: {
            "core": "Cooperation, balance, diplomacy, harmony, and partnership. It fosters strong relationships, intuition, and a gentle, supportive nature. Number 2 is the peacemaker, symbolizing duality, grace, and tact. It represents sensitivity, empathy, and the ability to work well with others. Those influenced by 2 are often mediators, seeking equilibrium and understanding. They excel in collaborative efforts and bring a calming presence to any situation.",
            "psychological": "Natural mediators with high emotional intelligence and sensitivity to others' needs.",
            "career": "Excel in counseling, diplomacy, team coordination, and supportive roles.",
            "relationships": "Devoted partners who create harmony but may sacrifice their own needs.",
            "challenges": "Tendency toward indecision and over-sensitivity to criticism.",
            "spiritual": "Learning to value their own needs while serving others."
        },
        3: {
            "core": "Creativity, self-expression, communication, optimism, and joy. It enhances social interactions, artistic pursuits, and a vibrant outlook on life. Number 3 is the number of expression, inspiration, and growth. It signifies imagination, enthusiasm, and a natural talent for communication. Individuals with a strong 3 influence are often charismatic, artistic, and enjoy being in the spotlight. They bring light and joy to others through their creative endeavors and optimistic spirit.",
            "psychological": "Natural entertainers and communicators with infectious enthusiasm and artistic flair.",
            "career": "Thrive in creative fields, media, entertainment, marketing, and any role involving communication.",
            "relationships": "Charming and social partners who bring joy but may struggle with depth and commitment.",
            "challenges": "Tendency to scatter energy and avoid difficult emotions through superficiality.",
            "spiritual": "Learning to channel creative gifts for higher purpose and authentic expression."
        },
        4: {
            "core": "Stability, diligent hard work, discipline, organization, and building strong foundations for lasting security. It signifies reliability and a practical approach. Number 4 is the builder, representing order, structure, and practicality. It is associated with responsibility, honesty, and a strong work ethic. Those influenced by 4 are often methodical, reliable, and persistent. They excel at creating systems, managing resources, and ensuring long-term security through diligent effort.",
            "psychological": "Methodical planners who value structure, reliability, and systematic approaches to life.",
            "career": "Excel in project management, finance, construction, systems analysis, and administrative roles.",
            "relationships": "Dependable partners who provide security but may struggle with spontaneity and emotional expression.",
            "challenges": "Tendency toward rigidity, resistance to change, and workaholism.",
            "spiritual": "Learning to embrace flexibility while maintaining their natural gift for creating order."
        },
        5: {
            "core": "Freedom, dynamic change, adventure, versatility, and adaptability. It encourages embracing new experiences, travel, and a love for personal liberty. Number 5 is the number of change, symbolizing curiosity, progress, and a desire for new horizons. It represents adaptability, resourcelessness, and a love for exploration. Individuals with a strong 5 influence are often restless, seeking variety and excitement. They thrive on challenges and are quick to embrace new opportunities, often leading to diverse life experiences.",
            "psychological": "Natural explorers with insatiable curiosity and need for variety and stimulation.",
            "career": "Thrive in travel, sales, media, technology, and any field offering variety and change.",
            "relationships": "Exciting partners who bring adventure but may struggle with commitment and routine.",
            "challenges": "Restlessness, difficulty with commitment, and tendency toward excess or addiction.",
            "spiritual": "Learning to find freedom within commitment and depth within exploration."
        },
        6: {
            "core": "Responsibility, nurturing, harmony, selfless service, and love. It fosters deep connections in family and community, embodying care and compassion. Number 6 is the number of harmony and domesticity, symbolizing love, empathy, and service to others. It is associated with responsibility, protection, and a strong sense of community. Those influenced by 6 are often caregivers, dedicated to their family and friends. They find fulfillment in supporting others and creating a peaceful, loving environment.",
            "psychological": "Natural caregivers with strong sense of responsibility and desire to heal and help others.",
            "career": "Excel in healthcare, education, counseling, social work, and family business.",
            "relationships": "Devoted partners who create beautiful homes but may become overly controlling or sacrificial.",
            "challenges": "Tendency toward martyrdom, interference in others' lives, and neglecting self-care.",
            "spiritual": "Learning to serve others while maintaining healthy boundaries and self-love."
        },
        7: {
            "core": "Spirituality, deep introspection, analytical thought, wisdom, and inner truth. It encourages seeking knowledge, solitude, and understanding deeper mysteries.",
            "psychological": "Natural researchers and philosophers with deep need for understanding and spiritual connection.",
            "career": "Excel in research, science, spirituality, psychology, and any field requiring deep analysis.",
            "relationships": "Loyal but private partners who need space for reflection and may struggle with emotional intimacy.",
            "challenges": "Tendency toward isolation, over-analysis, and difficulty with practical matters.",
            "spiritual": "Learning to share their wisdom while maintaining their need for solitude and reflection."
        },
        8: {
            "core": "Abundance, power, material success, leadership, and executive ability. It signifies achievement, financial gain, and capacity to manage large endeavors.",
            "psychological": "Natural executives with strong ambition and ability to manifest material success through strategic thinking.",
            "career": "Excel in business, finance, law, real estate, and positions of authority and influence.",
            "relationships": "Powerful partners who provide well but may prioritize success over emotional connection.",
            "challenges": "Tendency toward materialism, power struggles, and neglecting spiritual/emotional needs.",
            "spiritual": "Learning to use power and success in service of higher purpose and community good."
        },
        9: {
            "core": "Humanitarianism, compassion, completion, universal love, and wisdom. It represents selfless nature, broad perspective, and culmination of experiences.",
            "psychological": "Natural healers and teachers with broad perspective and deep compassion for humanity.",
            "career": "Excel in humanitarian work, teaching, healing arts, and any field serving the greater good.",
            "relationships": "Compassionate partners with high ideals who may struggle with disappointment and letting go.",
            "challenges": "Tendency toward emotional volatility, disappointment in others, and difficulty with boundaries.",
            "spiritual": "Learning to maintain compassion while developing practical wisdom and emotional resilience."
        },
        11: {
            "core": "Heightened intuition, spiritual insight, illumination, and inspiration. Master number signifying powerful ability to inspire and lead through spiritual understanding.",
            "psychological": "Highly sensitive visionaries with strong intuitive abilities and calling to inspire others.",
            "career": "Excel in spiritual teaching, counseling, artistic expression, and roles requiring inspiration and vision.",
            "relationships": "Deeply intuitive partners who seek soulmate connections but may struggle with hypersensitivity.",
            "challenges": "Nervous tension, high expectations of self and others, and feeling misunderstood.",
            "spiritual": "Learning to ground their high vibration while serving as a bridge between spiritual and material worlds."
        },
        22: {
            "core": "Master Builder, signifying large-scale achievement, practical idealism, and ability to manifest grand visions into reality.",
            "psychological": "Visionary builders who combine intuition with practical skills to create lasting impact.",
            "career": "Excel in architecture, large-scale business, international affairs, and transformational leadership.",
            "relationships": "Visionary partners with big dreams who may struggle with finding equally ambitious companions.",
            "challenges": "Overwhelming sense of responsibility, tendency toward perfectionism, and potential for burnout.",
            "spiritual": "Learning to balance their grand vision with practical steps and personal relationships."
        },
        33: {
            "core": "Master Healer/Teacher, embodying compassionate service, universal love, and profound spiritual guidance dedicated to humanity.",
            "psychological": "Natural healers and teachers with exceptional capacity for love and service to others.",
            "career": "Excel in healing professions, spiritual teaching, artistic expression, and humanitarian leadership.",
            "relationships": "Deeply loving partners who give unconditionally but may attract those who drain their energy.",
            "challenges": "Emotional overwhelm, taking on others' pain, and tendency toward self-sacrifice.",
            "spiritual": "Learning to heal and teach while maintaining energetic boundaries and self-care."
        }
    }

    @staticmethod
    def reduce_number(num: int, allow_master_numbers: bool = True) -> int:
        """
        Enhanced number reduction with master number preservation (11, 22, 33).
        If allow_master_numbers is True, 11, 22, 33 are returned as is.
        Otherwise, they are reduced to single digits (2, 4, 6 respectively).
        """
        if allow_master_numbers and num in AdvancedNumerologyCalculator.MASTER_NUMBERS:
            return num
        
        # If not a master number or master numbers are not allowed, reduce to single digit
        while num > 9:
            num = sum(int(digit) for digit in str(num))
        return num
    
    @staticmethod
    def calculate_expression_number(name: str) -> Tuple[int, Dict]:
        """
        Calculate expression number (Chaldean system) with detailed breakdown and planetary association.
        Number 9 is NEVER used in Chaldean letter assignments.
        """
        total = 0
        letter_breakdown = {}
        cleaned_name = ''.join(filter(str.isalpha, name)).upper()
        
        for letter in cleaned_name:
            if letter in AdvancedNumerologyCalculator.CHALDEAN_NUMEROLOGY_MAP:
                value, planet = AdvancedNumerologyCalculator.CHALDEAN_NUMEROLOGY_MAP[letter]
                total += value
                # Store the value and planet for each letter
                if letter not in letter_breakdown:
                    letter_breakdown[letter] = {"value": value, "planet": planet, "count": 0}
                letter_breakdown[letter]["count"] += 1
        
        # Expression number is usually reduced to a single digit, unless it's a Master Number (11, 22, 33)
        # The PDF implies Expression number is reduced unless it's a Master Number,
        # but the Chaldean system itself does not assign 9.
        # For the final reduced number, we use the general NUMBER_TO_PLANET_MAP.
        reduced = AdvancedNumerologyCalculator.reduce_number(total, True) # Allow master numbers for final expression
        
        return reduced, {
            "total_before_reduction": total,
            "letter_breakdown": letter_breakdown,
            "is_master_number": reduced in AdvancedNumerologyCalculator.MASTER_NUMBERS,
            "karmic_debt": total in AdvancedNumerologyCalculator.KARMIC_DEBT_NUMBERS,
            "planetary_ruler": AdvancedNumerologyCalculator.NUMBER_TO_PLANET_MAP.get(reduced, 'N/A')
        }
    
    @staticmethod
    def calculate_soul_urge_number(name: str) -> Tuple[int, Dict]:
        """Calculate soul urge (vowels only) using Chaldean map."""
        total = 0
        vowel_breakdown = {}
        cleaned_name = ''.join(filter(str.isalpha, name)).upper()
        
        for letter in cleaned_name:
            if letter in AdvancedNumerologyCalculator.VOWELS and letter in AdvancedNumerologyCalculator.CHALDEAN_NUMEROLOGY_MAP:
                value, _ = AdvancedNumerologyCalculator.CHALDEAN_NUMEROLOGY_MAP[letter]
                total += value
                vowel_breakdown[letter] = vowel_breakdown.get(letter, 0) + value
        
        reduced = AdvancedNumerologyCalculator.reduce_number(total, True) # Soul Urge can be a Master Number
        
        return reduced, {
            "total_before_reduction": total,
            "vowel_breakdown": vowel_breakdown,
            "is_master_number": reduced in AdvancedNumerologyCalculator.MASTER_NUMBERS
        }
    
    @staticmethod
    def calculate_personality_number(name: str) -> Tuple[int, Dict]:
        """Calculate personality number (consonants only) using Chaldean map."""
        total = 0
        consonant_breakdown = {}
        cleaned_name = ''.join(filter(str.isalpha, name)).upper()
        
        for letter in cleaned_name:
            if letter in AdvancedNumerologyCalculator.CONSONANTS and letter in AdvancedNumerologyCalculator.CHALDEAN_NUMEROLOGY_MAP:
                value, _ = AdvancedNumerologyCalculator.CHALDEAN_NUMEROLOGY_MAP[letter]
                total += value
                consonant_breakdown[letter] = consonant_breakdown.get(letter, 0) + value
        
        reduced = AdvancedNumerologyCalculator.reduce_number(total, True) # Personality can be a Master Number
        
        return reduced, {
            "total_before_reduction": total,
            "consonant_breakdown": consonant_breakdown,
            "is_master_number": reduced in AdvancedNumerologyCalculator.MASTER_NUMBERS
        }

    @staticmethod
    def calculate_life_path_number(birth_date_str: str) -> Tuple[int, Dict]:
        """
        Calculate Life Path Number (Destiny Number) by adding all digits of complete birth date.
        Master Numbers (11, 22, 33) are NOT reduced.
        """
        try:
            year, month, day = map(int, birth_date_str.split('-'))
        except ValueError:
            raise ValueError("Invalid birth date format. Please use YYYY-MM-DD.")

        karmic_debt_present = []
        
        # Check for karmic debt in the original numbers before reduction
        if month in AdvancedNumerologyCalculator.KARMIC_DEBT_NUMBERS:
            karmic_debt_present.append(month)
        if day in AdvancedNumerologyCalculator.KARMIC_DEBT_NUMBERS:
            karmic_debt_present.append(day)
        if year in AdvancedNumerologyCalculator.KARMIC_DEBT_NUMBERS: # Check year as a whole
             karmic_debt_present.append(year)

        # Reduce month, day, and year components, preserving master numbers
        month_reduced = AdvancedNumerologyCalculator.reduce_number(month, True)
        day_reduced = AdvancedNumerologyCalculator.reduce_number(day, True) 
        year_reduced = AdvancedNumerologyCalculator.reduce_number(year, True)
        
        total = month_reduced + day_reduced + year_reduced
        final_number = AdvancedNumerologyCalculator.reduce_number(total, True) # Final reduction for Life Path, preserving master numbers
        
        return final_number, {
            "month": month,
            "day": day, 
            "year": year,
            "month_reduced": month_reduced,
            "day_reduced": day_reduced,
            "year_reduced": year_reduced,
            "total_before_final_reduction": total,
            "karmic_debt_numbers": list(set(karmic_debt_present)), # Ensure unique karmic debts
            "is_master_number": final_number in AdvancedNumerologyCalculator.MASTER_NUMBERS
        }

    @staticmethod
    def calculate_birth_day_number(birth_date_str: str) -> Tuple[int, Dict]:
        """
        Calculates the Birth Day Number (Primary Ruling Number) from the day of birth.
        Master Numbers (11, 22, 33) are NOT reduced.
        Example: 25th = 2+5=7. 11th = 11.
        """
        if not birth_date_str:
            return 0, {"error": "No birth date provided."}
        try:
            day = int(birth_date_str.split('-')[2])
            if not (1 <= day <= 31):
                raise ValueError("Day must be between 1 and 31.")
            
            reduced_day = AdvancedNumerologyCalculator.reduce_number(day, True) # Preserve master numbers
            
            return reduced_day, {
                "original_day": day,
                "is_master_number": reduced_day in AdvancedNumerologyCalculator.MASTER_NUMBERS
            }
        except (ValueError, IndexError) as e:
            logger.error(f"Error calculating Birth Day Number for {birth_date_str}: {e}")
            return 0, {"error": f"Invalid birth day: {e}"}

    # 3️⃣ CORE NUMBERS FROM BIRTH (Lo Shu Grid)
    @staticmethod
    def calculate_lo_shu_grid(birth_date_str: str) -> Dict:
        """
        Calculates the Lo Shu Grid digit counts from a birth date.
        Identifies missing numbers and provides basic interpretations.
        """
        try:
            dob_digits = [int(d) for d in birth_date_str.replace('-', '') if d.isdigit()]
        except ValueError:
            raise ValueError("Invalid birth date format for Lo Shu Grid. Please use YYYY-MM-DD.")
        
        grid_counts = Counter(dob_digits)
        
        missing_numbers = sorted([i for i in range(1, 10) if i not in grid_counts])
        
        missing_impact = {
            1: "Challenges with independence or self-assertion. Needs to develop leadership.",
            2: "Difficulties with cooperation or sensitivity. Needs to foster diplomacy.",
            3: "Challenges in self-expression or creativity. Needs to communicate more.",
            4: "Potential issues with stability, discipline, or practical matters. Needs structure.",
            5: "Lack of grounding, adaptability, or a need for more freedom. Needs versatility.",
            6: "Harmony issues, challenges with responsibility or nurturing. Needs to serve and care.",
            7: "May indicate a need for deeper introspection or spiritual understanding. Needs wisdom.",
            8: "Potential struggles with material abundance or executive ability. Needs power and organization.",
            9: "Challenges with humanitarianism, compassion, or completion. Needs universal love."
        }
        
        detailed_missing_lessons = [
            {"number": num, "impact": missing_impact.get(num, "General energy imbalance.")}
            for num in missing_numbers
        ]

        return {
            "grid_counts": dict(grid_counts),
            "missing_numbers": missing_numbers,
            "missing_lessons": detailed_missing_lessons,
            "has_5": 5 in grid_counts,
            "has_6": 6 in grid_counts,
            "has_3": 3 in grid_counts,
            "has_8": 8 in grid_counts # For Expression 8 validation
        }

    # 5️⃣ VALIDATION LOGIC FOR EXPRESSION NUMBER (from PDF with refinements)
    @staticmethod
    def check_expression_validity(expression_number: int, life_path_number: int, lo_shu_grid: Dict, profession_desire: str, birth_day_number: int, astro_info: Dict) -> Dict:
        """
        Validates the Expression Number based on comprehensive rules from the PDF.
        Returns a dictionary indicating validity and any flags/recommendations.
        """
        is_valid = True
        flags = []
        recommendation = "Generally harmonious."
        
        # General Meaning and Chaldean Planetary Association
        # Numbers 1, 3, 5, 6 are generally considered ideal for Expression.
        # Numbers 2, 7 are spiritual/sensitive.
        # Numbers 4, 8, 9 are considered karmic/challenging and need careful validation.

        # Rule 1: General Expression Number Suitability
        if expression_number in {1, 3, 5, 6}:
            pass # These are generally favorable
        elif expression_number in {2, 7}:
            # Spiritual, sensitive, analytical - only if Lo Shu Grid has 5 & 6 and profession aligns
            if not (lo_shu_grid.get('has_5') and lo_shu_grid.get('has_6')):
                is_valid = False
                flags.append(f"Expression {expression_number} (Spiritual/Sensitive) may lack grounding or practicality without 5 (adaptability) and 6 (harmony/responsibility) in Lo Shu Grid.")
                recommendation = "Consider a name change to a more grounding number if Lo Shu grid is not supportive."
            
            # Check profession alignment for 2, 7
            profession_lower = profession_desire.lower()
            spiritual_professions = ["healer", "psychologist", "spiritual", "teacher", "counselor", "artist", "researcher"]
            if not any(p in profession_lower for p in spiritual_professions):
                flags.append(f"Expression {expression_number} (Spiritual/Sensitive) may not align with a purely material or leadership-focused desired outcome.")
        
        elif expression_number in {4, 8, 9}:
            # Heavy karma, delays, chaos - use only if validated astrologically (conceptual)
            flags.append(f"Expression {expression_number} (Karmic/Challenging) numbers often bring lessons related to hard work, power struggles, or humanitarian service. Astrological validation is highly recommended.")
            recommendation = "A name change to a more harmonious number is often beneficial unless strongly supported by astrology and the individual's life path."

        # Rule 2: Master Numbers (11, 22, 33)
        if expression_number in AdvancedNumerologyCalculator.MASTER_NUMBERS:
            # Only retain if Life Path or Birth Day Number supports them
            if life_path_number not in AdvancedNumerologyCalculator.MASTER_NUMBERS and expression_number != life_path_number:
                flags.append(f"Master Expression {expression_number} without a supporting Master Life Path can be challenging to embody fully, potentially reducing its energy to its single-digit sum (e.g., 11 to 2).")
                recommendation = "The energy of this Master Number may feel overwhelming or difficult to ground without strong alignment in other core numbers. Focus on grounding practices."
            if birth_day_number not in AdvancedNumerologyCalculator.MASTER_NUMBERS and expression_number != birth_day_number:
                 flags.append(f"Master Expression {expression_number} without a supporting Master Birth Day Number might indicate a struggle to fully integrate its higher vibrations into daily life.")


        # Rule 3: Lo Shu Grid Balance (from PDF)
        if not lo_shu_grid.get('has_5') and expression_number == 5:
            flags.append("Caution: Assigning Expression 5 when 5 is missing from Lo Shu Grid might make integration of adaptability and freedom challenging.")
        if not lo_shu_grid.get('has_6') and expression_number == 6:
            flags.append("Caution: Assigning Expression 6 when 6 is missing from Lo Shu Grid might make integration of harmony and responsibility challenging.")
        if not lo_shu_grid.get('has_3') and expression_number == 3:
            flags.append("Caution: Assigning Expression 3 when 3 is missing from Lo Shu Grid might make integration of creativity and communication challenging.")
        # If 8 is missing from Lo Shu Grid, Expression 8 is problematic
        if not lo_shu_grid.get('has_8') and expression_number == 8:
            flags.append("Significant Caution: Expression 8 (Saturn) is highly problematic if 8 is missing from Lo Shu Grid, indicating a lack of foundational energy for material success and power.")
            is_valid = False
            recommendation = "Strongly advise against Expression 8 if 8 is missing from Lo Shu Grid. It can lead to severe financial or power struggles."


        # Rule 4: Astrological Compatibility (Conceptual, from PDF)
        planetary_compatibility_flags = astro_info.get('planetary_compatibility', {}).get('compatibility_flags', [])
        if planetary_compatibility_flags:
            flags.extend(planetary_compatibility_flags)
            # If any significant conflict flags are present, mark as not valid
            if any("conflict" in f.lower() or "caution" in f.lower() or "problematic" in f.lower() for f in planetary_compatibility_flags):
                is_valid = False
                recommendation = "Review astrological compatibility. The suggested name's planetary energy may conflict with your birth chart's dominant influences."

        return {
            "is_valid": is_valid,
            "flags": list(set(flags)), # Remove duplicates
            "recommendation": recommendation
        }

    # 6️⃣ EDGE CASE HANDLING (from PDF with refinements)
    @staticmethod
    def analyze_edge_cases(profile_data: Dict) -> List[Dict]:
        """
        Identifies specific challenging numerological combinations and provides insights,
        now incorporating conceptual astrological data and Lo Shu Grid.
        """
        edge_cases = []

        expression_number = profile_data.get('expression_number')
        life_path_number = profile_data.get('life_path_number')
        birth_day_number = profile_data.get('birth_day_number')
        lo_shu_grid = profile_data.get('lo_shu_grid', {})
        profession_desire = profile_data.get('profession_desire', '').lower()
        astro_info = {
            "ascendant_info": profile_data.get('ascendant_info', {}),
            "moon_sign_info": profile_data.get('moon_sign_info', {}),
            "planetary_lords": profile_data.get('planetary_lords', []),
            "planetary_compatibility": profile_data.get('planetary_compatibility', {})
        }

        # Expression = 8, Life Path = 1 (Conflict: Saturn vs Sun) (from PDF)
        if expression_number == 8 and life_path_number == 1:
            edge_cases.append({
                "type": "Expression 8 / Life Path 1 Conflict (Saturn vs. Sun)",
                "description": "A clash between Saturn's discipline (8) and Sun's leadership (1) can lead to internal friction or external power struggles. The 8 may bring delays or heavy lessons to the pioneering 1, especially in areas of authority and recognition.",
                "resolution_guidance": "A name change to Expression 1, 3, 5, or 6 is often recommended to harmonize with the Life Path 1 energy. Focus on balanced leadership, integrity, and avoiding materialism. Cultivate patience and humility."
            })
        
        # Master Number Expression but LP = 4 (Master vs Builder) (from PDF)
        if expression_number in AdvancedNumerologyCalculator.MASTER_NUMBERS and life_path_number == 4:
            edge_cases.append({
                "type": f"Master Expression {expression_number} / Life Path 4 Challenge (Vision vs. Structure)",
                "description": f"The expansive, visionary energy of Master Number {expression_number} can feel constrained or frustrated by the practical, structured nature of Life Path 4. It's a call to ground big visions into tangible reality, which can be a heavy task and may lead to burnout if not managed.",
                "resolution_guidance": "Focus on disciplined manifestation. If the Master energy feels overwhelming, it may reduce to a single digit (e.g., 11 to 2, 22 to 4, 33 to 6) which might be more harmonious with the LP 4, offering a more practical path. Break down large goals into small, actionable steps."
            })

        # 4 in Expression and Birth Day Number = 4 (Too much karma/rigidity) (from PDF)
        if expression_number == 4 and birth_day_number == 4:
             edge_cases.append({
                "type": "Double 4 (Expression & Birth Day Number - Amplified Karma)",
                "description": "Having both Expression and Birth Day Numbers as 4 can amplify the karmic lessons associated with hard work, discipline, and potential rigidity. It suggests a life path heavily focused on building and structure, which can sometimes feel burdensome and limit flexibility.",
                "resolution_guidance": "It's crucial to balance hard work with self-care and flexibility. Seek opportunities for creative expression and avoid becoming too rigid or overwhelmed by responsibility. Astrological validation is highly recommended to understand specific challenges and periods of intense effort."
            })

        # Lo Shu Grid lacks 5 or 6 (Avoid assigning name #5 or #6 to Expression if missing) (from PDF)
        if not lo_shu_grid.get('has_5') and expression_number == 5:
            edge_cases.append({
                "type": "Expression 5 with Missing 5 in Lo Shu Grid (Adaptability Challenge)",
                "description": "A missing 5 in your birth date (Lo Shu Grid) indicates a need to develop adaptability and freedom. Assigning Expression 5 might make it challenging to fully integrate this energy, potentially leading to instability or restlessness.",
                "resolution_guidance": "Focus on developing inner flexibility and grounding practices. While the name aims for adaptability, the innate lack of 5 may require conscious effort to embrace change positively."
            })
        if not lo_shu_grid.get('has_6') and expression_number == 6:
            edge_cases.append({
                "type": "Expression 6 with Missing 6 in Lo Shu Grid (Harmony/Responsibility Challenge)",
                "description": "A missing 6 in your birth date (Lo Shu Grid) can indicate challenges with harmony, responsibility, or nurturing. Assigning Expression 6 might make it difficult to fully embody its compassionate and service-oriented traits.",
                "resolution_guidance": "Focus on developing compassion and responsibility through conscious effort. The name can support this, but foundational work on these themes may be necessary."
            })
        
        # Profession: Healer, Teacher (Accept 2, 7 only if supported by grid) (from PDF)
        spiritual_professions = ["healer", "teacher", "counselor", "artist", "researcher", "spiritual guide"]
        if any(p in profession_desire for p in spiritual_professions):
            if expression_number in {2, 7}:
                if not (lo_shu_grid.get('has_5') and lo_shu_grid.get('has_6')):
                    edge_cases.append({
                        "type": f"Spiritual Profession / Expression {expression_number} with Unsupportive Lo Shu Grid",
                        "description": f"While Expression {expression_number} aligns with your desired spiritual profession, the missing 5 (grounding) or 6 (harmony/service) in your Lo Shu Grid might make it challenging to fully ground and manifest these energies practically.",
                        "resolution_guidance": "Focus on practical application of your spiritual gifts and building stable foundations. Consider names that reinforce grounding numbers if appropriate, or consciously develop the missing energies."
                    })

        # Karmic Debt Numbers in Birth Date or Original Name
        original_karmic_debts = profile_data.get('life_path_details', {}).get('karmic_debt_numbers', [])
        original_name_expression_total = profile_data.get('expression_details', {}).get('total_before_reduction')
        if original_name_expression_total in AdvancedNumerologyCalculator.KARMIC_DEBT_NUMBERS:
            original_karmic_debts.append(original_name_expression_total)
        
        if original_karmic_debts:
            unique_debts = list(set(original_karmic_debts))
            edge_cases.append({
                "type": f"Karmic Debt Numbers Present: {', '.join(map(str, unique_debts))}",
                "description": f"These numbers indicate specific life lessons or challenges to overcome. The energy of these debts can manifest as recurring patterns or obstacles until the lessons are learned.",
                "resolution_guidance": "Consciously work on the lessons associated with each karmic debt number (e.g., 13 for hard work, 14 for adaptability, 16 for humility, 19 for independence). A name change can help mitigate but not eliminate these core lessons."
            })

        # Astrological Edge Cases (Conceptual, from PDF)
        expression_planet = AdvancedNumerologyCalculator.NUMBER_TO_PLANET_MAP.get(expression_number)
        ascendant_ruler = astro_info.get('ascendant_info', {}).get('ruler')
        planetary_lords = astro_info.get('planetary_lords', [])

        # Example: Expression 8 (Saturn) vs. Sun-ruled Ascendant (Leo) - from PDF
        if expression_number == 8 and ascendant_ruler == 'Sun (☉)':
            edge_cases.append({
                "type": "Expression 8 vs. Sun-ruled Ascendant Conflict",
                "description": f"Your Expression Number 8 (ruled by Saturn) may create friction with your Sun-ruled Ascendant ({astro_info.get('ascendant_info',{}).get('sign', 'N/A')}). This can manifest as struggles between ambition and personal freedom, or delays in leadership roles. Saturn's restrictive nature can challenge the Sun's expansive drive.",
                "resolution_guidance": "Focus on disciplined effort without rigidity. Cultivate patience and ensure your actions are aligned with your true self, not just material gain. Consider names with Expression 1, 3, 5, or 6 if seeking more direct alignment with solar energy."
            })
        
        # Example: Expression 4 (Rahu) vs. Jupiter-ruled Ascendant (Sagittarius/Pisces) - from PDF
        if expression_number == 4 and ascendant_ruler == 'Jupiter (♃)':
            edge_cases.append({
                "type": "Expression 4 vs. Jupiter-ruled Ascendant Caution",
                "description": f"Expression 4 (Rahu) can bring unconventional paths and sudden changes, which might challenge the wisdom and expansion of a Jupiter-ruled Ascendant ({astro_info.get('ascendant_info',{}).get('sign', 'N/A')}). This combination requires careful navigation of material pursuits and spiritual growth.",
                "resolution_guidance": "Embrace structure and practicality (4) but ensure it serves a higher purpose (Jupiter). Avoid impulsive decisions and seek spiritual guidance to navigate challenges. A name with Expression 3 or 9 might offer more harmonious expansion."
            })

        # Example: Debilitated Saturn in conceptual chart and Expression 8 - from PDF
        if expression_number == 8 and any(p['planet'] == 'Saturn (♄)' and p['nature'] == 'Malefic' for p in planetary_lords):
            edge_cases.append({
                "type": "Expression 8 with Challenged Saturn Influence",
                "description": "If your conceptual chart indicates a challenging Saturn influence, an Expression Number of 8 can amplify its difficult aspects, potentially leading to increased delays, obstacles, or feelings of burden related to career and material success. This combination demands immense patience and integrity.",
                "resolution_guidance": "Focus on integrity, patience, and selfless service. Avoid shortcuts and practice detachment from outcomes. Consider names with Expression 1, 3, 5, or 6 to balance this energy, unless your chart strongly supports a powerful 8."
            })

        # Example: Favored Jupiter in conceptual chart and Expression 3 - from PDF
        if expression_number == 3 and any(p['planet'] == 'Jupiter (♃)' and p['nature'] == 'Benefic' for p in planetary_lords):
            edge_cases.append({
                "type": "Expression 3 with Favored Jupiter Influence",
                "description": "An Expression Number of 3 (ruled by Jupiter) combined with a favored Jupiter in your conceptual chart can significantly amplify creativity, optimism, and opportunities for growth, especially in communication, teaching, and expansion. This is a very positive alignment.",
                "resolution_guidance": "Embrace opportunities for self-expression and knowledge sharing. Maintain focus to avoid scattering your energies. Your natural optimism and wisdom will be greatly enhanced. Channel this energy into constructive and uplifting endeavors."
            })
        
        # Add other planetary conflicts/supports as per PDF's "Planetary Conflicts and Support" section
        # If chart favors Mars, a name totaling 9 may amplify positivity.
        if expression_number == 9 and any(p['planet'] == 'Mars (♂)' and p['nature'] == 'Benefic' for p in planetary_lords):
            edge_cases.append({
                "type": "Expression 9 with Favored Mars Influence",
                "description": "Expression 9 (Mars) is amplified by a favored Mars in your conceptual chart, enhancing your drive, courage, and capacity for humanitarian action. This combination supports strong leadership in service.",
                "resolution_guidance": "Channel your energy into purposeful action and avoid aggression. Use your drive for the greater good."
            })
        # If chart has malefic Mars, avoid 9 (from PDF)
        if expression_number == 9 and any(p['planet'] == 'Mars (♂)' and p['nature'] == 'Malefic' for p in planetary_lords):
            edge_cases.append({
                "type": "Expression 9 with Malefic Mars Influence",
                "description": "Expression 9 (Mars) may bring aggressive energy, impulsiveness, or conflict if your conceptual chart indicates a malefic Mars. This can make humanitarian efforts challenging.",
                "resolution_guidance": "Cultivate patience and diplomacy. Focus on constructive outlets for your energy and avoid confrontations. A name with a different Expression number might be more harmonious."
            })

        # If chart favors Venus, a name totaling 6 may amplify positivity.
        if expression_number == 6 and any(p['planet'] == 'Venus (♀)' and p['nature'] == 'Benefic' for p in planetary_lords):
            edge_cases.append({
                "type": "Expression 6 with Favored Venus Influence",
                "description": "Expression 6 (Venus) is amplified by a favored Venus in your conceptual chart, enhancing harmony, creativity, and love in relationships and domestic life. This is highly beneficial for personal and family well-being.",
                "resolution_guidance": "Embrace your nurturing qualities and artistic talents. Create beauty and harmony in your surroundings."
            })
        # If chart has malefic Venus, avoid 6 (implied from PDF's general rule of avoiding conflicts)
        if expression_number == 6 and any(p['planet'] == 'Venus (♀)' and p['nature'] == 'Malefic' for p in planetary_lords):
            edge_cases.append({
                "type": "Expression 6 with Malefic Venus Influence",
                "description": "Expression 6 (Venus) may bring challenges in relationships or domestic harmony if your conceptual chart indicates a malefic Venus. This can lead to difficulties in nurturing or finding balance.",
                "resolution_guidance": "Focus on self-love and setting healthy boundaries in relationships. Avoid codependency. A name with a different Expression number might be more supportive."
            })

        # If chart favors Mercury, a name totaling 5 may amplify positivity.
        if expression_number == 5 and any(p['planet'] == 'Mercury (☿)' and p['nature'] == 'Benefic' for p in planetary_lords):
            edge_cases.append({
                "type": "Expression 5 with Favored Mercury Influence",
                "description": "Expression 5 (Mercury) is amplified by a favored Mercury in your conceptual chart, enhancing communication, adaptability, and intellectual pursuits. This supports versatility and quick thinking.",
                "resolution_guidance": "Embrace new experiences and opportunities for learning. Use your communication skills effectively. Avoid scattering your energy too widely."
            })
        # If chart has malefic Mercury, avoid 5 (implied from PDF's general rule of avoiding conflicts)
        if expression_number == 5 and any(p['planet'] == 'Mercury (☿)' and p['nature'] == 'Malefic' for p in planetary_lords):
            edge_cases.append({
                "type": "Expression 5 with Malefic Mercury Influence",
                "description": "Expression 5 (Mercury) may bring instability, restlessness, or communication challenges if your conceptual chart indicates a malefic Mercury. This can lead to difficulty focusing.",
                "resolution_guidance": "Practice grounding and focus. Ensure clear and concise communication. A name with a different Expression number might provide more stability."
            })

        # If chart favors Moon, a name totaling 2 or 7 may amplify positivity.
        if expression_number in [2,7] and any(p['planet'] == 'Moon (☽)' and p['nature'] == 'Benefic' for p in planetary_lords):
            edge_cases.append({
                "type": f"Expression {expression_number} with Favored Moon Influence",
                "description": f"Expression {expression_number} (Moon/Ketu) is amplified by a favored Moon in your conceptual chart, enhancing intuition, emotional balance, and spiritual insight. This supports deep connections and inner peace.",
                "resolution_guidance": "Trust your intuition and nurture your emotional well-being. Embrace your sensitive nature as a strength."
            })
        # If chart has malefic Moon, avoid 2 or 7 (implied from PDF's general rule of avoiding conflicts)
        if expression_number in [2,7] and any(p['planet'] == 'Moon (☽)' and p['nature'] == 'Malefic' for p in planetary_lords):
            edge_cases.append({
                "type": f"Expression {expression_number} with Malefic Moon Influence",
                "description": f"Expression {expression_number} (Moon/Ketu) may bring emotional volatility, mood swings, or difficulties in relationships if your conceptual chart indicates a malefic Moon. This can impact inner peace.",
                "resolution_guidance": "Practice emotional regulation and self-care. Seek stability in your environment. A name with a different Expression number might offer more emotional resilience."
            })

        # If chart favors Sun, a name totaling 1 may amplify positivity.
        if expression_number == 1 and any(p['planet'] == 'Sun (☉)' and p['nature'] == 'Benefic' for p in planetary_lords):
            edge_cases.append({
                "type": "Expression 1 with Favored Sun Influence",
                "description": "Expression 1 (Sun) is amplified by a favored Sun in your conceptual chart, enhancing leadership qualities, vitality, and self-confidence. This is highly beneficial for pioneering and achieving recognition.",
                "resolution_guidance": "Embrace your leadership potential and shine brightly. Inspire others through your actions."
            })
        # If chart has malefic Sun, avoid 1 (implied from PDF's general rule of avoiding conflicts)
        if expression_number == 1 and any(p['planet'] == 'Sun (☉)' and p['nature'] == 'Malefic' for p in planetary_lords):
            edge_cases.append({
                "type": "Expression 1 with Malefic Sun Influence",
                "description": "Expression 1 (Sun) may bring ego challenges, arrogance, or difficulties in asserting yourself if your conceptual chart indicates a malefic Sun. This can impact your leadership journey.",
                "resolution_guidance": "Practice humility and self-awareness. Focus on service-oriented leadership rather than personal glory. A name with a different Expression number might offer a more balanced approach."
            })


        return edge_cases

    # 7️⃣ ASTROLOGICAL INTEGRATION (Conceptual, based on PDF)
    @staticmethod
    def get_ascendant_info(birth_date_str: str, birth_time_str: Optional[str], birth_place: Optional[str]) -> Dict:
        """
        Conceptual: Simulates fetching Ascendant/Lagna info based on birth date and time.
        This is a simplified model and does not perform actual astrological calculations.
        Based on "Hour: Determines dominant planetary influence at birth" from PDF.
        """
        logger.warning("Astrological integration for Ascendant is conceptual. Providing simplified data.")
        
        ascendant_sign = "Not Calculated (Time/Place Needed)"
        ascendant_ruler = "Not Calculated"

        if not birth_time_str or not birth_place:
            return {"sign": ascendant_sign, "ruler": ascendant_ruler, "notes": "Birth time and place are needed for conceptual Ascendant calculation."}

        try:
            # Parse birth time (HH:MM)
            birth_hour = int(birth_time_str.split(':')[0])
            
            # Simplified conceptual mapping of hour to Ascendant sign and ruler
            # This is a very rough approximation and not real astrology.
            # The mapping tries to loosely follow a diurnal cycle for planet rulership.
            if 0 <= birth_hour < 2: # ~Midnight to 2 AM
                ascendant_sign = "Capricorn" # Saturn
            elif 2 <= birth_hour < 4: # ~2 AM to 4 AM
                ascendant_sign = "Aquarius" # Saturn / Rahu
            elif 4 <= birth_hour < 6: # ~4 AM to 6 AM (Pre-dawn)
                ascendant_sign = "Pisces" # Jupiter / Ketu
            elif 6 <= birth_hour < 8: # ~6 AM to 8 AM (Sunrise)
                ascendant_sign = "Aries" # Mars
            elif 8 <= birth_hour < 10: # ~8 AM to 10 AM
                ascendant_sign = "Taurus" # Venus
            elif 10 <= birth_hour < 12: # ~10 AM to Noon
                ascendant_sign = "Gemini" # Mercury
            elif 12 <= birth_hour < 14: # ~Noon to 2 PM
                ascendant_sign = "Cancer" # Moon
            elif 14 <= birth_hour < 16: # ~2 PM to 4 PM
                ascendant_sign = "Leo" # Sun
            elif 16 <= birth_hour < 18: # ~4 PM to 6 PM
                ascendant_sign = "Virgo" # Mercury
            elif 18 <= birth_hour < 20: # ~6 PM to 8 PM (Sunset)
                ascendant_sign = "Libra" # Venus
            elif 20 <= birth_hour < 22: # ~8 PM to 10 PM
                ascendant_sign = "Scorpio" # Mars
            else: # 22 <= birth_hour < 24 (~10 PM to Midnight)
                ascendant_sign = "Sagittarius" # Jupiter

            # Assign ruler based on conceptual sign
            ruler_map = {
                "Aries": "Mars (♂)", "Taurus": "Venus (♀)", "Gemini": "Mercury (☿)",
                "Cancer": "Moon (☽)", "Leo": "Sun (☉)", "Virgo": "Mercury (☿)",
                "Libra": "Venus (♀)", "Scorpio": "Mars (♂)", "Sagittarius": "Jupiter (♃)",
                "Capricorn": "Saturn (♄)", "Aquarius": "Saturn (♄)", "Pisces": "Jupiter (♃)"
            }
            ascendant_ruler = ruler_map.get(ascendant_sign, "Mixed")

        except (ValueError, TypeError):
            logger.warning("Invalid birth time format for conceptual ascendant calculation.")
            return {"sign": "Error", "ruler": "Error", "notes": "Invalid birth time format. Use HH:MM."}

        return {"sign": ascendant_sign, "ruler": ascendant_ruler, "notes": "Conceptual calculation based on birth hour. For precise astrological readings, consult a professional astrologer."}

    @staticmethod
    def get_moon_sign_info(birth_date_str: str, birth_time_str: Optional[str], birth_place: Optional[str]) -> Dict:
        """
        Conceptual: Simulates fetching Moon Sign info.
        This is a simplified model and does not perform actual astrological calculations.
        Based on "Rashi (Moon Sign): Influences name number compatibility" from PDF.
        """
        logger.warning("Astrological integration for Moon Sign is conceptual. Providing simplified data.")
        moon_sign = "Not Calculated"
        moon_ruler = "N/A"
        notes = "Conceptual calculation based on birth day. For precise astrological readings, consult a professional astrologer."

        try:
            day = int(birth_date_str.split('-')[2])
            # Very simplistic mapping for demonstration, aims for some variety
            if 1 <= day <= 5:
                moon_sign = "Aries" # Mars
                moon_ruler = "Mars (♂)"
            elif 6 <= day <= 10:
                moon_sign = "Taurus" # Venus
                moon_ruler = "Venus (♀)"
            elif 11 <= day <= 15:
                moon_sign = "Gemini" # Mercury
                moon_ruler = "Mercury (☿)"
            elif 16 <= day <= 20:
                moon_sign = "Cancer" # Moon
                moon_ruler = "Moon (☽)"
            elif 21 <= day <= 25:
                moon_sign = "Leo" # Sun
                moon_ruler = "Sun (☉)"
            else: # 26 <= day <= 31
                moon_sign = "Virgo" # Mercury
                moon_ruler = "Mercury (☿)"
        except (ValueError, IndexError):
            notes = "Birth date is needed for conceptual Moon Sign calculation."
            pass # Keep default unknown
        return {"sign": moon_sign, "ruler": moon_ruler, "notes": notes}

    @staticmethod
    def get_planetary_lords(birth_date_str: str, birth_time_str: Optional[str], birth_place: Optional[str]) -> List[Dict]:
        """
        Conceptual: Simulates fetching dominant planetary lords (benefic/malefic).
        This is a simplified model and does not perform actual astrological calculations.
        Based on "Planetary Conflicts and Support" section from PDF.
        """
        logger.warning("Astrological integration for Planetary Lords is conceptual. Providing simplified data.")
        
        lords = []
        try:
            month = int(birth_date_str.split('-')[1])
            day = int(birth_date_str.split('-')[2])
            
            # Conceptual benefic/malefic based on month and day (very rough approximation)
            # This aims to provide some dynamic 'benefic' or 'malefic' planets for testing.
            
            # Base lords (some always present for demo)
            lords.append({"planet": "Sun (☉)", "nature": "Neutral"})
            lords.append({"planet": "Moon (☽)", "nature": "Neutral"})
            lords.append({"planet": "Mercury (☿)", "nature": "Neutral"})
            lords.append({"planet": "Venus (♀)", "nature": "Neutral"})
            lords.append({"planet": "Mars (♂)", "nature": "Neutral"})
            lords.append({"planet": "Jupiter (♃)", "nature": "Neutral"})
            lords.append({"planet": "Saturn (♄)", "nature": "Neutral"})
            lords.append({"planet": "Rahu (☊)", "nature": "Neutral"})
            lords.append({"planet": "Ketu / Neptune (☋)", "nature": "Neutral"})

            # Introduce benefic/malefic based on date for demo purposes
            if month in [4, 8, 12]: # Water signs (Cancer, Scorpio, Pisces)
                # Moon and Jupiter tend to be benefic here
                for p in lords:
                    if p['planet'] == 'Moon (☽)' or p['planet'] == 'Jupiter (♃)':
                        p['nature'] = 'Benefic'
                # Mars can be malefic
                for p in lords:
                    if p['planet'] == 'Mars (♂)':
                        p['nature'] = 'Malefic'
            elif month in [1, 5, 9]: # Fire signs (Aries, Leo, Sagittarius)
                # Sun and Mars tend to be benefic
                for p in lords:
                    if p['planet'] == 'Sun (☉)' or p['planet'] == 'Mars (♂)':
                        p['nature'] = 'Benefic'
                # Saturn can be malefic
                for p in lords:
                    if p['planet'] == 'Saturn (♄)':
                        p['nature'] = 'Malefic'
            else: # Earth/Air signs (Taurus, Gemini, Virgo, Libra, Capricorn, Aquarius)
                # Mercury and Venus tend to be benefic
                for p in lords:
                    if p['planet'] == 'Mercury (☿)' or p['planet'] == 'Venus (♀)':
                        p['nature'] = 'Benefic'
                # Rahu/Ketu can be malefic
                for p in lords:
                    if p['planet'] == 'Rahu (☊)' or p['planet'] == 'Ketu / Neptune (☋)':
                        p['nature'] = 'Malefic'
            
            # Specific conditions from PDF for debilitated/favored
            # "If your chart shows a debilitated Saturn, then avoid name numbers totaling 8."
            # Conceptual: If day is a multiple of 4, make Saturn malefic
            if day % 4 == 0:
                for p in lords:
                    if p['planet'] == 'Saturn (♄)':
                        p['nature'] = 'Malefic'
            
            # "If your chart favors Jupiter, a name totaling 3 may amplify positivity."
            # Conceptual: If day is a multiple of 3, make Jupiter benefic
            if day % 3 == 0:
                for p in lords:
                    if p['planet'] == 'Jupiter (♃)':
                        p['nature'] = 'Benefic'

            # Filter out neutral ones if they haven't been assigned benefic/malefic
            lords = [p for p in lords if p['nature'] != 'Neutral']

        except (ValueError, IndexError):
            logger.warning(f"Could not parse birth date for conceptual planetary lords: {birth_date_str}")
            lords = [{"planet": "Mixed", "nature": "Neutral", "notes": "Invalid birth date format for conceptual planetary lords."}] # Default if date parsing fails

        return lords

    @staticmethod
    def check_planetary_compatibility(expression_number: int, astro_info: Dict) -> Dict:
        """
        Checks compatibility between expression number's planetary ruler and conceptual astrological info.
        Based on "Influence on Name Correction" and "Match Name Number → Planet" from PDF.
        """
        expression_planet = AdvancedNumerologyCalculator.NUMBER_TO_PLANET_MAP.get(expression_number)
        
        compatibility_flags = []
        
        ascendant_info = astro_info.get('ascendant_info', {})
        planetary_lords = astro_info.get('planetary_lords', [])

        ascendant_ruler = ascendant_info.get('ruler')

        # Rule 1: Harmony with Ascendant's Ruling Planet (from PDF)
        # Ruling Planet Favorable Numbers (from PDF)
        favorable_map_by_ascendant_ruler = {
            'Sun (☉)': [1],
            'Moon (☽)': [2], # PDF says "favor 7 if spiritual", but 2 is primary.
            'Jupiter (♃)': [3],
            'Rahu (☊)': [4], # PDF says "Handle with caution"
            'Mercury (☿)': [5],
            'Venus (♀)': [6],
            'Ketu / Neptune (☋)': [7],
            'Saturn (♄)': [8], # PDF says "Only if well-placed in chart"
            'Mars (♂)': [9] # PDF says "Aggressive energy; avoid unless supported"
        }

        # Specific conflicts/favored numbers based on Ascendant Ruler (from PDF)
        if ascendant_ruler and ascendant_ruler != "Not Calculated":
            # Check for favorable alignment
            if expression_number in favorable_map_by_ascendant_ruler.get(ascendant_ruler, []):
                # Special notes for 8, 4, 9 even if favorable, as per PDF's cautions
                if (expression_number == 8 and ascendant_ruler == 'Saturn (♄)') or \
                   (expression_number == 4 and ascendant_ruler == 'Rahu (☊)') or \
                   (expression_number == 9 and ascendant_ruler == 'Mars (♂)'):
                    compatibility_flags.append(f"Expression {expression_number} ({expression_planet}) aligns with {ascendant_ruler}-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}), but requires careful consideration or strong chart support as per numerological tradition.")
                else:
                    compatibility_flags.append(f"Expression {expression_number} ({expression_planet}) harmonizes well with your Ascendant ruler ({ascendant_ruler}), amplifying positive traits.")
            
            # Check for specific conflicts mentioned in PDF
            if ascendant_ruler == 'Sun (☉)' and expression_number in [8, 4]:
                compatibility_flags.append(f"Conflict: Expression {expression_number} ({expression_planet}) is generally avoided with a Sun-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}) due to energetic friction.")
            if ascendant_ruler == 'Moon (☽)' and expression_number == 8:
                compatibility_flags.append(f"Conflict: Expression {expression_number} (Saturn) is generally avoided with a Moon-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}).")
            if ascendant_ruler == 'Jupiter (♃)' and expression_number in [4, 8]:
                compatibility_flags.append(f"Caution: Expression {expression_number} ({expression_planet}) may conflict with a Jupiter-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}) unless specifically supported by the chart.")
            if ascendant_ruler == 'Mercury (☿)' and expression_number in [4, 8]:
                compatibility_flags.append(f"Caution: Expression {expression_number} ({expression_planet}) may conflict with a Mercury-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}).")
            if ascendant_ruler == 'Venus (♀)' and expression_number == 8:
                compatibility_flags.append(f"Caution: Expression {expression_number} (Saturn) may conflict with a Venus-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}).")
            if ascendant_ruler == 'Ketu / Neptune (☋)' and expression_number == 9:
                 compatibility_flags.append(f"Caution: Expression {expression_number} (Mars) may conflict with a Ketu/Neptune-ruled Ascendant ({ascendant_info.get('sign', 'N/A')}).")
            
            # General non-alignment (if not covered by specific conflicts/favors)
            if expression_planet and ascendant_ruler != 'Mixed' and expression_number not in favorable_map_by_ascendant_ruler.get(ascendant_ruler, []) and \
               not ((ascendant_ruler == 'Sun (☉)' and expression_number in [8, 4]) or \
                    (ascendant_ruler == 'Moon (☽)' and expression_number == 8) or \
                    (ascendant_ruler == 'Jupiter (♃)' and expression_number in [4, 8]) or \
                    (ascendant_ruler == 'Mercury (☿)' and expression_number in [4, 8]) or \
                    (ascendant_ruler == 'Venus (♀)' and expression_number == 8) or \
                    (ascendant_ruler == 'Ketu / Neptune (☋)' and expression_number == 9)):
                compatibility_flags.append(f"Expression {expression_number} ({expression_planet}) may not have direct harmony with your Ascendant ruler ({ascendant_ruler}).")


        # Rule 2: Planetary Conflicts/Support from conceptual chart (from PDF)
        # "If your chart shows a debilitated Saturn, then avoid name numbers totaling 8."
        if expression_number == 8 and any(p['planet'] == 'Saturn (♄)' and p['nature'] == 'Malefic' for p in planetary_lords):
            compatibility_flags.append("Caution: Expression 8 (Saturn) may be challenging if your conceptual chart indicates a debilitated Saturn, potentially amplifying obstacles.")
        
        # "If your chart favors Jupiter, a name totaling 3 may amplify positivity."
        if expression_number == 3 and any(p['planet'] == 'Jupiter (♃)' and p['nature'] == 'Benefic' for p in planetary_lords):
            compatibility_flags.append("Strong alignment: Expression 3 (Jupiter) is amplified by a favored Jupiter in your conceptual chart, enhancing creativity and wisdom.")

        # Other planetary rule checks (derived from the spirit of the PDF's rules)
        if expression_number == 9 and any(p['planet'] == 'Mars (♂)' and p['nature'] == 'Benefic' for p in planetary_lords):
            compatibility_flags.append("Strong alignment: Expression 9 (Mars) is amplified by a favored Mars, enhancing courage and drive for humanitarian action.")
        if expression_number == 9 and any(p['planet'] == 'Mars (♂)' and p['nature'] == 'Malefic' for p in planetary_lords):
            compatibility_flags.append("Caution: Expression 9 (Mars) may bring aggressive or conflict-prone energy if your conceptual chart indicates a malefic Mars.")

        if expression_number == 6 and any(p['planet'] == 'Venus (♀)' and p['nature'] == 'Benefic' for p in planetary_lords):
            compatibility_flags.append("Strong alignment: Expression 6 (Venus) is amplified by a favored Venus, enhancing harmony, love, and artistic expression.")
        if expression_number == 6 and any(p['planet'] == 'Venus (♀)' and p['nature'] == 'Malefic' for p in planetary_lords):
            compatibility_flags.append("Caution: Expression 6 (Venus) may bring challenges in relationships or domestic life if your conceptual chart indicates a malefic Venus.")

        if expression_number == 5 and any(p['planet'] == 'Mercury (☿)' and p['nature'] == 'Benefic' for p in planetary_lords):
            compatibility_flags.append("Strong alignment: Expression 5 (Mercury) is amplified by a favored Mercury, enhancing communication, adaptability, and intellectual pursuits.")
        if expression_number == 5 and any(p['planet'] == 'Mercury (☿)' and p['nature'] == 'Malefic' for p in planetary_lords):
            compatibility_flags.append("Caution: Expression 5 (Mercury) may bring instability or communication issues if your conceptual chart indicates a malefic Mercury.")

        if expression_number in [2,7] and any(p['planet'] == 'Moon (☽)' and p['nature'] == 'Benefic' for p in planetary_lords):
            compatibility_flags.append(f"Strong alignment: Expression {expression_number} (Moon/Ketu) is amplified by a favored Moon, enhancing intuition and emotional balance.")
        if expression_number in [2,7] and any(p['planet'] == 'Moon (☽)' and p['nature'] == 'Malefic' for p in planetary_lords):
            compatibility_flags.append(f"Caution: Expression {expression_number} (Moon/Ketu) may bring emotional volatility if your conceptual chart indicates a malefic Moon.")

        if expression_number == 1 and any(p['planet'] == 'Sun (☉)' and p['nature'] == 'Benefic' for p in planetary_lords):
            compatibility_flags.append("Strong alignment: Expression 1 (Sun) is amplified by a favored Sun, enhancing leadership and vitality.")
        if expression_number == 1 and any(p['planet'] == 'Sun (☉)' and p['nature'] == 'Malefic' for p in planetary_lords):
            compatibility_flags.append("Caution: Expression 1 (Sun) may bring ego challenges, arrogance, or difficulties in asserting yourself if your conceptual chart indicates a malefic Sun.")


        return {
            "expression_planet": expression_planet,
            "compatibility_flags": list(set(compatibility_flags)) # Remove duplicates
        }

    # 9️⃣ PHONETIC VIBRATION (“PRONOLOGY”) (Conceptual, based on PDF)
    @staticmethod
    def analyze_phonetic_vibration(name: str) -> Dict:
        """
        Simulates phonetic vibration analysis based on PDF rules:
        "Avoid harsh consonant combinations" and "Ensure pleasant vowel flow".
        """
        logger.warning("Phonetic vibration analysis is conceptual. Returning dummy data with more detail.")
        name_lower = name.lower()
        
        vowel_count = sum(1 for char in name_lower if char in 'aeiou')
        consonant_count = sum(1 for char in name_lower if char.isalpha() and char not in 'aeiou')
        
        # Simple detection of "harsh" consonant combinations (conceptual)
        harsh_combinations = ["th", "sh", "ch", "gh", "ph", "ck", "tch", "dge", "ght"] # Example combinations
        harsh_flags = [combo for combo in harsh_combinations if combo in name_lower]

        # Simple check for vowel flow
        vowel_ratio = vowel_count / max(1, len(name_lower.replace(" ", ""))) # Ignore spaces for ratio
        
        vibration_score = 0.7 # Start with a good score
        vibration_notes = []

        if len(name.replace(" ", "")) < 4:
            vibration_notes.append("Very short names may have less complex vibrational patterns.")
        
        if vowel_ratio < 0.3 or vowel_ratio > 0.6: # Too few or too many vowels (conceptual range)
            vibration_notes.append(f"Vowel-to-consonant balance ({vowel_ratio:.2f} vowel ratio) might affect the pleasantness of the flow.")
            vibration_score -= 0.1

        if harsh_flags:
            vibration_notes.append(f"Contains potentially harsh consonant combinations: {', '.join(harsh_flags)}. This might affect the overall sound vibration.")
            vibration_score -= 0.2
        
        # Check for pleasant vowel flow (conceptual)
        # Example: if name has many consecutive identical vowels or very abrupt vowel changes
        if re.search(r'(aa|ee|ii|oo|uu){2,}', name_lower): # Double vowels
             vibration_notes.append("Repeated consecutive vowels might create a drawn-out sound.")
             vibration_score -= 0.05
        
        # General positive note if no issues found
        if not vibration_notes:
            vibration_notes.append("Name appears to have a harmonious phonetic vibration with balanced vowel and consonant flow.")
            vibration_score = 0.9 # Excellent

        return {
            "score": max(0.1, min(1.0, vibration_score)), # Keep score between 0.1 and 1.0
            "notes": list(set(vibration_notes)), # Remove duplicates
            "is_harmonious": vibration_score > 0.65
        }


# --- Security Manager ---
class SecurityManager:
    @staticmethod
    def validate_input_security(data: str) -> bool:
        """Validate input for security threats"""
        dangerous_patterns = [
            r'<script.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'document\.',
            r'window\.'
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                return False
        return True
    
    @staticmethod
    def sanitize_name_input(name: str) -> str:
        """Sanitize name input while preserving valid characters"""
        # Allow only letters, spaces, hyphens, and apostrophes
        sanitized = re.sub(r"[^a-zA-Z\s\-']", "", name)
        # Remove excessive whitespace
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()
        return sanitized[:100]  # Limit length

# --- Message Parser ---
class MessageParser:
    """Handles parsing of different message types from the frontend."""
    
    @staticmethod
    def parse_validation_request(message: str) -> Optional[ValidationRequest]:
        """Parse VALIDATE_NAME_ADVANCED message format."""
        pattern = r"Original Full Name: \"(.*?)\", Birth Date: \"(.*?)\", Desired Outcome: \"(.*?)\", Suggested Name to Validate: \"(.*?)\""
        match = re.search(pattern, message)
        
        if not match:
            return None
        
        return ValidationRequest(
            original_full_name=match.group(1),
            birth_date=match.group(2),
            desired_outcome=match.group(3),
            suggested_name=match.group(4)
        )

    @staticmethod
    def parse_report_request(message: str) -> Optional[ReportRequest]:
        """
        Parse GENERATE_ADVANCED_REPORT message format.
        Updated to capture birth_time and birth_place.
        """
        pattern = r"GENERATE_ADVANCED_REPORT:\s*My full name is \"(.*?)\"\s*and my birth date is \"(.*?)\"\.\s*My current Name \(Expression\) Number is (\d+)\s*and Life Path Number is (\d+)\.\s*My birth time is \"(.*?)\"\s*and my birth place is \"(.*?)\"\.\s*I desire the following positive outcome in my life:\s*\"([\s\S]*?)\"\s*[\.\n]*$"
        match = re.search(pattern, message) 
        
        if not match:
            logger.error(f"Failed to parse report request message: {message}")
            return None
        
        return ReportRequest(
            full_name=match.group(1),
            birth_date=match.group(2),
            current_expression_number=int(match.group(3)),
            current_life_path_number=int(match.group(4)),
            birth_time=match.group(5), 
            birth_place=match.group(6), 
            desired_outcome=match.group(7).strip()
        )

    @staticmethod
    def parse_initial_validation_chat_message(message: str) -> Optional[Dict]:
        """
        Parse INITIATE_VALIDATION_CHAT message format.
        Updated to capture birth_time and birth_place.
        """
        pattern = r"INITIATE_VALIDATION_CHAT: Suggested Name: \"(.*?)\"\. My original profile: Full Name: \"(.*?)\", Birth Date: \"(.*?)\", Birth Time: \"(.*?)\", Birth Place: \"(.*?)\", Desired Outcome: \"(.*?)\"\. My current Expression Number is (\d+) and Life Path Number is (\d+)\. My Birth Number is (\d+)\."
        match = re.search(pattern, message)
        if not match:
            logger.error(f"Failed to parse initial validation chat message: {message}")
            return None
        
        return {
            "suggested_name": match.group(1),
            "original_full_name": match.group(2),
            "birth_date": match.group(3),
            "birth_time": match.group(4), 
            "birth_place": match.group(5), 
            "desired_outcome": match.group(6),
            "current_expression_number": int(match.group(7)),
            "current_life_path_number": int(match.group(8)),
            "current_birth_number": int(match.group(9)) # This is Birth Day Number
        }


# --- Name Suggestion Engine ---
class NameSuggestionEngine:
    """Handles logic for suggesting names and resolving names for calculation."""
    
    @staticmethod
    def resolve_full_name_for_calculation(original_name: str, suggested_name_part: str) -> str:
        """
        Determine the full name to use for calculation based on the suggested name part.
        If suggested_name_part is a single word, it's assumed to replace the first name.
        Otherwise, suggested_name_part is used as the full name.
        """
        original_parts = original_name.strip().split()
        suggested_parts = suggested_name_part.strip().split()
        
        if not suggested_name_part: # If suggested part is empty, return original
            return original_name

        # If suggested_name_part is a single word and original name has multiple parts
        if len(suggested_parts) == 1 and len(original_parts) > 1:
            # Replace the first name with the suggested name, preserving middle/last names
            return suggested_name_part + " " + " ".join(original_parts[1:])
        else:
            # If suggested_name_part is already a full name (multiple words) or original was single word, use it directly
            return suggested_name_part

    @staticmethod
    def determine_target_numbers_for_outcome(desired_outcome: str) -> List[int]:
        """Determine optimal numerology numbers based on desired outcome."""
        outcome_lower = desired_outcome.lower()
        
        target_map = {
            'success': [1, 8, 22],
            'leadership': [1, 8, 22],
            'business': [8, 22, 4],
            'relationships': [2, 6, 11],
            'love': [2, 6, 33],
            'creativity': [3, 11],
            'communication': [3, 5],
            'stability': [4, 22],
            'adventure': [5],
            'freedom': [5],
            'family': [6, 33],
            'spirituality': [7, 11, 33],
            'wisdom': [7, 9],
            'wealth': [8, 22],
            'compassion': [6, 9, 33],
            'healing': [6, 33],
            'inner peace': [2, 7],
            'public recognition': [1, 3, 8],
            'innovation': [1, 5, 11],
            'teaching': [3, 9, 33]
        }
        
        found_targets = []
        for keyword, numbers in target_map.items():
            if keyword in outcome_lower:
                found_targets.extend(numbers)
        
        if not found_targets:
            return [1, 3, 6, 8] # General positive numbers if no specific keywords found
        
        return sorted(list(set(found_targets))) # Return unique sorted target numbers

    @staticmethod
    def generate_name_suggestions(
        llm_instance: ChatGoogleGenerativeAI, 
        original_full_name: str, 
        desired_outcome: str, 
        target_expression_numbers: List[int]
    ) -> NameSuggestionsOutput:
        """
        Generates name suggestions based on desired outcome and target numerology numbers.
        Uses PydanticOutputParser for structured output.
        """
        parser = PydanticOutputParser(pydantic_object=NameSuggestionsOutput)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=NAME_SUGGESTION_SYSTEM_PROMPT.format(
                parser_instructions=parser.get_format_instructions()
            )),
            HumanMessage(content=NAME_SUGGESTION_HUMAN_PROMPT.format(
                original_full_name=original_full_name,
                desired_outcome=desired_outcome,
                target_expression_numbers=target_expression_numbers
            ))
        ])
        
        chain = prompt | llm_instance
        
        response = chain.invoke({
            "original_full_name": original_full_name, 
            "desired_outcome": desired_outcome, 
            "target_expression_numbers": target_expression_numbers 
        })
        
        try:
            parsed_output = parser.parse(response.content)
            return parsed_output
        except Exception as e:
            logger.warning(f"Failed to parse LLM output directly: {e}. Attempting to fix with OutputFixingParser.")
            fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_instance)
            try:
                fixed_output = fixing_parser.parse(response.content)
                return fixed_output
            except Exception as fix_e:
                logger.error(f"Failed to fix LLM output even with OutputFixingParser: {fix_e}")
                return NameSuggestionsOutput(suggestions=[], reasoning="Could not generate name suggestions due to parsing error.")


# --- LLM Manager ---
class LLMManager:
    def __init__(self):
        self.llm = None
        self.creative_llm = None
        self.analytical_llm = None
        self.memory = None
        self.validation_chat_memories = {} 
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def initialize(self) -> bool:
        """Initialize multiple LLM instances for different purposes"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable not found. LLM initialization failed.")
            return False
            
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                convert_system_message_to_human=True 
            )
            
            self.creative_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=api_key,
                temperature=0.9,
                top_p=0.95,
                top_k=60,
                convert_system_message_to_human=True 
            )
            
            self.analytical_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.3,
                top_p=0.8,
                top_k=20,
                convert_system_message_to_human=True 
            )
            
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history", 
                return_messages=True, 
                k=10
            )
            
            logger.info("All LLM instances initialized successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return False

# Global instance of LLM manager
llm_manager = LLMManager()

# --- Core Utility/Helper Functions ---
def performance_monitor(f):
    """Decorator to monitor function performance"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{f.__name__} executed in {execution_time:.3f} seconds")
        return result
    return wrapper

def safe_cache_key(*args, **kwargs):
    """Generate safe cache keys"""
    key_string = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_string.encode()).hexdigest()

def cached_operation(timeout=3600):
    """Caching decorator that works with or without Redis"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if cache is None:
                return f(*args, **kwargs)
            
            cache_key = f"{f.__name__}:{safe_cache_key(*args, **kwargs)}"
            result = cache.get(cache_key)
            
            if result is None:
                result = f(*args, **kwargs)
                cache.set(cache_key, result, timeout=timeout)
            
            return result
        return wrapper
    return decorator

def rate_limited(limit_string="30 per minute"):
    """Rate limiting decorator that works with or without limiter"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if limiter is not None:
                return limiter.limit(limit_string)(f)(*args, **kwargs)
            else:
                return f(*args, **kwargs)
        return wrapper
    return decorator

@cached_operation(timeout=3600)
@performance_monitor
def get_comprehensive_numerology_profile(full_name: str, birth_date: str, birth_time: Optional[str] = None, birth_place: Optional[str] = None, profession_desire: Optional[str] = None) -> Dict:
    """Get comprehensive numerology profile with caching and advanced calculations"""
    calculator = AdvancedNumerologyCalculator()
    
    # Calculate all core numbers
    expression_num, expression_details = calculator.calculate_expression_number(full_name)
    life_path_num, life_path_details = calculator.calculate_life_path_number(birth_date)
    soul_urge_num, soul_urge_details = calculator.calculate_soul_urge_number(full_name)
    personality_num, personality_details = calculator.calculate_personality_number(full_name)
    
    # New: Calculate Birth Day Number
    birth_day_num, birth_day_details = calculator.calculate_birth_day_number(birth_date)

    # New: Calculate Lo Shu Grid
    lo_shu_grid = calculator.calculate_lo_shu_grid(birth_date)
    
    # New: Conceptual Astrological Integration
    ascendant_info = calculator.get_ascendant_info(birth_date, birth_time, birth_place)
    moon_sign_info = calculator.get_moon_sign_info(birth_date, birth_time, birth_place)
    planetary_lords = calculator.get_planetary_lords(birth_date, birth_time, birth_place)
    
    # Pass astro_info for planetary compatibility check
    astro_for_compatibility = {
        "ascendant_info": ascendant_info, 
        "moon_sign_info": moon_sign_info, 
        "planetary_lords": planetary_lords
    }
    planetary_compatibility = calculator.check_planetary_compatibility(expression_num, astro_for_compatibility)

    # New: Conceptual Phonetic Vibration
    phonetic_vibration = calculator.analyze_phonetic_vibration(full_name)

    # New: Validation Logic for Expression Number - Pass full astro_info
    expression_validation = calculator.check_expression_validity(
        expression_num, 
        life_path_num, 
        lo_shu_grid, 
        profession_desire or "", 
        birth_day_num, 
        astro_for_compatibility # Pass full astro_info here
    )

    # New: Edge Case Handling - Pass entire profile data for comprehensive analysis
    profile_for_edge_cases = {
        "expression_number": expression_num,
        "life_path_number": life_path_num,
        "birth_day_number": birth_day_num,
        "lo_shu_grid": lo_shu_grid,
        "profession_desire": profession_desire,
        "ascendant_info": ascendant_info,
        "moon_sign_info": moon_sign_info,
        "planetary_lords": planetary_lords,
        "planetary_compatibility": planetary_compatibility,
        "life_path_details": life_path_details, # Needed for karmic debt from birth date
        "expression_details": expression_details # Needed for karmic debt from name
    }
    edge_cases = calculator.analyze_edge_cases(profile_for_edge_cases)

    # Calculate compatibility and insights (existing)
    compatibility_insights = calculate_number_compatibility(expression_num, life_path_num)
    karmic_lessons = analyze_karmic_lessons(full_name, birth_date) # Updated to take birth_date for Lo Shu karmic lessons

    return {
        "full_name": full_name,
        "birth_date": birth_date,
        "birth_time": birth_time,
        "birth_place": birth_place,
        "profession_desire": profession_desire,
        "expression_number": expression_num,
        "expression_details": expression_details,
        "life_path_number": life_path_num,
        "life_path_details": life_path_details,
        "birth_day_number": birth_day_num, # Changed from birth_number
        "birth_day_details": birth_day_details, # New details for birth_day_number
        "soul_urge_number": soul_urge_num,
        "soul_urge_details": soul_urge_details,
        "personality_number": personality_num,
        "personality_details": personality_details,
        "lo_shu_grid": lo_shu_grid,
        "ascendant_info": ascendant_info,
        "moon_sign_info": moon_sign_info,
        "planetary_lords": planetary_lords,
        "planetary_compatibility": planetary_compatibility,
        "phonetic_vibration": phonetic_vibration,
        "expression_validation": expression_validation,
        "edge_cases": edge_cases,
        "compatibility_insights": compatibility_insights,
        "karmic_lessons": karmic_lessons,
        "profile_hash": hashlib.md5(f"{full_name}{birth_date}{birth_time}{birth_place}{profession_desire}".encode()).hexdigest()
    }

def calculate_number_compatibility(expression: int, life_path: int) -> Dict:
    """Calculate compatibility between expression and life path numbers"""
    # Expanded compatibility scores for more nuance
    compatibility_scores = {
        (1, 1): 0.9, (1, 2): 0.6, (1, 3): 0.9, (1, 4): 0.5, (1, 5): 0.8,
        (1, 6): 0.7, (1, 7): 0.4, (1, 8): 0.9, (1, 9): 0.7, (1, 11): 0.85, (1, 22): 0.8, (1, 33): 0.75,

        (2, 2): 0.9, (2, 3): 0.7, (2, 4): 0.8, (2, 5): 0.5, (2, 6): 0.9,
        (2, 7): 0.8, (2, 8): 0.6, (2, 9): 0.8, (2, 11): 0.95, (2, 22): 0.85, (2, 33): 0.9,

        (3, 3): 0.9, (3, 4): 0.6, (3, 5): 0.9, (3, 6): 0.8, (3, 7): 0.7,
        (3, 8): 0.7, (3, 9): 0.9, (3, 11): 0.75, (3, 22): 0.65, (3, 33): 0.85,

        (4, 4): 0.9, (4, 5): 0.6, (4, 6): 0.8, (4, 7): 0.9, (4, 8): 0.7,
        (4, 9): 0.6, (4, 11): 0.7, (4, 22): 0.95, (4, 33): 0.75,

        (5, 5): 0.9, (5, 6): 0.6, (5, 7): 0.8, (5, 8): 0.7, (5, 9): 0.7,
        (5, 11): 0.8, (5, 22): 0.7, (5, 33): 0.6,

        (6, 6): 0.9, (6, 7): 0.7, (6, 8): 0.8, (6, 9): 0.9, (6, 11): 0.9,
        (6, 22): 0.8, (6, 33): 0.95,

        (7, 7): 0.9, (7, 8): 0.6, (7, 9): 0.8, (7, 11): 0.95, (7, 22): 0.85,
        (7, 33): 0.9,

        (8, 8): 0.9, (8, 9): 0.7, (8, 11): 0.7, (8, 22): 0.95, (8, 33): 0.8,

        (9, 9): 0.9, (9, 11): 0.8, (9, 22): 0.7, (9, 33): 0.95,

        (11, 11): 0.95, (11, 22): 0.85, (11, 33): 0.9,
        (22, 22): 0.95, (22, 33): 0.9,
        (33, 33): 0.95
    }
    
    key = tuple(sorted((expression, life_path)))
    score = compatibility_scores.get(key, 0.6) # Default to moderate if not explicitly defined
    
    synergy_areas = []
    if score > 0.85:
        synergy_areas = ["Exceptional harmony", "Powerful synergy", "Effortless understanding", "Accelerated growth"]
    elif score > 0.7:
        synergy_areas = ["Strong compatibility", "Complementary strengths", "Mutual support", "Positive dynamics"]
    elif score > 0.5:
        synergy_areas = ["Moderate compatibility", "Growth opportunities", "Areas for conscious effort", "Dynamic balance"]
    else:
        synergy_areas = ["Learning experiences", "Character building", "Requires conscious effort", "Potential friction points"]
    
    return {
        "compatibility_score": score,
        "synergy_areas": synergy_areas,
        "description": f"Your Expression Number {expression} and Life Path {life_path} create a {'highly harmonious' if score > 0.85 else 'strong' if score > 0.7 else 'moderate' if score > 0.5 else 'challenging'} compatibility."
    }

def analyze_karmic_lessons(name: str, birth_date_str: str) -> Dict:
    """
    Analyze karmic lessons from missing numbers in name (Chaldean) and Lo Shu Grid.
    This now combines both aspects for a holistic view.
    """
    calc = AdvancedNumerologyCalculator()
    
    # Karmic lessons from name (missing numbers in Chaldean letter values)
    name_numbers_present = set()
    for letter in name.upper():
        if letter in calc.CHALDEAN_NUMEROLOGY_MAP:
            name_numbers_present.add(calc.CHALDEAN_NUMEROLOGY_MAP[letter][0])
    
    missing_from_name = sorted(list(set(range(1, 9)) - name_numbers_present)) # Chaldean only uses 1-8 for letters
    
    # Karmic lessons from Lo Shu Grid (missing numbers in birth date digits)
    lo_shu_grid_data = calc.calculate_lo_shu_grid(birth_date_str)
    missing_from_lo_shu = lo_shu_grid_data.get('missing_numbers', [])

    karmic_lessons_map = {
        1: "Developing independence and leadership, cultivating self-reliance and initiative.",
        2: "Fostering cooperation, patience, and diplomacy, enhancing sensitivity and balance.", 
        3: "Cultivating creativity and communication, embracing self-expression and joy.",
        4: "Building discipline, organization, and stability, focusing on practical matters and hard work.",
        5: "Embracing change, freedom, and adaptability, fostering versatility and exploration.",
        6: "Accepting responsibility and nurturing, creating harmony and selfless service.",
        7: "Seeking spiritual understanding, engaging in introspection and acquiring wisdom.",
        8: "Mastering material world success, developing abundance, power, and executive ability.",
        9: "Developing universal compassion, embracing humanitarianism and completion."
    }
    
    all_missing_numbers = sorted(list(set(missing_from_name + missing_from_lo_shu)))
    
    lessons_list = [
        {"number": num, "lesson": karmic_lessons_map.get(num, "General energy imbalance.")}
        for num in all_missing_numbers
    ]
    
    return {
        "missing_numbers_combined": all_missing_numbers,
        "lessons_summary": lessons_list,
        "priority_lesson": lessons_list[0]['lesson'] if lessons_list else "All core lessons integrated or no significant missing numbers."
    }

def generate_timing_recommendations(profile: Dict) -> Dict:
    """Generate optimal timing recommendations"""
    current_year = datetime.datetime.now().year
    
    # Use client's birth month and day for Personal Year calculation
    try:
        birth_month = int(profile['birth_date'].split('-')[1])
        birth_day = int(profile['birth_date'].split('-')[2])
    except (ValueError, IndexError):
        # Fallback to current month/day if birth date parsing fails (though it should be valid from frontend)
        birth_month = datetime.date.today().month
        birth_day = datetime.date.today().day
        logger.warning("Could not parse birth month/day from profile for Personal Year. Using current month/day as fallback.")

    personal_year_calc_month = AdvancedNumerologyCalculator.reduce_number(birth_month, True)
    personal_year_calc_day = AdvancedNumerologyCalculator.reduce_number(birth_day, True)
    personal_year_calc_year = AdvancedNumerologyCalculator.reduce_number(current_year, True)
    
    personal_year_sum = personal_year_calc_month + personal_year_calc_day + personal_year_calc_year
    personal_year = AdvancedNumerologyCalculator.reduce_number(personal_year_sum, True) # Final reduction for Personal Year

    optimal_activities = {
        1: ["Initiating new projects", "Taking leadership roles", "Embracing independence", "Starting fresh"],
        2: ["Forming partnerships", "Collaborating on projects", "Nurturing relationships", "Seeking balance"],
        3: ["Creative expression", "Socializing", "Communicating ideas", "Enjoying life"],
        4: ["Building foundations", "Organizing finances", "Working diligently", "Establishing security"],
        5: ["Embracing change", "Traveling", "Seeking new experiences", "Adapting to new situations"],
        6: ["Focusing on family/home", "Providing service", "Nurturing self and others", "Cultivating harmony"],
        7: ["Introspection", "Study and research", "Spiritual development", "Seeking inner truth"],
        8: ["Material achievement", "Business expansion", "Financial management", "Exercising power responsibly"],
        9: ["Completion of cycles", "Humanitarian efforts", "Letting go of the past", "Universal compassion"],
        11: ["Spiritual insight", "Inspiring others", "Intuitive breakthroughs", "Mastering higher vibrations"],
        22: ["Large-scale building", "Manifesting big dreams", "Global impact", "Practical idealism"],
        33: ["Compassionate service", "Healing", "Teaching universal love", "Selfless dedication"]
    }
    
    # Best months map (conceptual, often aligns with the number itself or its reduced form)
    best_months_map = {
        1: [1, 10], 2: [2, 11], 3: [3, 12], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9],
        11: [2, 11], 22: [4], 33: [6]
    }
    
    recommended_months = best_months_map.get(personal_year, [1, 7]) # Default if not found
    
    return {
        "current_personal_year": personal_year,
        "optimal_activities": optimal_activities.get(personal_year, ["General growth and development"]),
        "energy_description": f"Your Personal Year {personal_year} resonates with the energy of {optimal_activities.get(personal_year, ['growth'])[0].lower()}.",
        "best_months_for_action": recommended_months,
    }

# --- Analytics and Insights Helper Functions ---

def get_mitigation_strategies(expression: int) -> List[str]:
    """Get strategies to mitigate challenges"""
    strategies = {
        1: ["Practice patience and listening", "Delegate tasks effectively", "Cultivate empathy and understanding"],
        2: ["Build self-confidence and assertiveness", "Practice decisive action", "Establish clear personal boundaries"],
        3: ["Focus and prioritize energy", "Develop discipline in creative pursuits", "Cultivate deeper, more meaningful relationships"],
        4: ["Embrace flexibility and adaptability", "Take regular breaks to avoid burnout", "Express creativity outside of work"],
        5: ["Practice commitment and follow-through", "Develop grounding habits like meditation", "Plan ahead to manage impulses"],
        6: ["Set healthy boundaries and learn to say no", "Prioritize self-care and personal needs", "Avoid over-commitment and martyrdom"],
        7: ["Actively engage in social activities", "Practice expressing emotions openly", "Apply knowledge practically in daily life"],
        8: ["Balance work-life commitments", "Nurture personal relationships", "Practice generosity and philanthropy"],
        9: ["Manage emotional volatility", "Accept imperfections in others", "Practice self-compassion and forgiveness"],
        11: ["Practice grounding exercises (e.g., walking in nature)", "Manage nervous energy through mindfulness", "Set realistic and achievable goals"],
        22: ["Delegate tasks effectively to avoid overwhelm", "Prioritize self-care and rest", "Break down large goals into smaller, manageable steps"],
        33: ["Emotional overwhelm, taking on others' pain, and tendency toward self-sacrifice"]
    }
    return strategies.get(expression, ["Practice mindfulness", "Seek balance", "Stay grounded"])


def get_growth_opportunities(expression: int) -> List[str]:
    """Get growth opportunities based on expression number"""
    opportunities = {
        1: ["Leadership development", "Mentoring others", "Innovation projects"],
        2: ["Conflict resolution", "Team building", "Emotional intelligence"],
        3: ["Public speaking", "Creative projects", "Social networking"],
        4: ["Project management", "Systems thinking", "Financial planning"],
        5: ["Cultural exploration", "Technology adoption", "Network expansion"],
        6: ["Community service", "Healing arts", "Family development"],
        7: ["Research projects", "Spiritual studies", "Analytical work"],
        8: ["Business development", "Financial management", "Strategic planning"],
        9: ["Global initiatives", "Teaching opportunities", "Humanitarian work"],
        11: ["Spiritual leadership", "Intuitive development", "Inspiring others"],
        22: ["Large-scale project execution", "Building lasting structures", "Transformational leadership"],
        33: ["Global humanitarian efforts", "Teaching universal love", "Healing initiatives"]
    }
    return opportunities.get(expression, ["Personal development", "Skill building", "Service opportunities"])

def calculate_uniqueness_score(profile: Dict) -> Dict:
    """Calculate how unique this numerological profile is"""
    expression = profile.get('expression_number', 5)
    life_path = profile.get('life_path_number', 5)
    birth_day_number = profile.get('birth_day_number', 0)
    
    rarity_multiplier = 1.0
    if expression in AdvancedNumerologyCalculator.MASTER_NUMBERS:
        rarity_multiplier += 0.5
    if life_path in AdvancedNumerologyCalculator.MASTER_NUMBERS:
        rarity_multiplier += 0.5
    if birth_day_number in AdvancedNumerologyCalculator.MASTER_NUMBERS:
        rarity_multiplier += 0.3

    # Consider rare combinations
    combination_rarity_score = 0.0
    if (expression, life_path) in [(1,8), (8,1), (4,5), (5,4)]: # Examples of challenging/unique combos
        combination_rarity_score += 0.2
    if (expression, life_path) in [(11,22), (22,11), (11,33), (33,11), (22,33), (33,22)]: # Master number combos
        combination_rarity_score += 0.3

    uniqueness = min(0.95, (combination_rarity_score + 0.3) * rarity_multiplier)
    
    rarity_factors = []
    if rarity_multiplier > 1.0:
        rarity_factors.append("Presence of Master Numbers")
    if combination_rarity_score > 0:
        rarity_factors.append("Unique core number combination")
    if not rarity_factors:
        rarity_factors.append("Standard numerical pattern")

    return {
        "score": uniqueness,
        "interpretation": "Highly unique and powerful" if uniqueness > 0.85 else "Unique and insightful" if uniqueness > 0.7 else "Common pattern with individual strengths",
        "rarity_factors": rarity_factors
    }

def predict_success_areas(profile: Dict) -> Dict:
    """Predict areas of likely success"""
    expression = profile.get('expression_number', 5)
    life_path = profile.get('life_path_number', 5)
    
    success_areas_map = {
        1: ["Leadership", "Entrepreneurship", "Innovation", "Pioneering new ventures"],
        2: ["Collaboration", "Diplomacy", "Counseling", "Partnerships", "Mediation"],
        3: ["Creative arts", "Communication", "Entertainment", "Public speaking", "Marketing"],
        4: ["Management", "Organization", "Finance", "Construction", "System building"],
        5: ["Travel", "Technology", "Sales", "Journalism", "Public relations"],
        6: ["Healthcare", "Education", "Service", "Family business", "Community work"],
        7: ["Research", "Analysis", "Spirituality", "Science", "Philosophy"],
        8: ["Business", "Finance", "Authority", "Real estate", "Law"],
        9: ["Humanitarian work", "Teaching", "Global reach", "Philanthropy", "Social reform"],
        11: ["Spiritual leadership", "Intuitive development", "Innovation", "Inspiring masses"],
        22: ["Large-scale building", "Transformation", "Leadership in big organizations", "Practical idealism"],
        33: ["Healing", "Teaching", "Service leadership", "Universal love advocacy", "Spiritual guidance"]
    }
    
    primary = success_areas_map.get(expression, ["General success"])
    secondary = success_areas_map.get(life_path, [])
    
    combined = list(set(primary + secondary))
    
    return {
        "primary_areas": primary,
        "secondary_areas": secondary,
        "combined_strengths": combined[:6],
        "confidence_level": "High" if len(combined) > 4 else "Moderate"
    }

def identify_potential_challenges(profile: Dict) -> Dict:
    """Identify potential challenges and growth areas"""
    expression = profile.get('expression_number', 5)
    life_path = profile.get('life_path_number', 5)
    
    challenges_map = {
        1: ["Impatience", "Domination tendency", "Isolation", "Ego challenges"],
        2: ["Over-sensitivity", "Indecision", "Dependency", "Lack of assertiveness"],
        3: ["Scattered energy", "Superficiality", "Inconsistency", "Emotional avoidance"],
        4: ["Rigidity", "Resistance to change", "Workaholic tendencies", "Lack of spontaneity"],
        5: ["Restlessness", "Commitment issues", "Impulsiveness", "Overindulgence"],
        6: ["Over-responsibility", "Interference", "Martyrdom", "Self-sacrifice"],
        7: ["Isolation", "Over-analysis", "Emotional detachment", "Cynicism"],
        8: ["Materialism", "Power struggles", "Relationship neglect", "Authoritarianism"],
        9: ["Emotional volatility", "Disappointment", "Burnout", "Difficulty with boundaries"],
        11: ["Nervous tension", "Unrealistic expectations", "Hypersensitivity", "Self-doubt"],
        22: ["Overwhelming pressure", "Perfectionism", "Burnout risk", "Difficulty delegating"],
        33: ["Emotional overwhelm, taking on others' pain, and tendency toward self-sacrifice", "Martyrdom", "Boundary issues"]
    }
    
    potential_challenges = challenges_map.get(expression, ["General life challenges"])
    # Add challenges from life path if they are distinct
    life_path_challenges = challenges_map.get(life_path, [])
    for challenge in life_path_challenges:
        if challenge not in potential_challenges:
            potential_challenges.append(challenge)

    return {
        "potential_challenges": potential_challenges,
        "mitigation_strategies": get_mitigation_strategies(expression),
        "growth_opportunities": get_growth_opportunities(expression)
    }

def get_development_recommendations(profile: Dict) -> Dict:
    """Get personalized development recommendations"""
    expression = profile.get('expression_number', 5)
    life_path = profile.get('life_path_number', 5)
    karmic_lessons_data = profile.get('karmic_lessons', {})
    karmic_lessons_summary = karmic_lessons_data.get('lessons_summary', [])
    
    # Extract specific lessons for actionable guidance
    karmic_work_items = [lesson['lesson'] for lesson in karmic_lessons_summary]
    
    return {
        "immediate_focus": f"Develop your Expression {expression} energy through {get_growth_opportunities(expression)[0].lower()}",
        "long_term_goal": f"Align with your Life Path {life_path} by embracing {get_growth_opportunities(life_path)[0].lower()}",
        "karmic_work": karmic_work_items[:3] if karmic_work_items else ["Continue spiritual growth and self-awareness."],
        "monthly_practices": [
            "Daily meditation or reflection to connect with inner wisdom.",
            "Journaling about number patterns and their manifestation in your life.",
            f"Actively practicing the positive traits of your Expression {expression} and Life Path {life_path} numbers.",
            "Consciously working on identified challenges and integrating mitigation strategies."
        ]
    }

def generate_yearly_forecast(profile: Dict, years: int = 3) -> Dict:
    """Generate multi-year numerological forecast"""
    current_year = datetime.datetime.now().year
    
    forecast = {}
    
    for year_offset in range(years):
        target_year = current_year + year_offset
        # Personal Year calculation based on birth month and day, and target year
        try:
            birth_month = int(profile['birth_date'].split('-')[1])
            birth_day = int(profile['birth_date'].split('-')[2])
        except (ValueError, IndexError):
            # Fallback to current month/day if birth date parsing fails (though it should be valid from frontend)
            birth_month = datetime.date.today().month
            birth_day = datetime.date.today().day
            logger.warning("Could not parse birth month/day for Personal Year. Using current month/day as fallback.")

        personal_year_calc_month = AdvancedNumerologyCalculator.reduce_number(birth_month, True)
        personal_year_calc_day = AdvancedNumerologyCalculator.reduce_number(birth_day, True)
        personal_year_calc_year = AdvancedNumerologyCalculator.reduce_number(target_year, True)
        
        personal_year_sum = personal_year_calc_month + personal_year_calc_day + personal_year_calc_year
        personal_year = AdvancedNumerologyCalculator.reduce_number(personal_year_sum, True)
        
        year_themes = {
            1: {"theme": "New Beginnings & Initiative", "focus": "Starting fresh, leadership opportunities, self-discovery."},
            2: {"theme": "Cooperation & Partnership", "focus": "Forming alliances, nurturing relationships, seeking balance."},
            3: {"theme": "Creative Expression & Communication", "focus": "Socializing, artistic pursuits, joyful self-expression."},
            4: {"theme": "Building Foundations & Discipline", "focus": "Hard work, organization, establishing security, practical planning."},
            5: {"theme": "Freedom & Dynamic Change", "focus": "Embracing new experiences, travel, adaptability, adventure."},
            6: {"theme": "Responsibility & Harmony", "focus": "Focusing on family/home, providing service, nurturing others, community engagement."},
            7: {"theme": "Introspection & Spiritual Growth", "focus": "Study, reflection, inner development, seeking deeper truths."},
            8: {"theme": "Material Success & Power", "focus": "Business expansion, financial management, achievement, leadership in material world."},
            9: {"theme": "Completion & Humanitarianism", "focus": "Endings, letting go of the past, humanitarian efforts, universal love."},
            11: {"theme": "Spiritual Illumination & Inspiration", "focus": "Intuitive breakthroughs, inspiring others, spiritual leadership."},
            22: {"theme": "Master Builder Year & Manifestation", "focus": "Large-scale manifestation, practical idealism, global impact."},
            33: {"theme": "Universal Service & Healing", "focus": "Compassionate leadership, healing the world, selfless dedication."}
        }
        
        year_info = year_themes.get(personal_year, {"theme": "General Growth", "focus": "Personal development and learning."})
        
        # Optimal months can be conceptually tied to the personal year number
        optimal_months_for_year = []
        if personal_year == 1: optimal_months_for_year = [1, 10]
        elif personal_year == 2: optimal_months_for_year = [2, 11]
        elif personal_year == 3: optimal_months_for_year = [3, 12]
        elif personal_year == 4: optimal_months_for_year = [4]
        elif personal_year == 5: optimal_months_for_year = [5]
        elif personal_year == 6: optimal_months_for_year = [6]
        elif personal_year == 7: optimal_months_for_year = [7]
        elif personal_year == 8: optimal_months_for_year = [8]
        elif personal_year == 9: optimal_months_for_year = [9]
        elif personal_year == 11: optimal_months_for_year = [2, 11] # Reduced to 2
        elif personal_year == 22: optimal_months_for_year = [4] # Reduced to 4
        elif personal_year == 33: optimal_months_for_year = [6] # Reduced to 6

        forecast[target_year] = {
            "personal_year": personal_year,
            "theme": year_info["theme"],
            "focus_areas": year_info["focus"],
            "optimal_months": optimal_months_for_year,
            "energy_level": "High" if personal_year in [1, 3, 5, 8, 11, 22, 33] else "Moderate" if personal_year in [2, 6, 9] else "Reflective"
        }
    
    return forecast

# --- PDF Generation Function ---
def create_numerology_pdf(report_data: Dict) -> bytes:
    """
    Generates a PDF numerology report using ReportLab.
    This function now expects ALL data to be pre-processed and sent from the frontend.
    It structures the PDF content based on the 11 sections and additional details.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    
    # Custom styles for better readability and structure
    styles.add(ParagraphStyle(name='TitleStyle', fontSize=24, leading=28, alignment=TA_CENTER, spaceAfter=20, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SubHeadingStyle', fontSize=16, leading=20, spaceBefore=15, spaceAfter=8, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SectionHeadingStyle', fontSize=14, leading=18, spaceBefore=12, spaceAfter=6, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='SubSectionHeadingStyle', fontSize=12, leading=16, spaceBefore=10, spaceAfter=4, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='BoldBodyText', fontSize=10, leading=14, spaceAfter=6, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='NormalBodyText', fontSize=10, leading=14, spaceAfter=6, fontName='Helvetica'))
    styles.add(ParagraphStyle(name='BulletStyle', fontSize=10, leading=14, leftIndent=36, bulletIndent=18, spaceAfter=3, fontName='Helvetica'))
    styles.add(ParagraphStyle(name='ItalicBodyText', fontSize=10, leading=14, spaceAfter=6, fontName='Helvetica-Oblique'))

    # Ensure standard styles are also configured for consistency
    styles['Heading1'].fontSize = 18
    styles['Heading1'].leading = 22
    styles['Heading1'].spaceBefore = 12
    styles['Heading1'].spaceAfter = 6
    styles['Heading1'].fontName = 'Helvetica-Bold'

    styles['Heading2'].fontSize = 14
    styles['Heading2'].leading = 18
    styles['Heading2'].spaceBefore = 10
    styles['Heading2'].spaceAfter = 4
    styles['Heading2'].fontName = 'Helvetica-Bold'

    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 14
    styles['BodyText'].spaceAfter = 6

    styles['Bullet'].fontSize = 10
    styles['Bullet'].leading = 14
    styles['Bullet'].leftIndent = 36
    styles['Bullet'].bulletIndent = 18
    styles['Bullet'].spaceAfter = 3

    Story = []

    # Title Page / Header
    Story.append(Paragraph("🌟 Your Personalized Numerology Report 🌟", styles['TitleStyle']))
    Story.append(Spacer(1, 0.2 * inch))
    Story.append(Paragraph(f"For: <b>{report_data.get('full_name', 'Client Name')}</b>", styles['SubHeadingStyle']))
    Story.append(Paragraph(f"Birth Date: <b>{report_data.get('birth_date', 'N/A')}</b>", styles['SubSectionHeadingStyle']))
    if report_data.get('birth_time'):
        Story.append(Paragraph(f"Birth Time: <b>{report_data.get('birth_time', 'N/A')}</b>", styles['SubSectionHeadingStyle']))
    if report_data.get('birth_place'):
        Story.append(Paragraph(f"Birth Place: <b>{report_data.get('birth_place', 'N/A')}</b>", styles['SubSectionHeadingStyle']))
    Story.append(Paragraph(f"Desired Outcome: <b>{report_data.get('desired_outcome', 'N/A')}</b>", styles['SubSectionHeadingStyle']))
    Story.append(Spacer(1, 0.5 * inch))
    Story.append(Paragraph("A Comprehensive Guide to Your Energetic Blueprint and Name Optimization", styles['ItalicBodyText']))
    Story.append(PageBreak())

    # 1. Introduction (from editable_main_report_content)
    # This section is the initial AI-generated report content, which the practitioner can edit.
    # It should contain all 11 sections as generated by the LLM.
    intro_content = report_data.get('intro_response', 'No introduction available.')
    # Replace markdown headers with ReportLab paragraph styles
    # H1 -> SubHeadingStyle, H2 -> SectionHeadingStyle, H3 -> SubSectionHeadingStyle
    # Use regex to replace markdown headers and bold text
    formatted_intro_content = intro_content
    formatted_intro_content = re.sub(r'### (.*)', r'<para style="SubSectionHeadingStyle">\1</para>', formatted_intro_content)
    formatted_intro_content = re.sub(r'## (.*)', r'<para style="SectionHeadingStyle">\1</para>', formatted_intro_content)
    formatted_intro_content = re.sub(r'# (.*)', r'<para style="SubHeadingStyle">\1</para>', formatted_intro_content)
    formatted_intro_content = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_intro_content) # Bold
    formatted_intro_content = re.sub(r'\*(.*?)\*', r'<i>\1</i>', formatted_intro_content) # Italic
    formatted_intro_content = formatted_intro_content.replace('\n', '<br/>') # Convert newlines to breaks

    Story.append(Paragraph(formatted_intro_content, styles['NormalBodyText']))
    Story.append(PageBreak())

    profile_details = report_data.get('profile_details', {})

    # 4. Strategic Name Corrections (from final_suggested_names_list)
    if report_data.get('suggested_names') and report_data['suggested_names'].get('suggestions'):
        Story.append(Paragraph("🌟 Strategic Name Corrections (Validated)", styles['Heading1']))
        Story.append(Paragraph(f"Overall Reasoning: {report_data['suggested_names'].get('reasoning', 'Based on a comprehensive numerological analysis and practitioner validation.')}", styles['NormalBodyText']))
        for suggestion in report_data['suggested_names']['suggestions']:
            Story.append(Paragraph(f"• <b>{suggestion['name']}</b> (Expression Number: {suggestion['expression_number']}): {suggestion['rationale']}", styles['BulletStyle']))
        Story.append(Spacer(1, 0.2 * inch))

    # NEW: Validation Chat Summary (Optional, if frontend provides)
    if report_data.get('validation_summary'):
        Story.append(Paragraph("✅ Validation Chat Summary & Practitioner's Conclusion", styles['Heading1']))
        validation_html = report_data['validation_summary'].replace('\n', '<br/>')
        validation_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', validation_html)
        Story.append(Paragraph(validation_html, styles['NormalBodyText']))
        Story.append(Spacer(1, 0.2 * inch))

    # NEW: Practitioner's Tailored Notes (Optional, if frontend provides)
    if report_data.get('practitioner_notes'):
        Story.append(Paragraph("✍️ Practitioner's Private Notes", styles['Heading1']))
        notes_html = report_data['practitioner_notes'].replace('\n', '<br/>')
        notes_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', notes_html)
        Story.append(Paragraph(notes_html, styles['NormalBodyText']))
        Story.append(Spacer(1, 0.2 * inch))

    # --- Append other detailed sections from `profile_details` for clarity in PDF
    # These sections are generated by the backend's numerology calculations and
    # are passed to the LLM for the main report. Including them explicitly here
    # ensures they are always present in the PDF, regardless of how the LLM phrases them.

    # Core Numerological Blueprint (Detailed Analysis)
    Story.append(Paragraph("📊 Core Numerological Blueprint", styles['Heading1']))
    Story.append(Paragraph(f"<b>Full Name:</b> {profile_details.get('full_name', 'N/A')}", styles['NormalBodyText']))
    Story.append(Paragraph(f"<b>Birth Date:</b> {profile_details.get('birth_date', 'N/A')}", styles['NormalBodyText']))
    if profile_details.get('birth_time'):
        Story.append(Paragraph(f"<b>Birth Time:</b> {profile_details.get('birth_time', 'N/A')}</b>", styles['NormalBodyText']))
    if profile_details.get('birth_place'):
        Story.append(Paragraph(f"<b>Birth Place:</b> {profile_details.get('birth_place', 'N/A')}</b>", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.1 * inch))

    Story.append(Paragraph("<b>Expression Number:</b>", styles['BoldBodyText']))
    Story.append(Paragraph(f"Your Expression Number is <b>{profile_details.get('expression_number', 'N/A')}</b> (ruled by {profile_details.get('expression_details', {}).get('planetary_ruler', 'N/A')}). This number reveals your natural talents, abilities, and the path you are destined to express in the world.", styles['NormalBodyText']))
    Story.append(Paragraph(f"<i>Core Interpretation:</i> {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(profile_details.get('expression_number'), {}).get('core', 'N/A')}", styles['ItalicBodyText']))
    Story.append(Spacer(1, 0.1 * inch))

    Story.append(Paragraph("<b>Life Path Number:</b>", styles['BoldBodyText']))
    Story.append(Paragraph(f"Your Life Path Number is <b>{profile_details.get('life_path_number', 'N/A')}</b>. This is your destiny number, outlining the major lessons, opportunities, and challenges you will encounter throughout your life's journey.", styles['NormalBodyText']))
    Story.append(Paragraph(f"<i>Core Interpretation:</i> {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(profile_details.get('life_path_number'), {}).get('core', 'N/A')}", styles['ItalicBodyText']))
    Story.append(Spacer(1, 0.1 * inch))

    Story.append(Paragraph("<b>Birth Day Number:</b>", styles['BoldBodyText']))
    Story.append(Paragraph(f"Your Birth Day Number is <b>{profile_details.get('birth_day_number', 'N/A')}</b>. This number reflects specific talents and traits you possess that influence your daily characteristics and how you approach life.", styles['NormalBodyText']))
    Story.append(Paragraph(f"<i>Core Interpretation:</i> {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(profile_details.get('birth_day_number'), {}).get('core', 'N/A')}", styles['ItalicBodyText']))
    Story.append(Spacer(1, 0.1 * inch))

    Story.append(Paragraph("<b>Soul Urge Number:</b>", styles['BoldBodyText']))
    Story.append(Paragraph(f"Your Soul Urge Number is <b>{profile_details.get('soul_urge_number', 'N/A')}</b>. This reveals your deepest desires, motivations, and what truly fulfills your heart.", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.1 * inch))

    Story.append(Paragraph("<b>Personality Number:</b>", styles['BoldBodyText']))
    Story.append(Paragraph(f"Your Personality Number is <b>{profile_details.get('personality_number', 'N/A')}</b>. This represents how others perceive you and the aspects of your personality you show to the world.", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.2 * inch))

    # Compatibility Synthesis
    comp_insights = profile_details.get('compatibility_insights', {})
    Story.append(Paragraph("<b>Compatibility Synthesis (Expression & Life Path):</b>", styles['BoldBodyText']))
    Story.append(Paragraph(f"{comp_insights.get('description', 'N/A')}", styles['NormalBodyText']))
    if comp_insights.get('synergy_areas'):
        Story.append(Paragraph("Synergy Areas:", styles['BoldBodyText']))
        for area in comp_insights['synergy_areas']:
            Story.append(Paragraph(f"• {area}", styles['BulletStyle']))
    Story.append(Spacer(1, 0.2 * inch))

    # Karmic Lessons & Soul Work
    karmic_lessons_data = profile_details.get('karmic_lessons', {})
    if karmic_lessons_data.get('lessons_summary'):
        Story.append(Paragraph("🔮 Karmic Lessons & Soul Work", styles['Heading1']))
        Story.append(Paragraph("The following numbers are missing from your name's Chaldean values or your birth date's digits, indicating specific karmic lessons or areas for soul growth:", styles['NormalBodyText']))
        for lesson_item in karmic_lessons_data['lessons_summary']:
            Story.append(Paragraph(f"• <b>Number {lesson_item['number']}</b>: {lesson_item['lesson']}", styles['BulletStyle']))
        Story.append(Spacer(1, 0.2 * inch))

    # Lo Shu Grid Analysis
    if profile_details.get('lo_shu_grid'):
        lo_shu = profile_details['lo_shu_grid']
        Story.append(Paragraph("🗺️ Lo Shu Grid Analysis", styles['Heading2']))
        Story.append(Paragraph(f"<b>Digits in Birth Date</b>: {lo_shu.get('grid_counts')}", styles['NormalBodyText']))
        if lo_shu.get('missing_numbers'):
            Story.append(Paragraph("<b>Missing Numbers & Their Impact</b>:", styles['BoldBodyText']))
            for lesson in lo_shu['missing_lessons']:
                Story.append(Paragraph(f"• {lesson['number']}: {lesson['impact']}", styles['BulletStyle']))
        else:
            Story.append(Paragraph("All numbers 1-9 are present in your birth date, indicating a balanced Lo Shu Grid.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.2 * inch))

    # Astrological Integration (Conceptual) - Enhanced to show compatibility flags
    if profile_details.get('ascendant_info') or profile_details.get('moon_sign_info') or profile_details.get('planetary_lords'):
        astro = profile_details
        Story.append(Paragraph("🌌 Astro-Numerological Insights (Conceptual)", styles['Heading1']))
        Story.append(Paragraph("This section provides conceptual astrological insights integrated with your numerological profile, based on your provided birth time and place. Please note, these are simplified interpretations and not precise astrological chart readings.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        Story.append(Paragraph(f"<b>Ascendant/Lagna (Rising Sign)</b>: {astro.get('ascendant_info', {}).get('sign', 'N/A')} (Conceptual Ruler: {astro.get('ascendant_info', {}).get('ruler', 'N/A')})", styles['NormalBodyText']))
        Story.append(Paragraph(f"<i>Notes:</i> {astro.get('ascendant_info', {}).get('notes', 'N/A')}", styles['ItalicBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        Story.append(Paragraph(f"<b>Moon Sign/Rashi</b>: {astro.get('moon_sign_info', {}).get('sign', 'N/A')} (Conceptual Ruler: {astro.get('moon_sign_info', {}).get('ruler', 'N/A')})", styles['NormalBodyText']))
        Story.append(Paragraph(f"<i>Notes:</i> {astro.get('moon_sign_info', {}).get('notes', 'N/A')}", styles['ItalicBodyText']))
        Story.append(Spacer(1, 0.1 * inch))
        
        Story.append(Paragraph("<b>Conceptual Planetary Lords (Benefic/Malefic Influences)</b>:", styles['BoldBodyText']))
        if astro.get('planetary_lords'):
            for lord in astro['planetary_lords']:
                Story.append(Paragraph(f"• {lord['planet']} ({lord['nature']})", styles['BulletStyle']))
        else:
            Story.append(Paragraph("No specific conceptual planetary lords identified.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.1 * inch))

        if astro.get('planetary_compatibility', {}).get('compatibility_flags'):
            Story.append(Paragraph("<b>Planetary Compatibility Notes for Suggested Name (Expression Number)</b>:", styles['BoldBodyText']))
            for flag in astro['planetary_compatibility']['compatibility_flags']:
                Story.append(Paragraph(f"• {flag}", styles['BulletStyle']))
        else:
            Story.append(Paragraph("No specific planetary compatibility flags identified for the suggested name based on conceptual analysis.", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.2 * inch))

    # Phonetic Vibration (Conceptual)
    if profile_details.get('phonetic_vibration'):
        phonetics = profile_details['phonetic_vibration']
        Story.append(Paragraph("🗣️ Phonetic Vibration Analysis (Conceptual)", styles['Heading1']))
        Story.append(Paragraph("This analysis provides conceptual insights into the sound and flow of your name, which can influence its energetic resonance.", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Overall Harmony Score</b>: {phonetics['score']:.2f} ({'Harmonious' if phonetics['is_harmonious'] else 'Needs Consideration'})", styles['NormalBodyText']))
        if phonetics.get('notes'):
            Story.append(Paragraph("<b>Notes on Sound Vibration</b>:", styles['BoldBodyText']))
            for note in phonetics['notes']:
                Story.append(Paragraph(f"• {note}", styles['BulletStyle']))
        Story.append(Spacer(1, 0.2 * inch))

    # Expression Number Validation
    if profile_details.get('expression_validation'):
        exp_val = profile_details['expression_validation']
        Story.append(Paragraph("✅ Expression Number Validation", styles['Heading1']))
        Story.append(Paragraph(f"<b>Validity Status</b>: {'Valid' if exp_val['is_valid'] else 'Needs Attention'}", styles['NormalBodyText']))
        if exp_val.get('flags'):
            Story.append(Paragraph("<b>Flags/Concerns</b>:", styles['BoldBodyText']))
            for flag in exp_val['flags']:
                Story.append(Paragraph(f"• {flag}", styles['BulletStyle']))
        Story.append(Paragraph(f"<b>Recommendation</b>: {exp_val['recommendation']}", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.2 * inch))

    # Edge Case Handling
    if profile_details.get('edge_cases'):
        edge_cases = profile_details['edge_cases']
        if edge_cases:
            Story.append(Paragraph("⚠️ Identified Edge Cases & Special Considerations", styles['Heading1']))
            Story.append(Paragraph("These are specific numerological or astro-numerological combinations that may present unique challenges or opportunities:", styles['NormalBodyText']))
            for ec in edge_cases:
                Story.append(Paragraph(f"<b>Type</b>: {ec['type']}", styles['BoldBodyText']))
                Story.append(Paragraph(f"Description: {ec['description']}", styles['NormalBodyText']))
                Story.append(Paragraph(f"Resolution Guidance: {ec['resolution_guidance']}", styles['NormalBodyText']))
                Story.append(Spacer(1, 0.1 * inch))
        Story.append(Spacer(1, 0.2 * inch))

    # Temporal Numerology
    if report_data.get('timing_recommendations'):
        timing = report_data['timing_recommendations']
        Story.append(Paragraph("⏰ Temporal Numerology (Timing & Cycles)", styles['Heading1']))
        Story.append(Paragraph(f"<b>Current Personal Year ({timing['current_personal_year']})</b>: {timing['energy_description']}", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Optimal Activities for this Year</b>: {', '.join(timing['optimal_activities'])}", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Best Months for Action/Implementation</b>: {', '.join(map(str, timing['best_months_for_action']))}", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.2 * inch))

    # Shadow Work & Growth Edge (Potential Challenges)
    if report_data.get('potential_challenges', {}).get('potential_challenges'):
        challenges = report_data['potential_challenges']
        Story.append(Paragraph("🚧 Shadow Work & Growth Edge", styles['Heading1']))
        Story.append(Paragraph("Understanding potential challenges helps in transforming them into growth opportunities:", styles['NormalBodyText']))
        Story.append(Paragraph("<b>Potential Challenges Identified</b>:", styles['BoldBodyText']))
        for challenge in challenges['potential_challenges']:
            Story.append(Paragraph(f"• {challenge}", styles['BulletStyle']))
        Story.append(Paragraph("<b>Mitigation Strategies</b>:", styles['BoldBodyText']))
        for strategy in challenges['mitigation_strategies']:
            Story.append(Paragraph(f"• {strategy}", styles['BulletStyle']))
        Story.append(Paragraph("<b>Growth Opportunities</b>:", styles['BoldBodyText']))
        for opportunity in challenges['growth_opportunities']:
            Story.append(Paragraph(f"• {opportunity}", styles['BulletStyle']))
        Story.append(Spacer(1, 0.2 * inch))

    # Personalized Development Blueprint
    if report_data.get('development_recommendations'):
        dev_recs = report_data['development_recommendations']
        Story.append(Paragraph("🌱 Personalized Development Blueprint", styles['Heading1']))
        Story.append(Paragraph(f"<b>Immediate Focus</b>: {dev_recs['immediate_focus']}", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Long-Term Goal</b>: {dev_recs['long_term_goal']}", styles['NormalBodyText']))
        if dev_recs['karmic_work']:
            Story.append(Paragraph(f"<b>Key Karmic Work Areas</b>: {', '.join(dev_recs['karmic_work'])}", styles['NormalBodyText']))
        Story.append(Paragraph("<b>Recommended Monthly Practices</b>:", styles['BoldBodyText']))
        for practice in dev_recs['monthly_practices']:
            Story.append(Paragraph(f"• {practice}", styles['BulletStyle']))
        Story.append(Spacer(1, 0.2 * inch))

    # Future Cycles Forecast
    if report_data.get('yearly_forecast'):
        Story.append(Paragraph("🗓️ Future Cycles Forecast (Next 3 Years)", styles['Heading1']))
        for year, forecast_data in report_data['yearly_forecast'].items():
            Story.append(Paragraph(f"<b>Year {year} (Personal Year {forecast_data['personal_year']})</b>: {forecast_data['theme']}", styles['Heading2']))
            Story.append(Paragraph(f"Focus: {forecast_data['focus_areas']}", styles['NormalBodyText']))
            Story.append(Paragraph(f"Energy Level: {forecast_data['energy_level']}", styles['NormalBodyText']))
            Story.append(Paragraph(f"Optimal Months: {', '.join(map(str, forecast_data['optimal_months']))}", styles['NormalBodyText']))
            Story.append(Spacer(1, 0.1 * inch))
        Story.append(Spacer(1, 0.2 * inch))

    # Numerical Uniqueness Profile
    if report_data.get('uniqueness_score'):
        uniqueness = report_data['uniqueness_score'] 
        Story.append(Paragraph("✨ Numerical Uniqueness Profile", styles['Heading1']))
        Story.append(Paragraph(f"<b>Your Profile Uniqueness Score</b>: {uniqueness['score']:.2f} ({uniqueness['interpretation']})", styles['NormalBodyText']))
        Story.append(Paragraph(f"<b>Key Rarity Factors</b>: {', '.join(uniqueness['rarity_factors'])}", styles['NormalBodyText']))
        Story.append(Spacer(1, 0.2 * inch))

    # Empowerment Conclusion
    Story.append(PageBreak())
    Story.append(Paragraph("💖 Empowerment & Final Thoughts 💖", styles['TitleStyle']))
    Story.append(Spacer(1, 0.2 * inch))
    Story.append(Paragraph("This report has unveiled the profound insights hidden within your unique numerological blueprint. Remember, this knowledge is a powerful tool to guide you, but your free will and conscious choices are the ultimate architects of your destiny. Embrace your numbers, align with your purpose, and step into your full potential.", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.5 * inch))
    Story.append(Paragraph("Thank you for choosing Sheelaa's Elite AI Numerology Assistant for your journey of self-discovery and transformation.", styles['NormalBodyText']))
    Story.append(Spacer(1, 0.2 * inch))
    Story.append(Paragraph("<i>Disclaimer: Numerology provides insights and guidance. It is not a substitute for professional advice. Your free will and choices ultimately shape your destiny.</i>", styles['ItalicBodyText']))

    doc.build(Story)
    buffer.seek(0)
    return buffer.getvalue()

# --- Flask Routes ---
@app.route('/')
def home():
    """Basic home route for health check."""
    return "Hello from Flask!"

@app.route('/chat', methods=['POST'])
@rate_limited("30 per minute")
@performance_monitor
def chat():
    """Handles chat messages and advanced report/validation requests."""
    data = request.json
    message = data.get('message')
    chat_type = data.get('type')

    if not message and chat_type != 'validation_chat':
        return jsonify({"error": "No message provided"}), 400
    
    if chat_type != 'validation_chat' and not SecurityManager.validate_input_security(message):
        logger.warning(f"Security alert: Malicious input detected: {message}")
        return jsonify({"error": "Invalid input. Potential security threat detected."}), 400

    logger.info(f"Received request type: {chat_type}")

    try:
        if chat_type == 'validation_chat':
            logger.info("Processing validation chat message.")
            original_profile = data.get('original_profile')
            suggested_name = data.get('suggested_name')
            chat_history_from_frontend = data.get('chat_history', [])
            current_message_content = data.get('current_message', '').strip()

            if not original_profile or not suggested_name:
                return jsonify({"error": "Missing original profile or suggested name for validation chat."}), 400
            
            # Recalculate suggested name's expression for context
            suggested_expression_num, suggested_expression_details = AdvancedNumerologyCalculator.calculate_expression_number(suggested_name)

            # Extract original profile data for conceptual astro calculations
            birth_date_val = original_profile.get('birthDate')
            birth_time_val = original_profile.get('birthTime')
            birth_place_val = original_profile.get('birthPlace')
            desired_outcome_val = original_profile.get('desiredOutcome')
            current_life_path_val = original_profile.get('currentLifePathNumber')
            current_birth_day_val = original_profile.get('currentBirthNumber') # This is Birth Day Number from frontend

            # Perform conceptual astro calculations for the validation context
            val_ascendant_info = AdvancedNumerologyCalculator.get_ascendant_info(birth_date_val, birth_time_val, birth_place_val)
            val_moon_sign_info = AdvancedNumerologyCalculator.get_moon_sign_info(birth_date_val, birth_time_val, birth_place_val)
            val_planetary_lords = AdvancedNumerologyCalculator.get_planetary_lords(birth_date_val, birth_time_val, birth_place_val)
            
            # Prepare astro_info for compatibility and edge case checks
            astro_for_validation = {
                "ascendant_info": val_ascendant_info, 
                "moon_sign_info": val_moon_sign_info, 
                "planetary_lords": val_planetary_lords
            }
            val_planetary_compatibility = AdvancedNumerologyCalculator.check_planetary_compatibility(suggested_expression_num, astro_for_validation)
            val_lo_shu_grid = AdvancedNumerologyCalculator.calculate_lo_shu_grid(birth_date_val)
            
            # Expression validation for the suggested name
            expression_validation_for_suggested = AdvancedNumerologyCalculator.check_expression_validity(
                suggested_expression_num, 
                current_life_path_val, 
                val_lo_shu_grid, 
                desired_outcome_val, 
                current_birth_day_val, 
                astro_for_validation
            )

            llm_validation_input_data = {
                "original_full_name": original_profile['fullName'],
                "birth_date": birth_date_val,
                "birth_time": birth_time_val, 
                "birth_place": birth_place_val, 
                "original_expression_number": original_profile['currentExpressionNumber'],
                "original_life_path_number": current_life_path_val,
                "original_birth_day_number": current_birth_day_val,
                "desired_outcome": desired_outcome_val,
                "suggested_name": suggested_name,
                "suggested_expression_num": suggested_expression_num,
                "suggested_core_interpretation": AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(suggested_expression_num, {}).get('core', 'N/A'),
                "lo_shu_grid": val_lo_shu_grid,
                "ascendant_info": val_ascendant_info,
                "moon_sign_info": val_moon_sign_info,
                "planetary_lords": val_planetary_lords,
                "planetary_compatibility": val_planetary_compatibility,
                "phonetic_vibration": AdvancedNumerologyCalculator.analyze_phonetic_vibration(suggested_name),
                "expression_validation_for_suggested": expression_validation_for_suggested
            }

            # Prepare chat history for Langchain
            langchain_chat_history = []
            for msg in chat_history_from_frontend:
                if msg['sender'] == 'user':
                    langchain_chat_history.append(HumanMessage(content=msg['text']))
                elif msg['sender'] == 'ai':
                    langchain_chat_history.append(AIMessage(content=msg['text']))

            # Pre-format JSON strings to avoid f-string backslash issue
            lo_shu_grid_json = json.dumps(llm_validation_input_data['lo_shu_grid'], indent=2)
            ascendant_info_json = json.dumps(llm_validation_input_data['ascendant_info'], indent=2)
            moon_sign_info_json = json.dumps(llm_validation_input_data['moon_sign_info'], indent=2)
            planetary_lords_json = json.dumps(llm_validation_input_data['planetary_lords'], indent=2)
            planetary_compatibility_json = json.dumps(llm_validation_input_data['planetary_compatibility'], indent=2)
            phonetic_vibration_json = json.dumps(llm_validation_input_data['phonetic_vibration'], indent=2)
            expression_validation_json = json.dumps(llm_validation_input_data['expression_validation_for_suggested'], indent=2)

                       # The prompt for validation chat
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=NAME_VALIDATION_SYSTEM_PROMPT.format(desired_outcome=llm_validation_input_data['desired_outcome'])),
                HumanMessage(
                    content=(
                        "**CLIENT PROFILE FOR VALIDATION:**\n"
                        f"Original Full Name: {llm_validation_input_data['original_full_name']}\n"
                        f"Birth Date: {llm_validation_input_data['birth_date']}\n"
                        f"Birth Time: {llm_validation_input_data['birth_time']}\n"
                        f"Birth Place: {llm_validation_input_data['birth_place']}\n"
                        f"Desired Outcome: {llm_validation_input_data['desired_outcome']}\n"
                        f"Current Expression Number: {llm_validation_input_data['original_expression_number']}\n"
                        f"Current Life Path Number: {llm_validation_input_data['original_life_path_number']}\n"
                        f"Current Birth Day Number: {llm_validation_input_data['original_birth_day_number']}\n\n"

                        "**SUGGESTED NAME FOR ANALYSIS:**\n"
                        f"Name: {llm_validation_input_data['suggested_name']}\n"
                        f"Calculated Expression Number: {llm_validation_input_data['suggested_expression_num']}\n"
                        f"Core Interpretation: {llm_validation_input_data['suggested_core_interpretation']}\n\n"

                        "**DETAILED NUMEROLOGICAL & ASTRO-NUMEROLOGICAL DATA FOR SUGGESTED NAME:**\n"
                        "Lo Shu Grid: {}\n"
                        "Ascendant Info (Conceptual): {}\n"
                        "Moon Sign Info (Conceptual): {}\n"
                        "Planetary Lords (Conceptual): {}\n"
                        "Planetary Compatibility Notes: {}\n"
                        "Phonetic Vibration: {}\n"
                        "Expression Validation Summary: {}\n\n"

                        "**CHAT HISTORY:**\n"
                        "{}\n\n"

                        "**LATEST USER MESSAGE:**\n"
                        "{}"
                    ).format(
                        lo_shu_grid_json,
                        ascendant_info_json,
                        moon_sign_info_json,
                        planetary_lords_json,
                        planetary_compatibility_json,
                        phonetic_vibration_json,
                        expression_validation_json,
                        "\n".join([f"{m.type.capitalize()}: {m.content}" for m in langchain_chat_history]),
                        current_message_content
                    )
                )
            ])

            
            chain = prompt | llm_manager.analytical_llm

            ai_response = chain.invoke({})
            
            return jsonify({"response": ai_response.content}), 200

        elif message:
            report_request = MessageParser.parse_report_request(message)

            if report_request:
                logger.info(f"Attempting to parse GENERATE_ADVANCED_REPORT message for: {report_request.full_name}")
                profile = get_comprehensive_numerology_profile(
                    report_request.full_name, 
                    report_request.birth_date,
                    birth_time=report_request.birth_time, 
                    birth_place=report_request.birth_place, 
                    profession_desire=report_request.desired_outcome
                )
                
                target_numbers = NameSuggestionEngine.determine_target_numbers_for_outcome(report_request.desired_outcome)
                
                name_suggestions = NameSuggestionEngine.generate_name_suggestions(
                    llm_manager.creative_llm, 
                    report_request.full_name, 
                    report_request.desired_outcome, 
                    target_numbers
                )
                logger.info(f"Generated Name Suggestions: {name_suggestions.json()}")

                timing_recommendations = generate_timing_recommendations(profile)
                uniqueness_score = calculate_uniqueness_score(profile)
                potential_challenges = identify_potential_challenges(profile)
                development_recommendations = get_development_recommendations(profile)
                yearly_forecast = generate_yearly_forecast(profile)

                llm_input_data = {
                    "full_name": report_request.full_name,
                    "birth_date": report_request.birth_date,
                    "birth_time": report_request.birth_time, 
                    "birth_place": report_request.birth_place, 
                    "desired_outcome": report_request.desired_outcome,
                    "current_expression_number": report_request.current_expression_number,
                    "current_life_path_number": report_request.current_life_path_number,
                    "profile_details": profile, # Contains all detailed calculations
                    "timing_recommendations": timing_recommendations,
                    "uniqueness_score": uniqueness_score,
                    "potential_challenges": potential_challenges,
                    "development_recommendations": development_recommendations,
                    "yearly_forecast": yearly_forecast,
                    "suggested_names": name_suggestions.dict()
                }
                
                system_message = SystemMessage(content=ADVANCED_REPORT_SYSTEM_PROMPT)
                
                human_message = HumanMessage(
                    content=ADVANCED_REPORT_HUMAN_PROMPT.format(
                        llm_input_data=json.dumps(llm_input_data, indent=2)
                    )
                )
                
                messages = [system_message, human_message]
                
                report_response = llm_manager.creative_llm.invoke(messages)
                
                full_report_data = {
                    **llm_input_data,
                    "intro_response": report_response.content # This is the full Markdown report
                }

                return jsonify({"response": report_response.content, "full_report_data_for_pdf": full_report_data}), 200

            else:
                logger.info(f"Processing general chat message: {message}")
                
                llm_manager.memory.chat_memory.add_user_message(HumanMessage(content=message))
                
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=GENERAL_CHAT_SYSTEM_PROMPT),
                    llm_manager.memory.chat_memory.messages[-1] 
                ])
                
                chain = prompt | llm_manager.llm
                
                ai_response = chain.invoke({"input": message})
                
                llm_manager.memory.chat_memory.add_ai_message(AIMessage(content=ai_response.content))
                
                return jsonify({"response": ai_response.content}), 200
        else:
            return jsonify({"error": "Invalid request type or missing message."}), 400

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred. Please try again later."}), 500

@app.route('/generate_pdf_report', methods=['POST'])
@rate_limited("5 per hour")
@performance_monitor
def generate_pdf_report_endpoint():
    """
    Endpoint to generate and return a PDF numerology report.
    This endpoint now expects the full, potentially modified/validated report data
    directly from the frontend.
    """
    report_data = request.json
    
    if not all([report_data.get('full_name'), report_data.get('birth_date'), report_data.get('intro_response')]):
        return jsonify({"error": "Missing essential data for PDF generation."}), 400

    try:
        pdf_bytes = create_numerology_pdf(report_data)
        
        logger.info(f"Generated PDF bytes size: {len(pdf_bytes)} bytes")

        filename = f"Numerology_Report_{report_data['full_name'].replace(' ', '_')}.pdf"
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        logger.error(f"Error generating PDF report: {e}", exc_info=True)
        return jsonify({"error": f"Failed to generate PDF report: {e}"}), 500

# --- Error Handlers ---
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded. Please try again later.",
        "retry_after": getattr(e, 'retry_after', 60)
    }), 429

@app.errorhandler(400)
def bad_request_handler(e):
    return jsonify({"error": "Bad request. Please check your input."}), 400

@app.errorhandler(500)
def internal_error_handler(e):
    logger.error(f"Internal server error: {e}", exc_info=True)
    return jsonify({"error": "Internal server error. Please try again later."}), 500

@app.errorhandler(404)
def not_found_handler(e):
    return jsonify({"error": "Endpoint not found."}), 404

# --- Application Initialization ---
def initialize_application():
    """Initialize all application components"""
    logger.info("Initializing Numerology Application...")
    
    if not llm_manager.initialize():
        logger.error("Failed to initialize LLM manager. Exiting application.")
        sys.exit(1)
    
    logger.info("Application initialized successfully")
    return True

initialize_application()

if __name__ == "__main__":
    logger.info("Starting development server...")
    app.run(
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 5000)), 
        debug=os.getenv("FLASK_DEBUG", "False").lower() == "true"
    )
