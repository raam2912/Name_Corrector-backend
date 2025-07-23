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
from langchain.output_parsers import PydanticOutputParser

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
    level=logging.INFO, # Reverted to INFO for production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('numerology_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
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
    "http://localhost:3000"
]}})

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
    desired_outcome: str
    current_expression_number: int
    current_life_path_number: int

class NameSuggestion(BaseModel):
    name: str = Field(description="Full name suggestion")
    rationale: str = Field(description="Brief rationale for the suggestion")
    cultural_score: float = Field(description="Cultural appropriateness score 0-1")

class NameSuggestionsOutput(BaseModel):
    suggestions: List[NameSuggestion] = Field(description="List of name suggestions")
    reasoning: str = Field(description="Overall reasoning for suggestions")

# --- Advanced Numerology Calculator ---
class AdvancedNumerologyCalculator:
    NUMEROLOGY_MAP = {
        'A': 1, 'J': 1, 'S': 1,
        'B': 2, 'K': 2, 'T': 2,
        'C': 3, 'L': 3, 'U': 3,
        'D': 4, 'M': 4, 'V': 4,
        'E': 5, 'N': 5, 'W': 5,
        'F': 6, 'O': 6, 'X': 6,
        'G': 7, 'P': 7, 'Y': 7,
        'H': 8, 'Q': 8, 'Z': 8,
        'I': 9, 'R': 9
    }
    
    VOWELS = set('AEIOU')
    CONSONANTS = set('BCDFGHJKLMNPQRSTVWXYZ')
    KARMIC_DEBT_NUMBERS = {13, 14, 16, 19}
    MASTER_NUMBERS = {11, 22, 33, 44}
    
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
        """Enhanced number reduction with master number preservation"""
        if allow_master_numbers and num in AdvancedNumerologyCalculator.MASTER_NUMBERS:
            return num
        
        while num > 9:
            if allow_master_numbers and num in AdvancedNumerologyCalculator.MASTER_NUMBERS:
                break
            num = sum(int(digit) for digit in str(num))
        return num
    
    @staticmethod
    def calculate_expression_number(name: str) -> Tuple[int, Dict]:
        """Calculate expression number with detailed breakdown"""
        total = 0
        letter_breakdown = {}
        cleaned_name = ''.join(filter(str.isalpha, name)).upper()
        
        for letter in cleaned_name:
            if letter in AdvancedNumerologyCalculator.NUMEROLOGY_MAP:
                value = AdvancedNumerologyCalculator.NUMEROLOGY_MAP[letter]
                total += value
                letter_breakdown[letter] = letter_breakdown.get(letter, 0) + value
        
        reduced = AdvancedNumerologyCalculator.reduce_number(total)
        
        return reduced, {
            "total_before_reduction": total,
            "letter_breakdown": letter_breakdown,
            "is_master_number": reduced in AdvancedNumerologyCalculator.MASTER_NUMBERS,
            "karmic_debt": total in AdvancedNumerologyCalculator.KARMIC_DEBT_NUMBERS
        }
    
    @staticmethod
    def calculate_soul_urge_number(name: str) -> Tuple[int, Dict]:
        """Calculate soul urge (vowels only)"""
        total = 0
        vowel_breakdown = {}
        cleaned_name = ''.join(filter(str.isalpha, name)).upper()
        
        for letter in cleaned_name:
            if letter in AdvancedNumerologyCalculator.VOWELS and letter in AdvancedNumerologyCalculator.NUMEROLOGY_MAP:
                value = AdvancedNumerologyCalculator.NUMEROLOGY_MAP[letter]
                total += value
                vowel_breakdown[letter] = vowel_breakdown.get(letter, 0) + value
        
        reduced = AdvancedNumerologyCalculator.reduce_number(total)
        
        return reduced, {
            "total_before_reduction": total,
            "vowel_breakdown": vowel_breakdown,
            "is_master_number": reduced in AdvancedNumerologyCalculator.MASTER_NUMBERS
        }
    
    @staticmethod
    def calculate_personality_number(name: str) -> Tuple[int, Dict]:
        """Calculate personality number (consonants only)"""
        total = 0
        consonant_breakdown = {}
        cleaned_name = ''.join(filter(str.isalpha, name)).upper()
        
        for letter in cleaned_name:
            if letter in AdvancedNumerologyCalculator.CONSONANTS and letter in AdvancedNumerologyCalculator.NUMEROLOGY_MAP:
                value = AdvancedNumerologyCalculator.NUMEROLOGY_MAP[letter]
                total += value
                consonant_breakdown[letter] = consonant_breakdown.get(letter, 0) + value
        
        reduced = AdvancedNumerologyCalculator.reduce_number(total)
        
        return reduced, {
            "total_before_reduction": total,
            "consonant_breakdown": consonant_breakdown,
            "is_master_number": reduced in AdvancedNumerologyCalculator.MASTER_NUMBERS
        }

    @staticmethod
    def calculate_life_path_number(birth_date_str: str) -> Tuple[int, Dict]:
        """Enhanced life path calculation with karmic debt detection"""
        try:
            year, month, day = map(int, birth_date_str.split('-'))
        except ValueError:
            raise ValueError("Invalid birth date format. Please use YYYY-MM-DD.")

        karmic_debt_present = []
        
        month_reduced = AdvancedNumerologyCalculator.reduce_number(month, True)
        day_reduced = AdvancedNumerologyCalculator.reduce_number(day, True) 
        year_reduced = AdvancedNumerologyCalculator.reduce_number(year, True)
        
        # Check for karmic debt in the original numbers
        if month in AdvancedNumerologyCalculator.KARMIC_DEBT_NUMBERS:
            karmic_debt_present.append(month)
        if day in AdvancedNumerologyCalculator.KARMIC_DEBT_NUMBERS:
            karmic_debt_present.append(day)

        total = month_reduced + day_reduced + year_reduced
        final_number = AdvancedNumerologyCalculator.reduce_number(total, True)
        
        return final_number, {
            "month": month,
            "day": day, 
            "year": year,
            "month_reduced": month_reduced,
            "day_reduced": day_reduced,
            "year_reduced": year_reduced,
            "total_before_final_reduction": total,
            "karmic_debt_numbers": karmic_debt_present,
            "is_master_number": final_number in AdvancedNumerologyCalculator.MASTER_NUMBERS
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
        FIX: Adjusted regex to correctly match the prefix and handle potential newlines.
        """
        pattern = r"GENERATE_ADVANCED_REPORT:\s*My full name is \"(.*?)\"\s*and my birth date is \"(.*?)\"\.\s*My current Name \(Expression\) Number is (\d+)\s*and Life Path Number is (\d+)\.\s*I desire the following positive outcome in my life:\s*\"(.*?)\"\s*[\.\n]?"
        match = re.search(pattern, message)
        
        if not match:
            return None
        
        return ReportRequest(
            full_name=match.group(1),
            birth_date=match.group(2),
            desired_outcome=match.group(5), # Desired outcome is the 5th group
            current_expression_number=int(match.group(3)),
            current_life_path_number=int(match.group(4))
        )

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
            'healing': [6, 33]
        }
        
        for keyword, numbers in target_map.items():
            if keyword in outcome_lower:
                return numbers
        
        if any(word in outcome_lower for word in ['career', 'work', 'job']):
            return [1, 8, 22]
        elif any(word in outcome_lower for word in ['peace', 'harmony', 'balance']):
            return [2, 6, 11]
        else:
            return [1, 3, 6, 8] # General positive numbers

# --- LLM Manager ---
class LLMManager:
    def __init__(self):
        self.llm = None
        self.creative_llm = None
        self.analytical_llm = None
        self.memory = None
        self.executor = ThreadPoolExecutor(max_workers=3)
        
    def initialize(self) -> bool:
        """Initialize multiple LLM instances for different purposes"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable not found. LLM initialization failed.")
            return False
            
        try:
            # Standard LLM for general responses
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.7,
                top_p=0.9,
                top_k=40
            )
            
            # Creative LLM for name generation
            self.creative_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=api_key,
                temperature=0.9,
                top_p=0.95,
                top_k=60
            )
            
            # Analytical LLM for interpretations
            self.analytical_llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=api_key,
                temperature=0.3,
                top_p=0.8,
                top_k=20
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
                # Use Flask-Limiter if available
                # Note: Flask-Limiter expects the decorated function to be a Flask view function
                # This decorator might need adjustment if used on non-view functions.
                return limiter.limit(limit_string)(f)(*args, **kwargs)
            else:
                # Simple in-memory rate limiting fallback (not implemented here)
                return f(*args, **kwargs)
        return wrapper
    return decorator

@cached_operation(timeout=3600)
@performance_monitor
def get_comprehensive_numerology_profile(full_name: str, birth_date: str) -> Dict:
    """Get comprehensive numerology profile with caching"""
    calculator = AdvancedNumerologyCalculator()
    
    # Calculate all numbers
    expression_num, expression_details = calculator.calculate_expression_number(full_name)
    life_path_num, life_path_details = calculator.calculate_life_path_number(birth_date)
    soul_urge_num, soul_urge_details = calculator.calculate_soul_urge_number(full_name)
    personality_num, personality_details = calculator.calculate_personality_number(full_name)
    
    # Calculate compatibility and insights
    compatibility_insights = calculate_number_compatibility(expression_num, life_path_num)
    karmic_lessons = analyze_karmic_lessons(full_name)
    
    return {
        "full_name": full_name,
        "birth_date": birth_date,
        "expression_number": expression_num,
        "expression_details": expression_details,
        "life_path_number": life_path_num,
        "life_path_details": life_path_details,
        "soul_urge_number": soul_urge_num,
        "soul_urge_details": soul_urge_details,
        "personality_number": personality_num,
        "personality_details": personality_details,
        "compatibility_insights": compatibility_insights,
        "karmic_lessons": karmic_lessons,
        "profile_hash": hashlib.md5(f"{full_name}{birth_date}".encode()).hexdigest()
    }

def calculate_number_compatibility(expression: int, life_path: int) -> Dict:
    """Calculate compatibility between expression and life path numbers"""
    # Simplified compatibility matrix
    compatibility_scores = {
        (1, 1): 0.8, (1, 2): 0.6, (1, 3): 0.9, (1, 4): 0.5, (1, 5): 0.8,
        (1, 6): 0.7, (1, 7): 0.4, (1, 8): 0.9, (1, 9): 0.7, (1, 11): 0.8,
        (2, 2): 0.9, (2, 3): 0.7, (2, 4): 0.8, (2, 5): 0.5, (2, 6): 0.9,
        (2, 7): 0.8, (2, 8): 0.6, (2, 9): 0.8, (2, 11): 0.9, (2, 22): 0.8,
        (3, 3): 0.9, (3, 4): 0.6, (3, 5): 0.9, (3, 6): 0.8, (3, 7): 0.7,
        (3, 8): 0.7, (3, 9): 0.9, (3, 11): 0.7, (3, 22): 0.6, (3, 33): 0.8,
        (4, 4): 0.9, (4, 5): 0.6, (4, 6): 0.8, (4, 7): 0.9, (4, 8): 0.7,
        (4, 9): 0.6, (4, 11): 0.7, (4, 22): 0.9, (4, 33): 0.7,
        (5, 5): 0.9, (5, 6): 0.6, (5, 7): 0.8, (5, 8): 0.7, (5, 9): 0.7,
        (5, 11): 0.8, (5, 22): 0.7, (5, 33): 0.6,
        (6, 6): 0.9, (6, 7): 0.7, (6, 8): 0.8, (6, 9): 0.9, (6, 11): 0.9,
        (6, 22): 0.8, (6, 33): 0.9,
        (7, 7): 0.9, (7, 8): 0.6, (7, 9): 0.8, (7, 11): 0.9, (7, 22): 0.8,
        (7, 33): 0.9,
        (8, 8): 0.9, (8, 9): 0.7, (8, 11): 0.7, (8, 22): 0.9, (8, 33): 0.8,
        (9, 9): 0.9, (9, 11): 0.8, (9, 22): 0.7, (9, 33): 0.9,
        (11, 11): 0.9, (11, 22): 0.8, (11, 33): 0.9,
        (22, 22): 0.9, (22, 33): 0.8,
        (33, 33): 0.9
    }
    
    # Ensure order for key lookup
    key = tuple(sorted((expression, life_path)))
    score = compatibility_scores.get(key, 0.6)  # Default compatibility
    
    synergy_areas = []
    if score > 0.8:
        synergy_areas = ["Natural harmony", "Shared goals", "Mutual support", "Effortless understanding"]
    elif score > 0.6:
        synergy_areas = ["Complementary strengths", "Growth opportunities", "Dynamic balance"]
    else:
        synergy_areas = ["Learning experiences", "Character building", "Areas for conscious effort"]
    
    return {
        "compatibility_score": score,
        "synergy_areas": synergy_areas,
        "description": f"Your Expression Number {expression} and Life Path {life_path} create {'excellent' if score > 0.8 else 'good' if score > 0.6 else 'challenging'} compatibility."
    }

def analyze_karmic_lessons(name: str) -> Dict:
    """Analyze karmic lessons from missing numbers in name"""
    calc = AdvancedNumerologyCalculator()
    name_numbers = set()
    
    for letter in name.upper():
        if letter in calc.NUMEROLOGY_MAP:
            name_numbers.add(calc.NUMEROLOGY_MAP[letter])
    
    missing_numbers = sorted(list(set(range(1, 10)) - name_numbers)) # Ensure consistent order
    
    karmic_lessons_map = {
        1: "Learning independence and leadership (Self-reliance, initiative)",
        2: "Developing cooperation and patience (Diplomacy, sensitivity)", 
        3: "Cultivating creativity and communication (Self-expression, joy)",
        4: "Building discipline and organization (Stability, hard work)",
        5: "Embracing change and freedom (Adaptability, versatility)",
        6: "Accepting responsibility and nurturing (Harmony, selfless service)",
        7: "Seeking spiritual understanding (Introspection, wisdom)",
        8: "Mastering material world success (Abundance, power)",
        9: "Developing universal compassion (Humanitarianism, completion)"
    }
    
    lessons_list = [karmic_lessons_map[num] for num in missing_numbers]
    
    return {
        "missing_numbers": missing_numbers,
        "lessons": lessons_list,
        "priority_lesson": lessons_list[0] if lessons_list else "All core lessons integrated"
    }

def generate_timing_recommendations(profile: Dict) -> Dict:
    """Generate optimal timing recommendations"""
    current_year = datetime.datetime.now().year
    # birth_year = int(profile["birth_date"].split("-")[0]) # Not directly used for personal year calc here
    
    # To calculate current personal year accurately:
    # Reduce current month, day, and current year to single digits (or master numbers)
    # Then sum them and reduce again.
    
    today = datetime.date.today()
    personal_year_calc_month = AdvancedNumerologyCalculator.reduce_number(today.month, True)
    personal_year_calc_day = AdvancedNumerologyCalculator.reduce_number(today.day, True)
    personal_year_calc_year = AdvancedNumerologyCalculator.reduce_number(current_year, True)
    
    personal_year = AdvancedNumerologyCalculator.reduce_number(personal_year_calc_month + personal_year_calc_day + personal_year_calc_year, True)
    
    optimal_activities = {
        1: ["Initiating new projects", "Taking leadership roles", "Embracing independence"],
        2: ["Forming partnerships", "Collaborating on projects", "Nurturing relationships"],
        3: ["Creative expression", "Socializing", "Communicating ideas"],
        4: ["Building foundations", "Organizing finances", "Working diligently"],
        5: ["Embracing change", "Traveling", "Seeking new experiences"],
        6: ["Focusing on family/home", "Providing service", "Nurturing self and others"],
        7: ["Introspection", "Study and research", "Spiritual development"],
        8: ["Material achievement", "Business expansion", "Financial management"],
        9: ["Completion of cycles", "Humanitarian efforts", "Letting go of the past"],
        11: ["Spiritual insight", "Inspiring others", "Intuitive breakthroughs"],
        22: ["Large-scale building", "Manifesting big dreams", "Global impact"],
        33: ["Compassionate service", "Healing", "Teaching universal love"]
    }
    
    # Best months can be complex, often related to the personal year number itself.
    # For simplicity, let's suggest months whose numerological value aligns or are traditionally favorable.
    # This is a placeholder; real numerology timing is more nuanced.
    best_months_map = {
        1: [1, 10], 2: [2, 11], 3: [3, 12], 4: [4], 5: [5], 6: [6], 7: [7], 8: [8], 9: [9],
        11: [2, 11], 22: [4], 33: [6]
    }
    
    recommended_months = best_months_map.get(personal_year, [1, 7]) # Default to Jan, July
    
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
        1: ["Impatience", "Domination tendency", "Isolation"], # These were challenges, not strategies. Corrected below.
        2: ["Build confidence", "Practice decision-making", "Set boundaries"],
        3: ["Focus on priorities", "Develop discipline", "Deepen relationships"],
        4: ["Embrace flexibility", "Take breaks", "Express creativity"],
        5: ["Practice commitment", "Develop grounding habits", "Plan ahead"],
        6: ["Over-responsibility", "Interference", "Martyrdom"], # These were challenges, not strategies. Corrected below.
        7: ["Engage socially", "Express emotions", "Apply knowledge practically"],
        8: ["Balance work-life", "Nurture relationships", "Practice generosity"],
        9: ["Set realistic expectations", "Practice self-compassion", "Take breaks"],
        11: ["Grounding exercises", "Manage nervous energy", "Set realistic goals"],
        22: ["Delegate tasks", "Prioritize self-care", "Break down large goals"],
        33: ["Establish strong boundaries", "Practice self-love", "Seek support"]
    }
    # Corrected strategies
    strategies[1] = ["Practice patience and listening", "Delegate tasks", "Cultivate empathy"]
    strategies[6] = ["Set healthy boundaries", "Prioritize self-care", "Avoid over-commitment"]
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
    
    # Master numbers are rarer
    rarity_multiplier = 1.5 if expression in AdvancedNumerologyCalculator.MASTER_NUMBERS else 1.0
    combination_rarity = abs(expression - life_path) / 10.0
    
    uniqueness = min(0.95, (combination_rarity + 0.3) * rarity_multiplier)
    
    return {
        "score": uniqueness,
        "interpretation": "Highly unique" if uniqueness > 0.8 else "Moderately unique" if uniqueness > 0.6 else "Common pattern",
        "rarity_factors": ["Master number present"] if rarity_multiplier > 1 else ["Standard numerical pattern"]
    }

def predict_success_areas(profile: Dict) -> Dict:
    """Predict areas of likely success"""
    expression = profile.get('expression_number', 5)
    life_path = profile.get('life_path_number', 5)
    
    success_areas = {
        1: ["Leadership", "Entrepreneurship", "Innovation"],
        2: ["Collaboration", "Diplomacy", "Counseling"],
        3: ["Creative arts", "Communication", "Entertainment"],
        4: ["Management", "Organization", "Finance"],
        5: ["Travel", "Technology", "Sales"],
        6: ["Healthcare", "Education", "Service"],
        7: ["Research", "Analysis", "Spirituality"],
        8: ["Business", "Finance", "Authority"],
        9: ["Humanitarian work", "Teaching", "Global reach"],
        11: ["Inspiration", "Spiritual teaching", "Innovation"],
        22: ["Large-scale building", "Transformation", "Leadership"],
        33: ["Healing", "Teaching", "Service leadership"]
    }
    
    primary = success_areas.get(expression, ["General success"])
    secondary = success_areas.get(life_path, [])
    
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
    
    challenges = {
        1: ["Impatience", "Domination tendency", "Isolation"],
        2: ["Over-sensitivity", "Indecision", "Dependency"],
        3: ["Scattered energy", "Superficiality", "Inconsistency"],
        4: ["Rigidity", "Resistance to change", "Workaholic tendencies"],
        5: ["Restlessness", "Commitment issues", "Impulsiveness"],
        6: ["Over-responsibility", "Interference", "Martyrdom"],
        7: ["Isolation", "Over-analysis", "Emotional detachment"],
        8: ["Materialism", "Power struggles", "Relationship neglect"],
        9: ["Emotional volatility", "Disappointment", "Burnout"],
        11: ["Nervous tension", "Unrealistic expectations", "Hypersensitivity"],
        22: ["Overwhelming pressure", "Perfectionism", "Burnout risk"],
        33: ["Emotional overwhelm", "Boundary issues", "Self-sacrifice"]
    }
    
    return {
        "potential_challenges": challenges.get(expression, ["General life challenges"]),
        "mitigation_strategies": get_mitigation_strategies(expression),
        "growth_opportunities": get_growth_opportunities(expression)
    }

def get_development_recommendations(profile: Dict) -> Dict:
    """Get personalized development recommendations"""
    expression = profile.get('expression_number', 5)
    life_path = profile.get('life_path_number', 5)
    karmic_lessons = profile.get('karmic_lessons', {}).get('lessons', [])
    
    return {
        "immediate_focus": f"Develop your Expression {expression} energy through {get_growth_opportunities(expression)[0].lower()}",
        "long_term_goal": f"Align with your Life Path {life_path} by embracing {get_growth_opportunities(life_path)[0].lower()}",
        "karmic_work": karmic_lessons[:2] if karmic_lessons else ["Continue spiritual growth"],
        "monthly_practices": [
            "Daily meditation or reflection",
            "Journaling about number patterns",
            "Practicing your key number's positive traits",
            "Working on challenge mitigation"
        ]
    }

def generate_yearly_forecast(profile: Dict, years: int = 3) -> Dict:
    """Generate multi-year numerological forecast"""
    current_year = datetime.datetime.now().year
    # birth_year = int(profile["birth_date"].split("-")[0]) # Not directly used for personal year calc here
    
    forecast = {}
    
    for year_offset in range(years):
        target_year = current_year + year_offset
        # Recalculate personal year for target_year
        today_for_calc = datetime.date(target_year, datetime.date.today().month, datetime.date.today().day)
        personal_year_calc_month = AdvancedNumerologyCalculator.reduce_number(today_for_calc.month, True)
        personal_year_calc_day = AdvancedNumerologyCalculator.reduce_number(today_for_calc.day, True)
        personal_year_calc_year = AdvancedNumerologyCalculator.reduce_number(target_year, True)
        
        personal_year = AdvancedNumerologyCalculator.reduce_number(personal_year_calc_month + personal_year_calc_day + personal_year_calc_year, True)
        
        year_themes = {
            1: {"theme": "New Beginnings", "focus": "Starting fresh, leadership opportunities"},
            2: {"theme": "Cooperation", "focus": "Partnerships, relationship building"},
            3: {"theme": "Creative Expression", "focus": "Communication, artistic pursuits"},
            4: {"theme": "Building Foundations", "focus": "Hard work, organization, stability"},
            5: {"theme": "Freedom & Change", "focus": "Travel, new experiences, adventure"},
            6: {"theme": "Responsibility", "focus": "Family, service, nurturing others"},
            7: {"theme": "Introspection", "focus": "Study, reflection, inner development"},
            8: {"theme": "Material Success", "focus": "Business, finance, achievement"},
            9: {"theme": "Completion", "focus": "Endings, humanitarian service, wisdom"},
            11: {"theme": "Spiritual Illumination", "focus": "Intuitive breakthroughs, inspiring others"},
            22: {"theme": "Master Builder Year", "focus": "Large-scale manifestation, practical idealism"},
            33: {"theme": "Universal Service", "focus": "Compassionate leadership, healing the world"}
        }
        
        year_info = year_themes.get(personal_year, {"theme": "Growth", "focus": "Personal development"})
        
        forecast[target_year] = {
            "personal_year": personal_year,
            "theme": year_info["theme"],
            "focus_areas": year_info["focus"],
            "optimal_months": [(personal_year + i - 1) % 12 + 1 for i in range(3)], # Simple placeholder
            "energy_level": "High" if personal_year in [1, 3, 5, 8, 11, 22, 33] else "Moderate" if personal_year in [2, 6, 9] else "Reflective"
        }
    
    return forecast

# --- PDF Generation Function ---
def create_numerology_pdf(report_data: Dict) -> bytes:
    """
    Generates a PDF numerology report using ReportLab.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles for better readability
    styles.add(ParagraphStyle(name='TitleStyle', fontSize=24, leading=28, alignment=TA_CENTER, spaceAfter=20))
    
    # FIX: Modify existing Heading1 style instead of adding a new one with the same name.
    # Or, if you want a completely new style, give it a unique name like 'CustomHeading1'.
    styles['Heading1'].fontSize = 18
    styles['Heading1'].leading = 22
    styles['Heading1'].spaceBefore = 12
    styles['Heading1'].spaceAfter = 6
    styles['Heading1'].fontName = 'Helvetica-Bold'

    styles.add(ParagraphStyle(name='Heading2', fontSize=14, leading=18, spaceBefore=10, spaceAfter=4, fontName='Helvetica-Bold'))
    styles.add(ParagraphStyle(name='BodyText', fontSize=10, leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name='Bullet', fontSize=10, leading=14, leftIndent=36, bulletIndent=18, spaceAfter=3))
    styles.add(ParagraphStyle(name='BoldBodyText', fontSize=10, leading=14, spaceAfter=6, fontName='Helvetica-Bold'))

    Story = []

    # Title
    Story.append(Paragraph("🌟 Your Personalized Numerology Report �", styles['TitleStyle']))
    Story.append(Spacer(1, 0.2 * inch))

    # Introduction (parsed from report_data.get('intro_response'))
    intro_html = report_data.get('intro_response', 'No introduction available.').replace('\n', '<br/>')
    # Basic Markdown to HTML conversion for bold text
    intro_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', intro_html)
    Story.append(Paragraph(intro_html, styles['BodyText']))
    Story.append(Spacer(1, 0.2 * inch))

    # Current Numerological Profile
    Story.append(Paragraph("✨ Your Current Numerological Profile", styles['Heading1'])) # Use the modified Heading1
    Story.append(Paragraph(f"<b>Expression Number {report_data.get('expression_number')}</b>: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(report_data.get('expression_number'), {}).get('core', 'Universal energy')}", styles['BodyText']))
    Story.append(Paragraph(f"<b>Life Path Number {report_data.get('life_path_number')}</b>: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(report_data.get('life_path_number'), {}).get('core', 'Universal energy')}", styles['BodyText']))
    Story.append(Paragraph(f"<b>Compatibility Analysis</b>: {report_data.get('compatibility_insights', {}).get('description', 'Your numbers work in harmony.')}", styles['BodyText']))
    
    if report_data.get('soul_urge_number'):
        Story.append(Paragraph(f"<b>Soul Urge Number {report_data.get('soul_urge_number')}</b>: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(report_data.get('soul_urge_number'), {}).get('core', 'Universal energy')}", styles['BodyText']))
    if report_data.get('personality_number'):
        Story.append(Paragraph(f"<b>Personality Number {report_data.get('personality_number')}</b>: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(report_data.get('personality_number'), {}).get('core', 'Universal energy')}", styles['BodyText']))

    Story.append(Spacer(1, 0.2 * inch))

    # Karmic Lessons
    if report_data.get('karmic_lessons', {}).get('lessons'):
        Story.append(Paragraph("🔮 Karmic Lessons to Address", styles['Heading2']))
        for lesson in report_data['karmic_lessons']['lessons']:
            Story.append(Paragraph(f"• {lesson}", styles['Bullet']))
        Story.append(Spacer(1, 0.1 * inch))

    # Timing Recommendations
    if report_data.get('timing_recommendations'):
        timing = report_data['timing_recommendations']
        Story.append(Paragraph("⏰ Optimal Timing & Energy", styles['Heading2']))
        Story.append(Paragraph(f"<b>Current Personal Year ({timing['current_personal_year']})</b>: {timing['energy_description']}", styles['BodyText']))
        Story.append(Paragraph(f"<b>Optimal Activities</b>: {', '.join(timing['optimal_activities'])}", styles['BodyText']))
        Story.append(Paragraph(f"<b>Best Months for Action</b>: {', '.join(map(str, timing['best_months_for_action']))}", styles['BodyText']))
        Story.append(Spacer(1, 0.1 * inch))

    # Potential Challenges
    if report_data.get('potential_challenges', {}).get('potential_challenges'):
        challenges = report_data['potential_challenges']
        Story.append(Paragraph("🚧 Potential Challenges & Growth Areas", styles['Heading2']))
        Story.append(Paragraph("<b>Challenges</b>:", styles['BoldBodyText']))
        for challenge in challenges['potential_challenges']:
            Story.append(Paragraph(f"• {challenge}", styles['Bullet']))
        Story.append(Paragraph("<b>Mitigation Strategies</b>:", styles['BoldBodyText']))
        for strategy in challenges['mitigation_strategies']:
            Story.append(Paragraph(f"• {strategy}", styles['Bullet']))
        Story.append(Paragraph("<b>Growth Opportunities</b>:", styles['BoldBodyText']))
        for opportunity in challenges['growth_opportunities']:
            Story.append(Paragraph(f"• {opportunity}", styles['Bullet']))
        Story.append(Spacer(1, 0.1 * inch))

    # Development Recommendations
    if report_data.get('development_recommendations'):
        dev_recs = report_data['development_recommendations']
        Story.append(Paragraph("🌱 Personalized Development Recommendations", styles['Heading2']))
        Story.append(Paragraph(f"<b>Immediate Focus</b>: {dev_recs['immediate_focus']}", styles['BodyText']))
        Story.append(Paragraph(f"<b>Long-Term Goal</b>: {dev_recs['long_term_goal']}", styles['BodyText']))
        if dev_recs['karmic_work']:
            Story.append(Paragraph(f"<b>Karmic Work</b>: {', '.join(dev_recs['karmic_work'])}", styles['BodyText']))
        Story.append(Paragraph("<b>Monthly Practices</b>:", styles['BoldBodyText']))
        for practice in dev_recs['monthly_practices']:
            Story.append(Paragraph(f"• {practice}", styles['Bullet']))
        Story.append(Spacer(1, 0.1 * inch))

    # Yearly Forecast
    if report_data.get('yearly_forecast'):
        Story.append(Paragraph("🗓️ Yearly Forecast (Next 3 Years)", styles['Heading1']))
        for year, forecast_data in report_data['yearly_forecast'].items():
            Story.append(Paragraph(f"<b>Year {year} (Personal Year {forecast_data['personal_year']})</b>: {forecast_data['theme']}", styles['Heading2']))
            Story.append(Paragraph(f"Focus: {forecast_data['focus_areas']}", styles['BodyText']))
            Story.append(Paragraph(f"Energy Level: {forecast_data['energy_level']}", styles['BodyText']))
            Story.append(Paragraph(f"Optimal Months: {', '.join(map(str, forecast_data['optimal_months']))}", styles['BodyText']))
            Story.append(Spacer(1, 0.1 * inch))
        Story.append(Spacer(1, 0.2 * inch))

    # Uniqueness Score
    if report_data.get('uniqueness_score'):
        uniqueness = report_data['uniqueness_score']
        Story.append(Paragraph("✨ Uniqueness of Your Profile", styles['Heading2']))
        Story.append(Paragraph(f"<b>Score</b>: {uniqueness['score']:.2f} ({uniqueness['interpretation']})", styles['BodyText']))
        Story.append(Paragraph(f"<b>Factors</b>: {', '.join(uniqueness['rarity_factors'])}", styles['BodyText']))
        Story.append(Spacer(1, 0.1 * inch))

    # Final Note
    Story.append(PageBreak()) # Start final note on a new page if content is long
    Story.append(Paragraph("Thank you for using Sheelaa's Elite AI Numerology Assistant!", styles['TitleStyle']))
    Story.append(Spacer(1, 0.2 * inch))
    Story.append(Paragraph("Disclaimer: Numerology provides insights and guidance. It is not a substitute for professional advice. Your free will and choices ultimately shape your destiny.", styles['BodyText']))

    doc.build(Story)
    buffer.seek(0)
    return buffer.getvalue()

# --- Flask Routes ---
@app.route('/')
def home():
    """Basic home route for health check."""
    return "Hello from Flask!"

# Helper to run async functions in a synchronous Flask context
def run_async_in_sync(coroutine):
    """
    Runs an async coroutine in a new, isolated event loop.
    This is necessary when calling async functions from a synchronous Flask view
    running under a WSGI server like Gunicorn.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError: # No running loop
        loop = None

    if loop and loop.is_running():
        # If a loop is already running (e.g., in a different thread of Gunicorn worker),
        # create a new loop for this specific task.
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()
            # It's good practice to restore the previous loop if it existed
            if loop:
                asyncio.set_event_loop(loop)
    else:
        # No running loop, safe to use asyncio.run
        return asyncio.run(coroutine)


@app.route('/chat', methods=['POST'])
@rate_limited("30 per minute") # Apply rate limit using the custom decorator
@performance_monitor
def chat(): # Removed 'async' keyword here
    """Handles chat messages and advanced report/validation requests."""
    data = request.json
    message = data.get('message')

    if not message:
        return jsonify({"error": "No message provided"}), 400

    if not SecurityManager.validate_input_security(message):
        logger.warning(f"Security alert: Malicious input detected: {message}")
        return jsonify({"error": "Invalid input. Potential security threat detected."}), 400

    logger.info(f"Received message: {message}")

    # Check for specific commands
    report_request = MessageParser.parse_report_request(message)
    validation_request = MessageParser.parse_validation_request(message)

    try:
        if report_request:
            logger.info(f"Attempting to parse GENERATE_ADVANCED_REPORT message: {message}")
            # Generate comprehensive profile
            profile = get_comprehensive_numerology_profile(report_request.full_name, report_request.birth_date)
            
            # Generate additional insights
            timing_recommendations = generate_timing_recommendations(profile)
            uniqueness_score = calculate_uniqueness_score(profile)
            potential_challenges = identify_potential_challenges(profile)
            development_recommendations = get_development_recommendations(profile)
            yearly_forecast = generate_yearly_forecast(profile)

            # Prepare data for LLM
            llm_input_data = {
                "full_name": report_request.full_name,
                "birth_date": report_request.birth_date,
                "desired_outcome": report_request.desired_outcome,
                "current_expression_number": report_request.current_expression_number,
                "current_life_path_number": report_request.current_life_path_number,
                "profile_details": profile,
                "timing_recommendations": timing_recommendations,
                "uniqueness_score": uniqueness_score,
                "potential_challenges": potential_challenges,
                "development_recommendations": development_recommendations,
                "yearly_forecast": yearly_forecast
            }
            
            # LLM prompt for advanced report
            system_message = SystemMessage(
                content=(
                    "You are Sheelaa's Elite AI Numerology Assistant. Your task is to provide a comprehensive, "
                    "personalized numerology report based on the user's full name, birth date, desired outcome, "
                    "and calculated numerology numbers. Structure the report as follows:\n\n"
                    "**1. Introduction:** A welcoming and personalized opening, acknowledging their name, birth date, and desired outcome.\n"
                    "**2. Core Numerological Insights:** Explain their Expression (Name) Number and Life Path Number, "
                    "drawing from the provided interpretations. Discuss their compatibility.\n"
                    "**3. Deeper Insights (Soul Urge & Personality):** Explain their Soul Urge and Personality Numbers.\n"
                    "**4. Karmic Lessons:** Detail any karmic lessons identified from their name.\n"
                    "**5. Optimal Timing & Energy:** Provide insights into their current personal year and optimal activities.\n"
                    "**6. Potential Challenges & Growth Areas:** Outline potential challenges and suggest mitigation strategies and growth opportunities.\n"
                    "**7. Personalized Development Recommendations:** Offer actionable recommendations for immediate focus, long-term goals, and monthly practices.\n"
                    "**8. Yearly Forecast:** Provide a brief 3-year forecast based on personal year cycles.\n"
                    "**9. Uniqueness of Your Profile:** Comment on the uniqueness of their numerological blueprint.\n"
                    "**10. Conclusion:** A positive and empowering closing statement. \n\n"
                    "Use clear, encouraging language. Format the report using Markdown for headings, bold text, and bullet points. "
                    "Do NOT include any disclaimers about AI generation in the report content itself. "
                    "Do NOT include any API keys or sensitive information. "
                    "Ensure the report is detailed, insightful, and directly addresses the user's desired outcome."
                )
            )
            
            human_message = HumanMessage(
                content=f"Generate a comprehensive numerology report for the following data:\n{json.dumps(llm_input_data, indent=2)}"
            )
            
            messages = [system_message, human_message]
            
            # Get response from creative LLM for the report
            report_response = run_async_in_sync(llm_manager.creative_llm.invoke(messages))
            
            # Store the full report data, including the LLM's response, for PDF generation
            full_report_data = {
                **llm_input_data, # Include all calculated profile data
                "intro_response": report_response.content # This is the LLM generated Markdown report
            }

            return jsonify({"response": report_response.content}), 200

        elif validation_request:
            logger.info(f"Attempting to parse VALIDATE_NAME_ADVANCED message: {message}")
            # Generate comprehensive profile for original name
            original_profile = get_comprehensive_numerology_profile(validation_request.original_full_name, validation_request.birth_date)
            
            # Resolve the full name for calculation based on suggested part
            full_name_for_validation = NameSuggestionEngine.resolve_full_name_for_calculation(
                validation_request.original_full_name,
                validation_request.suggested_name
            )
            
            # Calculate numerology for suggested name
            suggested_expression_num, suggested_expression_details = AdvancedNumerologyCalculator.calculate_expression_number(full_name_for_validation)

            # LLM prompt for name validation
            system_message = SystemMessage(
                content=(
                    "You are Sheelaa's Elite AI Numerology Assistant. Your task is to validate a suggested name "
                    "based on the user's original profile, desired outcome, and the numerology of the suggested name. "
                    "Provide a detailed analysis, including:\n\n"
                    "**1. Suggested Name Analysis:** The numerology of the suggested name (Expression Number) and its core interpretation.\n"
                    "**2. Alignment with Desired Outcome:** How well the suggested name's energy aligns with the user's desired outcome.\n"
                    "**3. Compatibility with Life Path:** How the suggested name's Expression Number interacts with the user's Life Path Number.\n"
                    "**4. Overall Recommendation:** A clear recommendation (e.g., 'Excellent Fit', 'Good Fit, with considerations', 'Not Recommended') and a concise rationale.\n\n"
                    "Use Markdown for formatting. Be encouraging and insightful. Do NOT include any disclaimers about AI generation."
                )
            )
            
            human_message = HumanMessage(
                content=f"Original Full Name: {validation_request.original_full_name}\n"
                        f"Birth Date: {validation_request.birth_date}\n"
                        f"Desired Outcome: {validation_request.desired_outcome}\n"
                        f"Suggested Name to Validate: {validation_request.suggested_name} (Calculated Expression Number: {suggested_expression_num})\n"
                        f"Original Expression Number: {original_profile['expression_number']}\n"
                        f"Original Life Path Number: {original_profile['life_path_number']}\n"
                        f"Suggested Name's Core Interpretation: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(suggested_expression_num, {}).get('core', 'N/A')}"
            )
            
            messages = [system_message, human_message]
            
            # Get response from analytical LLM for validation
            validation_response = run_async_in_sync(llm_manager.analytical_llm.invoke(messages))
            
            return jsonify({"response": validation_response.content}), 200

        else:
            # Fallback for general chat messages
            logger.info(f"Processing general chat message: {message}")
            
            # FIX: Corrected memory usage - directly call add_user_message on memory object
            llm_manager.memory.add_user_message(message)
            
            # Define a simple prompt for general chat
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are Sheelaa's Elite AI Numerology Assistant. You provide general information and guidance about numerology. Do not attempt to calculate numbers or provide personalized reports unless explicitly asked with specific details (full name, birth date). Keep responses concise and helpful."),
                # FIX: Corrected memory usage - directly access messages on memory object
                llm_manager.memory.messages[-1] # Only include the last message for context to avoid excessive token usage
            ])
            
            chain = prompt | llm_manager.llm
            
            # Get response from LLM for general chat
            ai_response = run_async_in_sync(chain.invoke({"input": message}))
            
            # FIX: Corrected memory usage - directly call add_ai_message on memory object
            llm_manager.memory.add_ai_message(ai_response.content)
            
            return jsonify({"response": ai_response.content}), 200

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred. Please try again later."}), 500

@app.route('/generate_pdf_report', methods=['POST'])
@rate_limited("5 per hour") # Limit PDF generation due to resource intensity
@performance_monitor
def generate_pdf_report_endpoint():
    """Endpoint to generate and return a PDF numerology report."""
    data = request.json
    
    # Extract necessary data from the request body
    full_name = data.get('full_name')
    birth_date = data.get('birth_date')
    desired_outcome = data.get('desired_outcome')
    current_expression_number = data.get('current_expression_number')
    current_life_path_number = data.get('current_life_path_number')
    intro_response = data.get('intro_response') # The LLM-generated Markdown report

    if not all([full_name, birth_date, desired_outcome, current_expression_number, current_life_path_number, intro_response]):
        return jsonify({"error": "Missing required data for PDF generation."}), 400

    try:
        # Re-calculate comprehensive profile to ensure all data for PDF is fresh and complete
        # This avoids relying on frontend-only data for complex PDF details
        profile = get_comprehensive_numerology_profile(full_name, birth_date)
        
        # Generate additional insights required for the PDF
        timing_recommendations = generate_timing_recommendations(profile)
        uniqueness_score = calculate_uniqueness_score(profile)
        potential_challenges = identify_potential_challenges(profile)
        development_recommendations = get_development_recommendations(profile)
        yearly_forecast = generate_yearly_forecast(profile)

        pdf_report_data = {
            "full_name": full_name,
            "birth_date": birth_date,
            "desired_outcome": desired_outcome,
            "expression_number": profile['expression_number'],
            "life_path_number": profile['life_path_number'],
            "soul_urge_number": profile['soul_urge_number'],
            "personality_number": profile['personality_number'],
            "compatibility_insights": profile['compatibility_insights'],
            "karmic_lessons": profile['karmic_lessons'],
            "timing_recommendations": timing_recommendations,
            "uniqueness_score": uniqueness_score,
            "potential_challenges": potential_challenges,
            "development_recommendations": development_recommendations,
            "yearly_forecast": yearly_forecast,
            "intro_response": intro_response # Pass the LLM-generated Markdown report
        }

        pdf_bytes = create_numerology_pdf(pdf_report_data)
        
        # Determine filename
        filename = f"Numerology_Report_{full_name.replace(' ', '_')}.pdf"
        
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename # Use download_name for Flask 2.x and newer
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
    logger.error(f"Internal server error: {e}", exc_info=True) # Log full traceback
    return jsonify({"error": "Internal server error. Please try again later."}), 500

@app.errorhandler(404)
def not_found_handler(e):
    return jsonify({"error": "Endpoint not found."}), 404

# --- Application Initialization ---
def initialize_application():
    """Initialize all application components"""
    logger.info("Initializing Numerology Application...")
    
    # Initialize LLM Manager
    if not llm_manager.initialize():
        logger.error("Failed to initialize LLM manager. Exiting application.")
        sys.exit(1) # Exit if LLM manager fails to initialize
    
    logger.info("Application initialized successfully")
    return True

# Call initialization directly when the module is loaded.
# This ensures it runs once when Gunicorn loads the app.
initialize_application()

# Development server initialization
if __name__ == "__main__":
    # When run directly (e.g., python app.py), initialize_application()
    # will have already been called once above. So, no need to call it again here.
    
    logger.info("Starting development server...")
    app.run(
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 5000)), 
        debug=os.getenv("FLASK_DEBUG", "False").lower() == "true"
    )
