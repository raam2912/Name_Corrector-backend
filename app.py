from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import os
from dotenv import load_dotenv
import datetime
import re
import json
import asyncio # Not explicitly used in current routes but often in async Flask
import logging
from functools import wraps
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
from concurrent.futures import ThreadPoolExecutor
import time
import secrets # Not explicitly used in current routes
import hmac # Not explicitly used in current routes
import psutil
from collections import Counter # Not explicitly used in current routes

# Langchain imports (cleaned up to only what's strictly used now)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# Fix for Pydantic v1 deprecation warning
from pydantic.v1 import BaseModel, Field # Changed from langchain_core.pydantic_v1
from langchain.output_parsers import PydanticOutputParser # Still used for structured output

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
app.config['CACHE_TYPE'] = 'simple'  # Use simple cache if Redis not available
app.config['CACHE_DEFAULT_TIMEOUT'] = 300

# Initialize extensions (cache and limiter)
cache = None
limiter = None
try:
    cache = Cache(app)
    # Fix: Ensure key_func is passed correctly and only once.
    # The default_limits argument should not implicitly set key_func.
    limiter = Limiter(
        app,
        key_func=get_remote_address,
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

# --- Data Models (Defined early to avoid NameError) ---
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

# --- Advanced Numerology Calculator (Defined early) ---
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

# --- Security Manager (Defined early) ---
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

# --- Message Parser (Defined early) ---
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
        """Parse GENERATE_ADVANCED_REPORT message format."""
        # Definitive fix for regex: made spaces/punctuation more flexible and final period optional
        pattern = r"My full name is \"(.*?)\"\s*and my birth date is \"(.*?)\"\.\s*My current Name \(Expression\) Number is (\d+)\s*and Life Path Number is (\d+)\.\s*I desire the following positive outcome in my life:\s*\"(.*?)\"\s*\.?"
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

    # Note: RequestType enum and its usage were removed in previous iterations,
    # but if you re-introduce it, ensure it's defined.
    # For now, the if/elif in enhanced_chat handles routing directly based on string prefixes.

# --- Name Suggestion Engine (Defined early) ---
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

# --- LLM Manager (Defined early) ---
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
            logger.error("GOOGLE_API_KEY not found")
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
            
            logger.info("All LLM instances initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            return False

# Global instance of LLM manager (Defined immediately after its class)
llm_manager = LLMManager()

# --- Utility Functions (These can be defined after classes as they don't create instances of the classes directly at global scope) ---
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

# --- API Endpoints ---
@app.route("/health", methods=["GET"])
def health_check():
    """Enhanced health check endpoint"""
    try:
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        system_metrics = {"status": f"metrics unavailable: {e}"}
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "llm_initialized": llm_manager.llm is not None,
        "cache_available": cache is not None,
        "rate_limiter_available": limiter is not None,
        "system_metrics": system_metrics
    }), 200

@app.route("/profile", methods=["POST"])
@rate_limited("10 per minute")
@performance_monitor
def get_numerology_profile_endpoint(): # Renamed to avoid conflict if any
    """Get comprehensive numerology profile"""
    data = request.get_json()
    
    if not data or not data.get("full_name") or not data.get("birth_date"):
        return jsonify({"error": "Both full_name and birth_date are required"}), 400
    
    # Security validation
    full_name = SecurityManager.sanitize_name_input(data["full_name"])
    birth_date = data["birth_date"]
    
    if not SecurityManager.validate_input_security(full_name + birth_date):
        return jsonify({"error": "Invalid input detected"}), 400
    
    try:
        profile = get_comprehensive_numerology_profile(full_name, birth_date)
        timing_recommendations = generate_timing_recommendations(profile)
        profile["timing_recommendations"] = timing_recommendations
        
        return jsonify(profile), 200
        
    except Exception as e:
        logger.error(f"Error generating profile: {e}")
        return jsonify({"error": "Failed to generate numerology profile"}), 500

@app.route("/chat", methods=["POST"])
@rate_limited("30 per minute")
@performance_monitor 
def enhanced_chat():
    """Enhanced chat endpoint with advanced features"""
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    logger.info(f"Received message: {user_message[:100]}...")

    if not llm_manager.llm: # Check if the main LLM is initialized
        return jsonify({"error": "Chatbot not initialized"}), 500

    # Security validation
    if not SecurityManager.validate_input_security(user_message):
        return jsonify({"error": "Invalid input detected"}), 400

    try:
        if user_message.startswith("GENERATE_ADVANCED_REPORT:"):
            return generate_advanced_report(user_message)
        elif user_message.startswith("VALIDATE_NAME_ADVANCED:"):
            return validate_name_advanced(user_message)
        elif user_message.startswith("GET_NAME_SUGGESTIONS:"):
            return generate_name_suggestions(user_message)
        else:
            # Standard chat processing with enhanced context
            system_prompt = """You are Sheelaa's Elite AI Numerology Assistant. You are warm, wise, and deeply knowledgeable about numerology. 
            Provide insightful, practical guidance while maintaining a professional yet caring tone. 
            Always focus on empowering the user and helping them understand their numerological profile."""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = llm_manager.llm.invoke(messages)
            return jsonify({"response": response.content}), 200

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        return jsonify({"error": "Processing failed"}), 500

def generate_advanced_report(user_message: str) -> tuple:
    """Generate advanced numerology report with comprehensive analysis"""
    try:
        # Parse the message to extract user details
        # Definitive fix for regex: made spaces/punctuation more flexible and final period optional
        report_data_match = re.search(
           r"My full name is \"(.*?)\"\s*and my birth date is \"(.*?)\"\.\s*My current Name \(Expression\) Number is (\d+)\s*and Life Path Number is (\d+)\.\s*I desire the following positive outcome in my life:\s*\"(.*?)\"\s*\.?",
            user_message
        )
        
        if not report_data_match:
            return jsonify({"error": "Invalid format for GENERATE_ADVANCED_REPORT message."}), 400
        
        original_full_name, birth_date, current_exp_num_str, current_life_path_num_str, desired_outcome = report_data_match.groups()
        current_exp_num = int(current_exp_num_str)
        current_life_path_num = int(current_life_path_num_str)

        # Get comprehensive profile (this will use caching)
        profile = get_comprehensive_numerology_profile(original_full_name, birth_date)
        
        # Generate introduction
        intro_prompt = f"""
        You are Sheelaa's Elite AI Assistant. Create a warm, comprehensive introduction for a personalized numerology report.
        
        User Details:
        Full Name: "{original_full_name}"
        Birth Date: "{birth_date}"
        Expression Number: {current_exp_num}
        Life Path Number: {current_life_path_num}
        Desired Outcome: "{desired_outcome}"
        
        Using these interpretations:
        Expression {current_exp_num}: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(current_exp_num, {}).get('core', 'Universal energy')}
        Life Path {current_life_path_num}: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(current_life_path_num, {}).get('core', 'Universal energy')}
        
        Write a detailed, empathetic introduction that:
        1. Acknowledges their unique numerological signature. Use **strong bolding** for key numbers and concepts.
        2. Explains how their current numbers work together, highlighting their strengths.
        3. Connects their profile to their desired outcome, emphasizing potential for transformation.
        4. Sets the stage for the insightful name corrections and advanced insights that follow.
        
        Be warm, wise, and encouraging. Use Markdown to structure and emphasize key points. Aim for 3-4 paragraphs.
        """
        
        intro_response = llm_manager.analytical_llm.invoke(intro_prompt).content
        
        # Generate name suggestions using creative LLM
        name_suggestions_prompt = f"""
        As Sheelaa's Elite AI Assistant, generate 8-10 sophisticated name suggestions for someone seeking: "{desired_outcome}"
        
        Original Name: "{original_full_name}"
        Current Expression: {current_exp_num}
        Life Path: {current_life_path_num}
        
        Guidelines:
        - Maintain connection to original identity
        - Ensure cultural appropriateness
        - Focus on names that numerologically support the desired outcome
        - Include variations (new first names, middle names, full name changes)
        - Consider professional usability
        
        Format as a simple list of names, one per line.
        """
        
        suggestions_response = llm_manager.creative_llm.invoke(name_suggestions_prompt).content
        suggested_names = [name.strip('- ').strip() for name in suggestions_response.split('\n') if name.strip()]
        
        # Build the full report
        full_report = f"""# 🌟 Your Personalized Numerology Report

{intro_response}

## ✨ Your Current Numerological Profile

**Expression Number {current_exp_num}**: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(current_exp_num, {}).get('core', 'Universal energy')}

**Life Path Number {current_life_path_num}**: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(current_life_path_num, {}).get('core', 'Universal energy')}

**Compatibility Analysis**: {profile.get('compatibility_insights', {}).get('description', 'Your numbers work in harmony.')}

"""
        # Add other profile details
        if profile.get('soul_urge_number'):
            full_report += f"""
**Soul Urge Number {profile['soul_urge_number']}**: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(profile['soul_urge_number'], {}).get('core', 'Universal energy')}
"""
        if profile.get('personality_number'):
            full_report += f"""
**Personality Number {profile['personality_number']}**: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(profile['personality_number'], {}).get('core', 'Universal energy')}
"""
        if profile.get('expression_details', {}).get('karmic_debt'):
            full_report += f"""
**Karmic Debt (Expression):** Yes (Number {profile['expression_details']['karmic_debt']}) - This indicates a specific lesson related to past actions.
"""
        if profile.get('life_path_details', {}).get('karmic_debt_numbers'):
            full_report += f"""
**Karmic Debt (Life Path):** Yes (Numbers {', '.join(map(str, profile['life_path_details']['karmic_debt_numbers']))}) - These are areas for growth and balance.
"""


        full_report += """
## 🎯 Suggested Name Corrections

"""
        
        # Calculate and explain each suggested name
        for suggested_name in suggested_names[:3]:  # OPTIMIZATION: Further reduced to 3 suggestions
            try:
                calc = AdvancedNumerologyCalculator()
                new_expression, details = calc.calculate_expression_number(suggested_name)
                
                explanation_prompt = f"""
                Explain how the name "{suggested_name}" (Expression Number {new_expression}) supports the goal: "{desired_outcome}"
                
                Original name: "{original_full_name}" (Expression {current_exp_num})
                New Expression interpretation: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(new_expression, {}).get('core', 'Universal energy')}
                
                Write a concise 1-2 sentence explanation of why this name change would be beneficial.
                **Strongly emphasize the new Expression Number and its core benefits** using bold Markdown.
                Focus on practical benefits and energy alignment.
                """
                
                explanation = llm_manager.analytical_llm.invoke(explanation_prompt).content
                
                full_report += f"""
**Suggested Name:** {suggested_name}
**Expression Number:** {new_expression}
**Benefits:** {explanation}

"""
            except Exception as e:
                logger.error(f"Error processing name {suggested_name}: {e}")
                continue
        
        # Add a note about limited suggestions for free tier
        if len(suggested_names) > 3:
            full_report += """
*Note: Only the top 3 suggestions are detailed here to optimize processing for the free tier. For more personalized options, consider a full consultation.*
"""

        # Add timing recommendations if available
        if 'timing_recommendations' in profile:
            timing = profile['timing_recommendations']
            full_report += f"""
## ⏰ Optimal Timing for Name Change

**Current Personal Year:** {timing['current_personal_year']}
**Recommended Activities:** {', '.join(timing['optimal_activities'])}
**Best Implementation Months:** {', '.join(map(str, timing['best_months_for_action']))}

{timing['energy_description']}

"""
        
        # Add karmic lessons
        if profile.get('karmic_lessons', {}).get('lessons'):
            full_report += f"""
## 🔮 Karmic Lessons & Growth Areas

**Areas for Development:** {', '.join(profile['karmic_lessons']['lessons'])}
**Priority Focus:** {profile['karmic_lessons']['priority_lesson']}

"""
        # Add analytics and insights
        insights = {
            "uniqueness_score": calculate_uniqueness_score(profile),
            "success_prediction": predict_success_areas(profile),
            "challenge_analysis": identify_potential_challenges(profile),
            "development_recommendations": get_development_recommendations(profile),
            "yearly_forecast": generate_yearly_forecast(profile, 3)  # 3-year forecast
        }

        full_report += f"""
## 📊 Advanced Insights for Your Journey

**Uniqueness of Your Profile:** {insights['uniqueness_score']['interpretation']} (Score: {insights['uniqueness_score']['score']:.2f})
*Factors:* {', '.join(insights['uniqueness_score']['rarity_factors'])}

**Predicted Success Areas:** {', '.join(insights['success_prediction']['combined_strengths'])}
*Confidence:* {insights['success_prediction']['confidence_level']}

**Potential Challenges:** {', '.join(insights['challenge_analysis']['potential_challenges'])}
*Strategies:* {', '.join(insights['challenge_analysis']['mitigation_strategies'])}

**Development Recommendations:**
*Immediate Focus:* {insights['development_recommendations']['immediate_focus']}
*Long-Term Goal:* {insights['development_recommendations']['long_term_goal']}
*Karmic Work:* {', '.join(insights['development_recommendations']['karmic_work'])}
*Monthly Practices:* {', '.join(insights['development_recommendations']['monthly_practices'])}

"""
        
        full_report += """
### 🗓️ Your 3-Year Numerological Forecast
"""
        for year, data in insights['yearly_forecast'].items():
            full_report += f"""
**{year} (Personal Year {data['personal_year']}):**
*Theme:* {data['theme']}
*Focus:* {data['focus_areas']}
*Optimal Months:* {', '.join(map(str, data['optimal_months']))}
*Energy Level:* {data['energy_level']}
"""

        full_report += "\n---\n*For a detailed consultation and personalized guidance, book your appointment at Sheelaa.com*"
        
        return jsonify({"response": full_report}), 200
        
    except Exception as e:
        logger.error(f"Error generating advanced report: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to generate advanced report"}), 500

def validate_name_advanced(user_message: str) -> tuple:
    """Advanced name validation with comprehensive analysis"""
    try:
        # Parse validation request
        validation_match = re.search(
            r"Original Full Name: \"(.*?)\", Birth Date: \"(.*?)\", Desired Outcome: \"(.*?)\", Suggested Name to Validate: \"(.*?)\"",
            user_message
        )
        
        if not validation_match:
            return jsonify({"error": "Invalid format for VALIDATE_NAME_ADVANCED message."}), 400

        original_full_name, birth_date, desired_outcome, suggested_name = validation_match.groups()
        
        # Determine full name for calculation
        full_name_for_calculation = NameSuggestionEngine.resolve_full_name_for_calculation(
            original_full_name,
            suggested_name
        )
        
        # Calculate numerology for the suggested name
        calc = AdvancedNumerologyCalculator()
        new_expression, expression_details = calc.calculate_expression_number(full_name_for_calculation)
        original_expression, _ = calc.calculate_expression_number(original_full_name)
        life_path, _ = calc.calculate_life_path_number(birth_date)
        
        # Calculate compatibility score
        compatibility = calculate_number_compatibility(new_expression, life_path)
        
        # Determine validation result
        is_valid = compatibility['compatibility_score'] >= 0.7 # Adjusted threshold for "valid"
        status_emoji = "✅" if is_valid else "❌"
        status_text = "Valid for your goals" if is_valid else "Needs optimization"
        
        # Generate comprehensive explanation
        explanation_prompt = f"""
        As Sheelaa's Elite AI Assistant, provide a detailed validation analysis:
        
        Original Name: "{original_full_name}" (Expression {original_expression})
        Suggested Name: "{suggested_name}" → Full Name: "{full_name_for_calculation}" (Expression {new_expression})
        Life Path: {life_path}
        Desired Outcome: "{desired_outcome}"
        Compatibility Score: {compatibility['compatibility_score']:.2f}
        
        Interpretations:
        Original Expression {original_expression}: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(original_expression, {}).get('core', 'Universal energy')}
        New Expression {new_expression}: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(new_expression, {}).get('core', 'Universal energy')}
        
        Explain:
        1. How well the **new expression number ({new_expression})** supports the desired outcome.
        2. The **energetic shift** from original to new name.
        3. **Compatibility** with life path number.
        4. Practical considerations for this name change.
        5. If invalid, suggest what type of energy/number would be better (e.g., "Consider names aligning with 8 or 1 for greater success.").
        
        Use **bold Markdown** for key numbers, concepts, and status indicators. Use bullet points for lists where appropriate.
        Be specific, encouraging, and practical. Write 2-3 concise sentences, clearly structured.
        """
        
        detailed_explanation = llm_manager.analytical_llm.invoke(explanation_prompt).content
        
        # Build validation response
        validation_response = f"""**Suggested Name Validation for '{suggested_name}':**

**Full Name Analyzed:** {full_name_for_calculation}
**Expression Number:** {new_expression}
**Compatibility Score:** {compatibility['compatibility_score']:.2f}/1.0
**Status:** **{status_emoji} {status_text}**

**Detailed Analysis:**
{detailed_explanation}

**Compatibility Insights:** {compatibility['description']}
**Synergy Areas:** {', '.join(compatibility['synergy_areas'])}

**Energy Shift:** Moving from Expression {original_expression} to {new_expression} represents a shift from {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(original_expression, {}).get('psychological', 'current energy')} to {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(new_expression, {}).get('psychological', 'new energy')}.

---
*For a much detailed report, book your appointment using Sheelaa.com.*"""
        
        return jsonify({"response": validation_response}), 200
        
    except Exception as e:
        logger.error(f"Error in advanced name validation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Failed to validate name"}), 500

def generate_name_suggestions(user_message: str) -> tuple:
    """Generate structured name suggestions with rationales"""
    try:
        # Parse request for name suggestions
        suggestion_match = re.search(
            r"Full Name: \"(.*?)\", Birth Date: \"(.*?)\", Desired Outcome: \"(.*?)\"",
            user_message
        )
        
        if not suggestion_match:
            return jsonify({"error": "Invalid format for GET_NAME_SUGGESTIONS message."}), 400

        full_name, birth_date, desired_outcome = suggestion_match.groups()
        
        # Get comprehensive profile
        profile = get_comprehensive_numerology_profile(full_name, birth_date)
        current_expression = profile['expression_number']
        life_path = profile['life_path_number']
        
        # Generate suggestions using structured output
        parser = PydanticOutputParser(pydantic_object=NameSuggestionsOutput)
        
        suggestion_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are Sheelaa's Elite AI Assistant, expert in numerology-based name corrections. 
            Generate culturally appropriate, professionally viable name suggestions that numerologically align with the user's goals."""),
            
            HumanMessage(content=f"""
            Generate 8-10 sophisticated name suggestions for:
            
            Original Name: {full_name}
            Current Expression: {current_expression}
            Life Path: {life_path}
            Desired Outcome: {desired_outcome}
            
            Consider:
            1. Cultural appropriateness and respect
            2. Professional usability in modern contexts
            3. Numerological alignment with desired outcomes
            4. Maintaining connection to original identity
            5. Phonetic harmony and memorability
            
            {parser.get_format_instructions()}
            """)
        ])
        
        try:
            chain = suggestion_prompt | llm_manager.creative_llm | parser
            structured_suggestions = chain.invoke({})
            
            # Format response with calculations
            response = f"""# 🎯 Personalized Name Suggestions for {full_name}

**Current Profile:**
- Expression Number: {current_expression}
- Life Path Number: {life_path}
- Desired Outcome: {desired_outcome}

**Strategic Reasoning:** {structured_suggestions.reasoning}

## ✨ Recommended Name Corrections:

"""
            
            for i, suggestion in enumerate(structured_suggestions.suggestions, 1):
                try:
                    calc = AdvancedNumerologyCalculator()
                    new_expression, _ = calc.calculate_expression_number(suggestion.name)
                    
                    response += f"""
**{i}. {suggestion.name}**
- Expression Number: {new_expression}
- Cultural Appropriateness: {suggestion.cultural_score:.1f}/1.0
- Rationale: {suggestion.rationale}
- Numerological Benefit: {AdvancedNumerologyCalculator.NUMEROLOGY_INTERPRETATIONS.get(new_expression, {}).get('core', 'Universal energy')[:100]}...

"""
                except Exception as e:
                    logger.error(f"Error calculating for suggestion {suggestion.name}: {e}")
                    continue
            
            response += "\n---\n*For detailed analysis of any suggestion, use the name validation feature.*"
            
            return jsonify({"response": response}), 200
            
        except Exception as e:
            logger.error(f"Error with structured output: {e}")
            # Fallback to regular LLM generation
            fallback_prompt = f"""
            Generate 8 sophisticated name suggestions for {full_name} (Expression {current_expression}, Life Path {life_path}) 
            seeking: {desired_outcome}. Format as a numbered list with brief explanations.
            """
            
            fallback_response = llm_manager.creative_llm.invoke(fallback_prompt).content
            return jsonify({"response": fallback_response}), 200
        
    except Exception as e:
        logger.error(f"Error generating name suggestions: {e}")
        return jsonify({"error": "Failed to generate name suggestions"}), 500

# --- Analytics and Insights ---
@app.route("/analytics/profile-insights", methods=["POST"])
@rate_limited("5 per minute")
def get_profile_insights():
    """Get advanced insights and analytics for a profile"""
    data = request.get_json()
    
    if not data or not data.get("full_name") or not data.get("birth_date"):
        return jsonify({"error": "Both full_name and birth_date are required"}), 400
    
    try:
        profile = get_comprehensive_numerology_profile(data["full_name"], data["birth_date"])
        
        # Generate advanced insights
        insights = {
            "uniqueness_score": calculate_uniqueness_score(profile),
            "success_prediction": predict_success_areas(profile),
            "challenge_analysis": identify_potential_challenges(profile),
            "development_recommendations": get_development_recommendations(profile),
            "yearly_forecast": generate_yearly_forecast(profile, 3)  # 3-year forecast
        }
        
        return jsonify(insights), 200
        
    except Exception as e:
        logger.error(f"Error generating profile insights: {e}")
        return jsonify({"error": "Failed to generate insights"}), 500

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

def get_mitigation_strategies(expression: int) -> list:
    """Return mitigation strategies for a given expression number."""
    strategies = {
        1: ["Practice patience", "Collaborate with others", "Balance independence with teamwork"],
        2: ["Build self-confidence", "Set boundaries", "Develop decision-making skills"],
        3: ["Focus on depth", "Commit to projects", "Embrace emotional honesty"],
        4: ["Embrace flexibility", "Try new approaches", "Balance work with relaxation"],
        5: ["Cultivate stability", "Commit to routines", "Practice mindfulness"],
        6: ["Prioritize self-care", "Delegate responsibilities", "Respect others' autonomy"],
        7: ["Connect with others", "Balance analysis with action", "Express emotions openly"],
        8: ["Value relationships", "Balance material and spiritual pursuits", "Practice humility"],
        9: ["Set healthy boundaries", "Accept imperfections", "Practice emotional resilience"],
        11: ["Ground yourself", "Manage expectations", "Practice self-care"],
        22: ["Delegate tasks", "Set realistic goals", "Take breaks to avoid burnout"],
        33: ["Maintain boundaries", "Practice self-love", "Seek support when overwhelmed"]
    }
    return strategies.get(expression, ["General self-improvement strategies"])

def get_growth_opportunities(expression: int) -> list:
    """Return growth opportunities for a given expression number."""
    opportunities = {
        1: ["Leadership development", "Initiative", "Self-reliance"],
        2: ["Cooperation", "Diplomacy", "Emotional intelligence"],
        3: ["Creative expression", "Communication", "Optimism"],
        4: ["Discipline", "Organization", "Building foundations"],
        5: ["Adaptability", "Embracing change", "Exploration"],
        6: ["Nurturing", "Responsibility", "Service to others"],
        7: ["Introspection", "Spiritual growth", "Analytical thinking"],
        8: ["Material success", "Strategic planning", "Leadership"],
        9: ["Compassion", "Humanitarianism", "Wisdom"],
        11: ["Spiritual insight", "Inspiration", "Visionary thinking"],
        22: ["Manifestation", "Practical idealism", "Large-scale achievement"],
        33: ["Healing", "Teaching", "Universal service"]
    }
    return opportunities.get(expression, ["Personal growth", "Self-improvement"])

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
    logger.error(f"Internal server error: {e}")
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
        logger.error("Failed to initialize LLM manager")
        return False
    
    logger.info("Application initialized successfully")
    return True

# Call initialization directly when the module is loaded by Gunicorn
# This replaces @app.before_first_request for Flask 2.3+ compatibility
initialize_application()

# Development server initialization
if __name__ == "__main__":
    # In development, initialize_application() is already called above.
    # We just run the Flask app here.
    logger.info("Starting development server...")
    app.run(
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 5000)),
        debug=os.getenv("FLASK_ENV") == "development" # Debug mode based on FLASK_ENV
    )
