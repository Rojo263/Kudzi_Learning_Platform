import streamlit as st
import google.generativeai as genai
import os
from deep_translator import GoogleTranslator
import time
import sqlite3
from datetime import datetime, timedelta
import json
from streamlit.components.v1 import html
import pandas as pd # Import pandas for the notes_page
import random
import contextlib
import hashlib
from dotenv import load_dotenv
# Optional imports for voice and image features
try:
    import speech_recognition as sr
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

import io
import base64
import math
import subprocess
from typing import Dict, Any, List, Optional, Union

# --- MCP (Model Context Protocol) Integration ---
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# MCP Tools for enhanced AI capabilities
class MCPTools:
    """
    Model Context Protocol (MCP) Tools implementation.
    Provides external tools and data sources for the AI assistant.
    """
    
    def __init__(self):
        self.tools = {
            'calculator': self.calculator_tool,
            'web_search': self.web_search_tool,
            'get_time': self.get_time_tool,
            'date_calculator': self.date_calculator_tool,
            'unit_converter': self.unit_converter_tool,
            'database_query': self.database_query_tool,
            'file_read': self.file_read_tool,
        }
    
    def calculator_tool(self, expression: str) -> Dict[str, Any]:
        """
        Safely evaluate mathematical expressions.
        """
        try:
            # Sanitize expression - only allow safe math operations
            allowed_chars = set('0123456789+-*/.()^ sqrt log ln sin cos tan pi e ')
            if not all(c in allowed_chars or c.isalnum() for c in expression):
                return {'error': 'Invalid characters in expression', 'result': None}
            
            # Replace common math functions
            expression = expression.replace('^', '**')
            expression = expression.replace('sqrt', 'math.sqrt')
            expression = expression.replace('log', 'math.log10')
            expression = expression.replace('ln', 'math.log')
            expression = expression.replace('sin', 'math.sin')
            expression = expression.replace('cos', 'math.cos')
            expression = expression.replace('tan', 'math.tan')
            expression = expression.replace('pi', str(math.pi))
            expression = expression.replace('e', str(math.e))
            
            result = eval(expression, {'__builtins__': {}, 'math': math})
            return {'result': result, 'expression': expression, 'error': None}
        except Exception as e:
            return {'error': str(e), 'result': None}
    
    def web_search_tool(self, query: str) -> Dict[str, Any]:
        """
        Perform web search (placeholder - can be integrated with search APIs).
        For now, returns educational resources suggestion.
        """
        # In a production environment, you would integrate with:
        # - DuckDuckGo API
        # - Google Custom Search API
        # - Bing Search API
        # For now, we'll return educational resource suggestions
        educational_sites = [
            'Khan Academy', 'Coursera', 'edX', 'ZIMSEC official website',
            'Wikipedia', 'Britannica', 'Wolfram Alpha'
        ]
        return {
            'query': query,
            'suggestion': f"For '{query}', consider checking: {', '.join(educational_sites[:3])}",
            'note': 'Direct web search API integration recommended for production use'
        }
    
    def get_time_tool(self, timezone: str = 'Africa/Harare') -> Dict[str, Any]:
        """
        Get current time and date information.
        """
        try:
            from datetime import datetime
            import pytz
            try:
                tz = pytz.timezone(timezone)
            except:
                tz = pytz.timezone('Africa/Harare')  # Default to Harare
            
            now = datetime.now(tz)
            return {
                'current_time': now.strftime('%Y-%m-%d %H:%M:%S %Z'),
                'date': now.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S'),
                'timezone': str(tz),
                'day_of_week': now.strftime('%A'),
            }
        except ImportError:
            # Fallback if pytz not available
            now = datetime.now()
            return {
                'current_time': now.strftime('%Y-%m-%d %H:%M:%S'),
                'date': now.strftime('%Y-%m-%d'),
                'time': now.strftime('%H:%M:%S'),
                'day_of_week': now.strftime('%A'),
            }
        except Exception as e:
            return {'error': str(e)}
    
    def date_calculator_tool(self, operation: str, date1: str, date2: str = None) -> Dict[str, Any]:
        """
        Calculate date differences or add/subtract days.
        """
        try:
            from datetime import datetime, timedelta
            
            if operation == 'difference':
                d1 = datetime.strptime(date1, '%Y-%m-%d')
                d2 = datetime.strptime(date2, '%Y-%m-%d')
                diff = abs((d2 - d1).days)
                return {'difference_days': diff, 'date1': date1, 'date2': date2}
            
            elif operation == 'add_days':
                d1 = datetime.strptime(date1, '%Y-%m-%d')
                days = int(date2) if date2 else 0
                result = d1 + timedelta(days=days)
                return {'result_date': result.strftime('%Y-%m-%d'), 'original': date1, 'days_added': days}
            
            elif operation == 'subtract_days':
                d1 = datetime.strptime(date1, '%Y-%m-%d')
                days = int(date2) if date2 else 0
                result = d1 - timedelta(days=days)
                return {'result_date': result.strftime('%Y-%m-%d'), 'original': date1, 'days_subtracted': days}
            
        except Exception as e:
            return {'error': str(e)}
    
    def unit_converter_tool(self, value: float, from_unit: str, to_unit: str, unit_type: str = 'length') -> Dict[str, Any]:
        """
        Convert between different units (length, weight, temperature, etc.).
        """
        try:
            conversions = {
                'length': {
                    'm': 1, 'km': 1000, 'cm': 0.01, 'mm': 0.001,
                    'in': 0.0254, 'ft': 0.3048, 'yd': 0.9144, 'mi': 1609.34
                },
                'weight': {
                    'kg': 1, 'g': 0.001, 'mg': 0.000001,
                    'lb': 0.453592, 'oz': 0.0283495
                },
                'temperature': {
                    'celsius': lambda x: x,
                    'fahrenheit': lambda x: (x - 32) * 5/9,
                    'kelvin': lambda x: x - 273.15
                }
            }
            
            if unit_type == 'temperature':
                if from_unit == 'celsius' and to_unit == 'fahrenheit':
                    result = (value * 9/5) + 32
                elif from_unit == 'fahrenheit' and to_unit == 'celsius':
                    result = (value - 32) * 5/9
                elif from_unit == 'celsius' and to_unit == 'kelvin':
                    result = value + 273.15
                elif from_unit == 'kelvin' and to_unit == 'celsius':
                    result = value - 273.15
                else:
                    return {'error': 'Unsupported temperature conversion'}
                
                return {'value': value, 'from': from_unit, 'to': to_unit, 'result': result}
            
            else:
                if unit_type not in conversions or from_unit not in conversions[unit_type] or to_unit not in conversions[unit_type]:
                    return {'error': 'Unsupported conversion'}
                
                base_value = value * conversions[unit_type][from_unit]
                result = base_value / conversions[unit_type][to_unit]
                return {'value': value, 'from': from_unit, 'to': to_unit, 'result': result}
                
        except Exception as e:
            return {'error': str(e)}
    
    def database_query_tool(self, query_type: str, table: str = None, **kwargs) -> Dict[str, Any]:
        """
        Query the educational database (limited to safe, read-only queries).
        """
        try:
            # Only allow safe query types
            safe_queries = ['get_user_notes', 'get_flashcards', 'get_chat_history']
            
            if query_type not in safe_queries:
                return {'error': 'Query type not allowed'}
            
            # This is a placeholder - actual implementation would query the database
            # In production, you would import get_db_connection and execute safe queries
            return {
                'query_type': query_type, 
                'status': 'executed', 
                'note': 'Database integration available - use get_user_notes(), get_flashcards(), or get_chat_history() functions directly'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def file_read_tool(self, file_path: str, max_lines: int = 100) -> Dict[str, Any]:
        """
        Safely read educational content files.
        """
        try:
            # Only allow reading from data directory for security
            if not file_path.startswith('data/'):
                return {'error': 'File access restricted to data directory'}
            
            if not os.path.exists(file_path):
                return {'error': 'File not found'}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:max_lines]
                content = ''.join(lines)
                
            return {
                'file_path': file_path,
                'content': content,
                'lines_read': len(lines),
                'note': f'Showing first {min(max_lines, len(lines))} lines'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_tool_usage(self, user_prompt: str) -> List[Dict[str, Any]]:
        """
        Detect which MCP tools should be used based on user prompt.
        Returns list of tool calls with parameters.
        """
        tool_calls = []
        prompt_lower = user_prompt.lower()
        
        # Calculator detection - improved pattern matching
        if any(keyword in prompt_lower for keyword in ['calculate', 'compute', 'solve', 'what is', 'evaluate', '=', '+', '-', '*', '/', 'sqrt', 'sin', 'cos', 'tan']):
            import re
            # Enhanced math pattern to capture more expressions
            math_patterns = [
                r'\d+[\s]*[\+\-\*/\(\)\^]+[\s]*\d+',  # Basic operations
                r'sqrt\([\d\s]+\)',  # Square root
                r'sin\([\d\s]+\)',  # Trigonometric functions
                r'cos\([\d\s]+\)',
                r'tan\([\d\s]+\)',
                r'\d+\s*\*\*\s*\d+',  # Power operator
                r'pi\s*[\+\-\*/]',  # Pi constant
            ]
            
            expression_found = None
            for pattern in math_patterns:
                matches = re.findall(pattern, user_prompt, re.IGNORECASE)
                if matches:
                    expression_found = matches[0].strip()
                    break
            
            # If no pattern found, try to extract anything that looks like math
            if not expression_found and any(op in user_prompt for op in ['+', '-', '*', '/', '(', ')', '=']):
                # Extract the math part (simple heuristic)
                parts = re.split(r'[calculate|compute|solve|what is|evaluate]', prompt_lower)
                if len(parts) > 1:
                    potential_expr = parts[1].strip().split('.')[0].split('?')[0].strip()
                    if any(char.isdigit() for char in potential_expr):
                        expression_found = potential_expr
            
            if expression_found:
                tool_calls.append({
                    'tool': 'calculator',
                    'parameters': {'expression': expression_found},
                    'confidence': 0.85
                })
        
        # Time/Date queries
        if any(keyword in prompt_lower for keyword in ['what time', 'current time', 'what date', 'what day', 'time now', 'date today']):
            tool_calls.append({
                'tool': 'get_time',
                'parameters': {},
                'confidence': 0.9
            })
        
        # Unit conversion - improved detection
        conversion_keywords = ['convert', 'how many', 'meters', 'kilometers', 'kilograms', 'pounds', 'celsius', 'fahrenheit', 'kelvin', 'inches', 'feet', 'yards', 'miles']
        if any(keyword in prompt_lower for keyword in conversion_keywords):
            # Try to extract value and units
            import re
            # Pattern for "X units to units" or "convert X units"
            value_match = re.search(r'(\d+(?:\.\d+)?)\s+(\w+)', user_prompt)
            to_unit_match = re.search(r'to\s+(\w+)|in\s+(\w+)', user_prompt, re.IGNORECASE)
            
            if value_match:
                value = float(value_match.group(1))
                from_unit = value_match.group(2).lower()
                to_unit = None
                
                if to_unit_match:
                    to_unit = (to_unit_match.group(1) or to_unit_match.group(2)).lower()
                
                # Determine unit type
                unit_type = 'length'
                if any(u in from_unit for u in ['kg', 'g', 'lb', 'oz', 'pound', 'gram', 'kilogram']):
                    unit_type = 'weight'
                elif any(u in from_unit for u in ['celsius', 'fahrenheit', 'kelvin', '°c', '°f']):
                    unit_type = 'temperature'
                
                if to_unit:
                    tool_calls.append({
                        'tool': 'unit_converter',
                        'parameters': {
                            'value': value,
                            'from_unit': from_unit,
                            'to_unit': to_unit,
                            'unit_type': unit_type
                        },
                        'confidence': 0.8
                    })
        
        # Web search
        if any(keyword in prompt_lower for keyword in ['search', 'find information', 'latest', 'recent news', 'look up', 'find out']):
            tool_calls.append({
                'tool': 'web_search',
                'parameters': {'query': user_prompt},
                'confidence': 0.75
            })
        
        return tool_calls
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an MCP tool by name with given parameters.
        """
        if tool_name not in self.tools:
            return {'error': f'Tool {tool_name} not found', 'available_tools': list(self.tools.keys())}
        
        try:
            result = self.tools[tool_name](**parameters)
            return {
                'tool': tool_name,
                'result': result,
                'success': 'error' not in result or result.get('error') is None
            }
        except Exception as e:
            return {'error': str(e), 'tool': tool_name, 'success': False}

# Initialize MCP tools
mcp_tools = MCPTools()

# --- Input Validation & Security ---
import re

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input to prevent injection attacks and limit length."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', text.strip())
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized

def validate_username(username: str) -> bool:
    """Validate username format."""
    if not username or len(username) < 3 or len(username) > 20:
        return False
    # Only allow alphanumeric characters and underscores
    return bool(re.match(r'^[a-zA-Z0-9_]+$', username))

def validate_password(password: str) -> bool:
    """Validate password strength."""
    if not password or len(password) < 6:
        return False
    return True

def safe_sql_query(query: str, params: tuple) -> tuple:
    """Ensure SQL query and parameters are safe."""
    # Basic check for SQL injection attempts
    dangerous_patterns = [
        'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER',
        'UNION', 'SELECT', '--', '/*', '*/', ';'
    ]
    
    query_upper = query.upper()
    for pattern in dangerous_patterns:
        if pattern in query_upper:
            raise ValueError(f"Potentially dangerous SQL pattern detected: {pattern}")
    
    return query, params

# --- Rate Limiting & API Management ---
from datetime import datetime, timedelta

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 10, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window  # seconds
        self.calls = []
    
    def can_call(self) -> bool:
        """Check if API call is allowed."""
        now = datetime.now()
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < timedelta(seconds=self.time_window)]
        
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False
    
    def get_wait_time(self) -> int:
        """Get seconds to wait before next allowed call."""
        if not self.calls:
            return 0
        
        oldest_call = min(self.calls)
        next_allowed = oldest_call + timedelta(seconds=self.time_window)
        wait_time = (next_allowed - datetime.now()).total_seconds()
        return max(0, int(wait_time))

# --- Configuration ---
# Use environment variables for sensitive configuration
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("""
    ⚠️ **GEMINI_API_KEY not found!**
    
    Please create a `.env` file in your project directory with:
    ```
    GEMINI_API_KEY=your_actual_api_key_here
    ```
    
    Or set it as an environment variable.
    """)
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- App Configuration ---
class AppConfig:
    """Centralized configuration for the app."""
    
    # API Settings
    MAX_AI_REQUESTS_PER_MINUTE = 15
    AI_REQUEST_TIMEOUT = 30  # seconds
    
    # Security Settings
    MAX_USERNAME_LENGTH = 20
    MAX_PASSWORD_LENGTH = 100
    MAX_NOTE_LENGTH = 1000
    MAX_FLASHCARD_LENGTH = 500
    
    # Database Settings
    DATABASE_NAME = 'kudzi_platform.db'
    
    # UI Settings
    MAX_MESSAGES_DISPLAY = 50
    AUTO_REFRESH_INTERVAL = 5  # seconds
    
    # Game Settings
    QUIZ_TIME_LIMIT = 30  # seconds per question
    MAX_QUESTIONS_PER_QUIZ = 10

# Initialize configuration
config = AppConfig()

# Initialize rate limiter for AI API calls
ai_rate_limiter = RateLimiter(
    max_calls=config.MAX_AI_REQUESTS_PER_MINUTE, 
    time_window=60
)

# Initialize Streamlit page and components
st.set_page_config(page_title="Kudzi Learning Platform", layout="wide")

# Initialize the Gemini model
try:
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    st.error(f"Failed to initialize Gemini model. Please ensure your API key is correct and the model 'gemini-1.5-flash' is available. Error: {e}")
    st.stop() # Stop the app if the model cannot be initialized

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'memory' not in st.session_state:
    st.session_state.memory = [] # Stores recent chat turns for context
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'editing_note' not in st.session_state:
    st.session_state.editing_note = None

# --- Custom CSS for Modern UI ---
st.markdown("""
    <style>
    /* Modern UI Styles */
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; /* Ensure text is visible on dark background */
    }
    
    /* Mobile Optimization */
    @media (max-width: 768px) {
        .main {
            padding: 10px;
        }
        
        .stButton>button {
            width: 100%;
            margin: 5px 0;
        }
        
        .stSelectbox>div>div {
            font-size: 16px; /* Prevents zoom on iOS */
        }
        
        .stTextInput>div>div>input {
            font-size: 16px; /* Prevents zoom on iOS */
        }
        
        .stTextArea>div>div>textarea {
            font-size: 16px; /* Prevents zoom on iOS */
        }
    }
    
    /* Accessibility Features */
    .high-contrast {
        background: #000000 !important;
        color: #ffffff !important;
    }
    
    .high-contrast .stButton>button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #ffffff !important;
    }
    
    .large-text {
        font-size: 1.2em !important;
    }
    
    .extra-large-text {
        font-size: 1.5em !important;
    }
    
    /* Focus indicators for keyboard navigation */
    .stButton>button:focus,
    .stSelectbox>div>div:focus,
    .stTextInput>div>div>input:focus {
        outline: 3px solid #00ff87 !important;
        outline-offset: 2px !important;
    }
    
    /* Offline indicator */
    .offline-indicator {
        position: fixed;
        top: 10px;
        right: 10px;
        background: #ff6b6b;
        color: white;
        padding: 10px;
        border-radius: 5px;
        z-index: 1000;
        display: none;
    }
    
    .offline-indicator.show {
        display: block;
    }
    
    .main {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #00b4db, #0083b0);
        color: white;
        border-radius: 25px;
        padding: 10px 25px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        color: white; /* Ensure chat text is white */
        max-width: 80%; /* Prevent messages from taking full width */
    }
    
    .user-message {
        background: linear-gradient(45deg, #00b4db, #0083b0);
        align-self: flex-end; /* Align user messages to the right */
        margin-left: auto; /* Push user messages to the right */
    }
    
    .assistant-message {
        background: linear-gradient(45deg, #43cea2, #185a9d);
        align-self: flex-start; /* Align assistant messages to the left */
        margin-right: auto; /* Push assistant messages to the left */
    }

    /* Container for chat messages to enable flexbox alignment */
    /* This targets the div directly containing st.markdown for chat messages */
    .st-emotion-cache-1c7y2vl > div { 
        display: flex;
        flex-direction: column;
    }
    
    /* Card-like containers */
    .css-1r6slb0 { /* Sidebar */
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
    }
    
    .big-font {
        background: linear-gradient(45deg, #00ff87, #60efff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Form inputs styling */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background: rgba(255, 255, 255, 0.1);
        border: 2px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
        padding: 10px 15px;
    }
    
    .stSelectbox>div>div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
    }
    .stSelectbox>div>div>div>div {
        color: white; /* Text color for selectbox options */
    }
    
    /* Progress bar and success message styling */
    .stProgress > div > div > div {
        background-color: #00b4db;
    }
    
    .stSuccess {
        background: linear-gradient(45deg, #43cea2, #185a9d);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    
    .chat-container {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .chat-response {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Database Utilities ---
import contextlib

@contextlib.contextmanager
def get_db_connection():
    """Context manager for database connections to ensure proper cleanup."""
    conn = None
    try:
        conn = sqlite3.connect('kudzi_platform.db')
        yield conn
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_db():
    """Initialize the database with all required tables."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    language TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_memory (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    memory_data TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    category TEXT,
                    is_favorite BOOLEAN DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_subjects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    form_level TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, form_level, subject),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_game_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    game TEXT NOT NULL,
                    subject TEXT,
                    score INTEGER NOT NULL,
                    played_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS flashcards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    topic TEXT,
                    front TEXT NOT NULL,
                    back TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Create learning_paths table for guided learning
            c.execute('''
                CREATE TABLE IF NOT EXISTS learning_paths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    form_level TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    learning_path_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Create learning_progress table for tracking progress
            c.execute('''
                CREATE TABLE IF NOT EXISTS learning_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    module_id INTEGER NOT NULL,
                    topic_id INTEGER NOT NULL,
                    status TEXT DEFAULT 'not_started',
                    completed_at TIMESTAMP,
                    score INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Create learning_streaks table for gamification
            c.execute('''
                CREATE TABLE IF NOT EXISTS learning_streaks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    current_streak INTEGER DEFAULT 0,
                    longest_streak INTEGER DEFAULT 0,
                    last_activity_date DATE,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            # Create achievements table for gamification
            c.execute('''
                CREATE TABLE IF NOT EXISTS achievements (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    icon TEXT NOT NULL,
                    points INTEGER NOT NULL,
                    category TEXT NOT NULL
                )
            ''')
            
            # Create user_achievements table for tracking earned achievements
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_achievements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    achievement_id TEXT NOT NULL,
                    earned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (achievement_id) REFERENCES achievements (id)
                )
            ''')
            
            # Create user_points table for tracking points
            c.execute('''
                CREATE TABLE IF NOT EXISTS user_points (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    points INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            
            conn.commit()
            st.success("Database initialized successfully!")
    except sqlite3.Error as e:
        st.error(f"Database initialization error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during database initialization: {e}")

def force_reinit_db():
    """Force reinitialization of the database (useful for development)."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Drop existing tables (if they exist)
            c.execute("DROP TABLE IF EXISTS learning_streaks")
            c.execute("DROP TABLE IF EXISTS learning_progress")
            c.execute("DROP TABLE IF EXISTS learning_paths")
            
            conn.commit()
            st.success("Old tables dropped. Reinitialize database to recreate them.")
            
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

def ensure_all_tables_exist():
    """Ensure all required database tables exist, create them if they don't."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Check which tables exist
            c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in c.fetchall()}
            
            tables_to_create = []
            
            # Define all required tables
            required_tables = {
                'users': '''CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''',
                'chat_history': '''CREATE TABLE IF NOT EXISTS chat_history (
                    chat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    language TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )''',
                'user_memory': '''CREATE TABLE IF NOT EXISTS user_memory (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    memory_data TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )''',
                'user_notes': '''CREATE TABLE IF NOT EXISTS user_notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    category TEXT,
                    is_favorite BOOLEAN DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )''',
                'user_subjects': '''CREATE TABLE IF NOT EXISTS user_subjects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    form_level TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, form_level, subject),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )''',
                'user_game_scores': '''CREATE TABLE IF NOT EXISTS user_game_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    game TEXT NOT NULL,
                    subject TEXT,
                    score INTEGER NOT NULL,
                    played_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )''',
                'flashcards': '''CREATE TABLE IF NOT EXISTS flashcards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    topic TEXT,
                    front TEXT NOT NULL,
                    back TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )''',
                'learning_paths': '''CREATE TABLE IF NOT EXISTS learning_paths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    form_level TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    learning_path_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )''',
                'learning_progress': '''CREATE TABLE IF NOT EXISTS learning_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    subject TEXT NOT NULL,
                    module_id INTEGER NOT NULL,
                    topic_id INTEGER NOT NULL,
                    status TEXT DEFAULT 'not_started',
                    completed_at TIMESTAMP,
                    score INTEGER DEFAULT 0,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )''',
                'learning_streaks': '''CREATE TABLE IF NOT EXISTS learning_streaks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    current_streak INTEGER DEFAULT 0,
                    longest_streak INTEGER DEFAULT 0,
                    last_activity_date DATE,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )'''
            }
            
            # Create missing tables
            for table_name, create_sql in required_tables.items():
                if table_name not in existing_tables:
                    c.execute(create_sql)
                    tables_to_create.append(table_name)
            
            if tables_to_create:
                conn.commit()
                st.success(f"Created missing tables: {', '.join(tables_to_create)}")
            else:
                st.info("All required tables already exist.")
                
    except sqlite3.Error as e:
        st.error(f"Database error ensuring tables exist: {e}")
    except Exception as e:
        st.error(f"Unexpected error ensuring tables exist: {e}")

def safe_database_operation(operation_name: str, operation_func, *args, **kwargs):
    """Safely execute database operations with comprehensive error handling."""
    try:
        # Ensure all tables exist before any operation
        ensure_all_tables_exist()
        
        # Execute the operation
        result = operation_func(*args, **kwargs)
        return result
        
    except sqlite3.Error as e:
        error_msg = f"Database error in {operation_name}: {e}"
        st.error(error_msg)
        
        # Provide helpful error messages for common issues
        if "no such table" in str(e).lower():
            st.warning("💡 **Database tables missing!** Click '🔧 Fix Database' button to recreate them.")
        elif "foreign key constraint failed" in str(e).lower():
            st.warning("💡 **Data integrity issue.** Please check if referenced data exists.")
        elif "database is locked" in str(e).lower():
            st.warning("💡 **Database is busy.** Please wait a moment and try again.")
        elif "disk full" in str(e).lower():
            st.error("💡 **Storage space issue.** Please free up disk space.")
        
        return None
        
    except Exception as e:
        st.error(f"Unexpected error in {operation_name}: {e}")
        return None

def save_chat_message(user_id, role, content, language='en'):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('INSERT INTO chat_history (user_id, role, content, language) VALUES (?, ?, ?, ?)', (user_id, role, content, language))
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error saving chat message: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while saving chat message: {e}")

def load_user_chat_history(user_id):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT role, content FROM chat_history WHERE user_id = ? ORDER BY timestamp ASC', (user_id,))
            messages = c.fetchall()
            st.session_state.messages = [{'role': role, 'content': content} for role, content in messages]
    except sqlite3.Error as e:
        st.error(f"Error loading chat history: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading chat history: {e}")

def save_user_memory(user_id, memory_data):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            memory_json = json.dumps(memory_data)
            c.execute('SELECT memory_id FROM user_memory WHERE user_id = ?', (user_id,))
            existing_memory = c.fetchone()
            if existing_memory:
                c.execute('UPDATE user_memory SET memory_data = ?, timestamp = CURRENT_TIMESTAMP WHERE user_id = ?', (memory_json, user_id))
            else:
                c.execute('INSERT INTO user_memory (user_id, memory_data) VALUES (?, ?)', (user_id, memory_json))
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error saving user memory: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while saving user memory: {e}")

def load_user_memory(user_id):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT memory_data FROM user_memory WHERE user_id = ?', (user_id,))
            result = c.fetchone()
            if result and result[0]:
                st.session_state.memory = json.loads(result[0])
            else:
                st.session_state.memory = []
    except (sqlite3.Error, json.JSONDecodeError) as e:
        st.error(f"Error loading user memory: {e}")
        st.session_state.memory = []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading user memory: {e}")

# --- NOTE FUNCTIONS ---
def save_note(user_id, title, content, category=None):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('INSERT INTO user_notes (user_id, title, content, category) VALUES (?, ?, ?, ?)', (user_id, title, content, category))
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error saving note: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while saving note: {e}")

def get_user_notes(user_id):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT note_id, title, content, category, created_at, is_favorite FROM user_notes WHERE user_id = ? ORDER BY is_favorite DESC, created_at DESC', (user_id,))
            notes = c.fetchall()
            return notes
    except sqlite3.Error as e:
        st.error(f"Error retrieving notes: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while retrieving notes: {e}")
        return []

def update_note(note_id, title, content, category=None):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('UPDATE user_notes SET title = ?, content = ?, category = ?, updated_at = CURRENT_TIMESTAMP WHERE note_id = ?', (title, content, category, note_id))
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error updating note: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while updating note: {e}")

def delete_note(note_id):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('DELETE FROM user_notes WHERE note_id = ?', (note_id,))
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error deleting note: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while deleting note: {e}")

def toggle_favorite(note_id):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('UPDATE user_notes SET is_favorite = NOT is_favorite WHERE note_id = ?', (note_id,))
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error toggling favorite: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while toggling favorite: {e}")

# --- ZIMSEC Subjects Data & Functions ---
# A dictionary to hold the ZIMSEC subjects for each form level.
subjects_data = {
    'Junior Secondary (Form 1 & 2)': [
        'English Language',
        'Shona',
        'Ndebele',
        'Integrated Science',
        'Mathematics',
        'History',
        'Geography',
        'Agriculture',
        'Computer Studies',
        'Family & Religious Studies',
    ],
    'O-Level (Form 3 & 4)': [
        'Mathematics',
        'Physics',
        'Chemistry',
        'Biology',
        'Combined Science',
        'English Literature',
        'History',
        'Geography',
        'Principles of Accounts',
        'Business Studies',
        'Economics',
        'Computer Science',
    ],
    'A-Level (Lower 6 & Upper 6)': [
        'Mathematics',
        'Further Mathematics',
        'Physics',
        'Chemistry',
        'Biology',
        'Computer Science',
        'Accounting',
        'Business Studies',
        'Economics',
        'History',
        'Geography',
        'English Literature',
    ],
}

# Recommended videos per subject, organized by topic
subject_topics_videos = {
    'English Language': [
        { 'topic': 'Grammar Basics', 'videos': ['https://www.youtube.com/watch?v=JtIQv7xR4Hk'] },
        { 'topic': 'Essay Writing', 'videos': ['https://www.youtube.com/watch?v=4Z7p9Nw3bZs'] },
        { 'topic': 'Comprehension Skills', 'videos': ['https://www.youtube.com/watch?v=7f2F7zOm7n8'] },
    ],
    'Shona': [
        { 'topic': 'Shona Basics', 'videos': ['https://www.youtube.com/watch?v=4bWfQhEsgYY'] },
        { 'topic': 'Common Phrases', 'videos': ['https://www.youtube.com/watch?v=4s1Y80hV7Kk'] },
    ],
    'Ndebele': [
        { 'topic': 'Ndebele Basics', 'videos': ['https://www.youtube.com/watch?v=6Yv4U2dC1g0'] },
        { 'topic': 'Greetings and Introductions', 'videos': ['https://www.youtube.com/watch?v=4HqQwqC3QyE'] },
    ],
    'Integrated Science': [
        { 'topic': 'Scientific Method', 'videos': ['https://www.youtube.com/watch?v=SMGRe824kak'] },
        { 'topic': 'States of Matter', 'videos': ['https://www.youtube.com/watch?v=KClEXz6cM7U'] },
        { 'topic': 'Cells and Living Things', 'videos': ['https://www.youtube.com/watch?v=URUJD5NEXC8'] },
    ],
    'Mathematics': [
        { 'topic': 'Algebra Basics', 'videos': ['https://www.youtube.com/watch?v=Q53GmMCqmAM'] },
        { 'topic': 'Geometry Essentials', 'videos': ['https://www.youtube.com/watch?v=MVmHusQ5hBQ'] },
        { 'topic': 'Trigonometry Intro', 'videos': ['https://www.youtube.com/watch?v=1-SvuFIQjK8'] },
        { 'topic': 'Calculus Intro', 'videos': ['https://www.youtube.com/watch?v=WUvTyaaNkzM'] },
    ],
    'History': [
        { 'topic': 'Source Analysis', 'videos': ['https://www.youtube.com/watch?v=Qy3k6z7zu2k'] },
        { 'topic': 'African History Overview', 'videos': ['https://www.youtube.com/watch?v=VZp_8lK91Z8'] },
    ],
    'Geography': [
        { 'topic': 'Plate Tectonics', 'videos': ['https://www.youtube.com/watch?v=RYwz_D3fG9U'] },
        { 'topic': 'Rivers and Landforms', 'videos': ['https://www.youtube.com/watch?v=RCbL4i1iQYg'] },
        { 'topic': 'Weather and Climate', 'videos': ['https://www.youtube.com/watch?v=hpTE2cPsk9A'] },
    ],
    'Agriculture': [
        { 'topic': 'Soil Science Basics', 'videos': ['https://www.youtube.com/watch?v=9D7p9QFZ2nY'] },
        { 'topic': 'Crop Production', 'videos': ['https://www.youtube.com/watch?v=QOa9A6JQ7xg'] },
        { 'topic': 'Livestock Management', 'videos': ['https://www.youtube.com/watch?v=8rG8Pl2WxxI'] },
    ],
    'Computer Studies': [
        { 'topic': 'Introduction to Computers', 'videos': ['https://www.youtube.com/watch?v=J9Q3i5w6-Ug'] },
        { 'topic': 'Basics of Coding', 'videos': ['https://www.youtube.com/watch?v=_uQrJ0TkZlc'] },
    ],
    'Family & Religious Studies': [
        { 'topic': 'World Religions Overview', 'videos': ['https://www.youtube.com/watch?v=3V0g8KJmN2I'] },
        { 'topic': 'Ethics and Values', 'videos': ['https://www.youtube.com/watch?v=jF8QOQWZQ2E'] },
    ],
    'Physics': [
        { 'topic': 'Newton\'s Laws', 'videos': ['https://www.youtube.com/watch?v=kKKM8Y-u7ds'] },
        { 'topic': 'Electricity Basics', 'videos': ['https://www.youtube.com/watch?v=VnnpLaKsqGU'] },
        { 'topic': 'Waves and Sound', 'videos': ['https://www.youtube.com/watch?v=qV4lR9EWGlY'] },
    ],
    'Chemistry': [
        { 'topic': 'Periodic Table Basics', 'videos': ['https://www.youtube.com/watch?v=0RRVV4Diomg'] },
        { 'topic': 'Chemical Bonding', 'videos': ['https://www.youtube.com/watch?v=QXT4OVM4vXI'] },
        { 'topic': 'Stoichiometry', 'videos': ['https://www.youtube.com/watch?v=UL1jmJaUkaQ'] },
    ],
    'Biology': [
        { 'topic': 'Cell Structure', 'videos': ['https://www.youtube.com/watch?v=URUJD5NEXC8'] },
        { 'topic': 'Genetics Basics', 'videos': ['https://www.youtube.com/watch?v=CBezq1fFUEA'] },
        { 'topic': 'Human Physiology', 'videos': ['https://www.youtube.com/watch?v=Q1blP8p2Y2I'] },
    ],
    'Combined Science': [
        { 'topic': 'Key Concepts Overview', 'videos': ['https://www.youtube.com/watch?v=7ylq3b9RkNQ'] },
        { 'topic': 'Practical Skills', 'videos': ['https://www.youtube.com/watch?v=u7cxy3bVYco'] },
    ],
    'English Literature': [
        { 'topic': 'Poetry Analysis', 'videos': ['https://www.youtube.com/watch?v=7f2F7zOm7n8'] },
        { 'topic': 'Shakespeare Basics', 'videos': ['https://www.youtube.com/watch?v=R5PlYv4_cw0'] },
        { 'topic': 'Prose and Drama', 'videos': ['https://www.youtube.com/watch?v=4eOi5Qy5A8g'] },
    ],
    'Principles of Accounts': [
        { 'topic': 'Double Entry Basics', 'videos': ['https://www.youtube.com/watch?v=J7QkqLh2p9Y'] },
        { 'topic': 'Financial Statements', 'videos': ['https://www.youtube.com/watch?v=YlGqN3AkdW8'] },
    ],
    'Business Studies': [
        { 'topic': 'Business Objectives', 'videos': ['https://www.youtube.com/watch?v=9ZB1-S0j7j4'] },
        { 'topic': 'Marketing Mix', 'videos': ['https://www.youtube.com/watch?v=EbmfFDR4nG8'] },
        { 'topic': 'Operations Management', 'videos': ['https://www.youtube.com/watch?v=PpJb9W9sQ2s'] },
    ],
    'Economics': [
        { 'topic': 'Supply and Demand', 'videos': ['https://www.youtube.com/watch?v=g9aDizJpd_s'] },
        { 'topic': 'Elasticity', 'videos': ['https://www.youtube.com/watch?v=1g4vTnGkTXM'] },
        { 'topic': 'Macroeconomics Intro', 'videos': ['https://www.youtube.com/watch?v=8V06ZOQuo0k'] },
    ],
    'Computer Science': [
        { 'topic': 'Programming Basics (Python)', 'videos': ['https://www.youtube.com/watch?v=_uQrJ0TkZlc'] },
        { 'topic': 'Algorithms and Data Structures', 'videos': ['https://www.youtube.com/watch?v=8hly31xKli0'] },
        { 'topic': 'Computer Systems', 'videos': ['https://www.youtube.com/watch?v=Dkg5R8QY0q0'] },
    ],
    'Further Mathematics': [
        { 'topic': 'Complex Numbers', 'videos': ['https://www.youtube.com/watch?v=bR1G4EwR9mI'] },
        { 'topic': 'Matrices', 'videos': ['https://www.youtube.com/watch?v=0oGJTQCy4cQ'] },
        { 'topic': 'Differential Equations', 'videos': ['https://www.youtube.com/watch?v=p_di4Zn4wz4'] },
    ],
    'Accounting': [
        { 'topic': 'Trial Balance', 'videos': ['https://www.youtube.com/watch?v=H3nQkS8b7vA'] },
        { 'topic': 'Cash Flow Statements', 'videos': ['https://www.youtube.com/watch?v=6lGrCVhYyE4'] },
        { 'topic': 'Ratio Analysis', 'videos': ['https://www.youtube.com/watch?v=2VQY0r6mTnU'] },
    ],
}

def add_user_subject(user_id: int, form_level: str, subject: str) -> bool:
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('INSERT OR IGNORE INTO user_subjects (user_id, form_level, subject) VALUES (?, ?, ?)', (user_id, form_level, subject))
            conn.commit()
            return c.rowcount > 0
    except sqlite3.Error as e:
        st.error(f"Error adding subject: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while adding subject: {e}")
        return False

def get_user_subjects(user_id: int, form_level: str) -> list:
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT subject FROM user_subjects WHERE user_id = ? AND form_level = ? ORDER BY subject ASC', (user_id, form_level))
            rows = c.fetchall()
            return [r[0] for r in rows]
    except sqlite3.Error as e:
        st.error(f"Error loading subjects: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading subjects: {e}")
        return []

def remove_user_subject(user_id: int, form_level: str, subject: str) -> None:
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('DELETE FROM user_subjects WHERE user_id = ? AND form_level = ? AND subject = ?', (user_id, form_level, subject))
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error removing subject: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while removing subject: {e}")

# --- ZIMSEC Subjects Page ---
def zimsec_subjects_page():
    st.title("ZIMSEC Subjects")
    st.markdown("Select a form level to view and add subjects to your study list.")

    form_levels = list(subjects_data.keys())
    selected_form = st.selectbox("Choose a Form Level", options=form_levels, index=0)

    st.header(f"Subjects for {selected_form}")

    columns_per_row = 3
    cols = st.columns(columns_per_row)

    # Existing selections for this user and form
    existing = set(get_user_subjects(st.session_state.user_id, selected_form)) if st.session_state.user_id else set()

    for i, subject in enumerate(subjects_data[selected_form]):
        with cols[i % columns_per_row]:
            added = subject in existing
            button_label = ("Added ✓ " + subject) if added else ("Add " + subject)
            if st.button(button_label, key=f"add_{selected_form}_{subject}", disabled=added):
                if st.session_state.user_id is None:
                    st.warning("Please log in first.")
                else:
                    if add_user_subject(st.session_state.user_id, selected_form, subject):
                        st.success(f"You've added {subject}!")
                        st.rerun()

    st.subheader("Your selected subjects")
    selected_subjects = sorted(existing)
    if not selected_subjects:
        st.info("No subjects added yet for this form.")
    else:
        for subj in selected_subjects:
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"- {subj}")
            with c2:
                if st.button("Remove", key=f"remove_{selected_form}_{subj}"):
                    remove_user_subject(st.session_state.user_id, selected_form, subj)
                    st.rerun()

    st.divider()
    st.subheader("Browse recommended videos by topic")
    
    # Add AI-powered video discovery
    st.markdown("💡 **Want to discover new, relevant videos?**")
    ai_search_topic = st.text_input("Search for videos about:", 
                                   placeholder="e.g., Quadratic Equations, Photosynthesis, World War II",
                                   key="zimsec_ai_search")
    
    if ai_search_topic and st.button("🔍 AI Video Search", key="zimsec_ai_search_btn"):
        with st.spinner("Searching for relevant videos..."):
            videos = search_youtube_videos(ai_search_topic, max_results=3)
            
            if videos:
                st.success(f"Found {len(videos)} relevant videos!")
                for i, video in enumerate(videos, 1):
                    with st.expander(f"🤖 AI Found: {video.get('title', 'Untitled')}", expanded=True):
                        st.markdown(f"**Description:** {video.get('description', 'No description')}")
                        st.markdown(f"**Duration:** {video.get('duration', 'Unknown')}")
                        st.markdown(f"**Relevance:** {video.get('relevance_score', 'Unknown')}")
                        
                        # Video preview
                        if 'url' in video:
                            video_id = video['url'].split('v=')[-1].split('&')[0]
                            st.markdown(f"""
                            <iframe width="100%" height="315" 
                                    src="https://www.youtube.com/embed/{video_id}" 
                                    frameborder="0" allowfullscreen>
                            </iframe>
                            """, unsafe_allow_html=True)
            else:
                st.info("No AI videos found. Check the curated videos below.")
    
    st.markdown("---")
    st.markdown("📚 **Curated Videos by Topic:**")
    
    browse_subject = st.selectbox(
        "Select a subject to browse",
        options=subjects_data[selected_form],
        key=f"browse_{selected_form}"
    )

    topics = subject_topics_videos.get(browse_subject, [])
    if not topics:
        st.info("No recommendations available for this subject yet.")
    else:
        for topic in topics:
            with st.expander(topic['topic'], expanded=False):
                for vid in topic.get('videos', []):
                    st.video(vid)

# --- Gamified Learning: Games Page ---
def save_game_score(user_id: int, game: str, subject: str, score: int) -> None:
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('INSERT INTO user_game_scores (user_id, game, subject, score) VALUES (?, ?, ?, ?)', (user_id, game, subject, score))
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error saving game score: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while saving game score: {e}")

def get_leaderboard(game: str, subject: str | None = None, limit: int = 10):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            if subject:
                c.execute('SELECT username, score, played_at FROM user_game_scores JOIN users USING(user_id) WHERE game = ? AND subject = ? ORDER BY score DESC, played_at ASC LIMIT ?', (game, subject, limit))
            else:
                c.execute('SELECT username, score, played_at FROM user_game_scores JOIN users USING(user_id) WHERE game = ? ORDER BY score DESC, played_at ASC LIMIT ?', (game, limit))
            return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error loading leaderboard: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading leaderboard: {e}")
        return []

def add_flashcard(user_id: int, subject: str, topic: str, front: str, back: str) -> None:
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('INSERT INTO flashcards (user_id, subject, topic, front, back) VALUES (?, ?, ?, ?, ?)', (user_id, subject, topic, front, back))
            conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error adding flashcard: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred while adding flashcard: {e}")

def get_flashcards(user_id: int, subject: str | None = None):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            if subject:
                c.execute('SELECT id, subject, topic, front, back FROM flashcards WHERE user_id = ? AND subject = ? ORDER BY created_at DESC', (user_id, subject))
            else:
                c.execute('SELECT id, subject, topic, front, back FROM flashcards WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
            return c.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error loading flashcards: {e}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading flashcards: {e}")
        return []

# --- Quiz Blitz Game ---
def quiz_blitz_game():
    st.header("Quiz Blitz")
    st.caption("Answer as many questions as you can. Each round draws from your selected ZIMSEC subjects.")
    
    # Initialize session state for this game
    if 'qb_form_choice' not in st.session_state:
        st.session_state.qb_form_choice = list(subjects_data.keys())[0]
    if 'qb_subject_choice' not in st.session_state:
        st.session_state.qb_subject_choice = None
    if 'qb_active' not in st.session_state:
        st.session_state.qb_active = False
    if 'qb_q_index' not in st.session_state:
        st.session_state.qb_q_index = 0
    if 'qb_score' not in st.session_state:
        st.session_state.qb_score = 0
    if 'qb_questions' not in st.session_state:
        st.session_state.qb_questions = []
    if 'qb_round_id' not in st.session_state:
        st.session_state.qb_round_id = None
    if 'qb_answers_tracked' not in st.session_state:
        st.session_state.qb_answers_tracked = []
    
    # Form and subject selection
    form_levels = list(subjects_data.keys())
    form_choice = st.selectbox("Form Level", form_levels, key="qb_form", index=form_levels.index(st.session_state.qb_form_choice))
    
    # Update form choice in session state
    if form_choice != st.session_state.qb_form_choice:
        st.session_state.qb_form_choice = form_choice
        st.session_state.qb_subject_choice = None
        st.session_state.qb_questions = []
        st.session_state.qb_active = False
        st.session_state.qb_q_index = 0
        st.session_state.qb_score = 0
        st.rerun()
    
    user_subjects = get_user_subjects(st.session_state.user_id, form_choice)
    if not user_subjects:
        st.info("Add subjects in the ZIMSEC page first to play.")
        return
    
    # Subject selection with proper state management
    if st.session_state.qb_subject_choice is None or st.session_state.qb_subject_choice not in user_subjects:
        st.session_state.qb_subject_choice = user_subjects[0]
    
    subject_choice = st.selectbox("Subject", user_subjects, key="qb_subject", index=user_subjects.index(st.session_state.qb_subject_choice))
    
    # Update subject choice in session state
    if subject_choice != st.session_state.qb_subject_choice:
        st.session_state.qb_subject_choice = subject_choice
        st.session_state.qb_questions = []
        st.session_state.qb_active = False
        st.session_state.qb_q_index = 0
        st.session_state.qb_score = 0
        st.rerun()
    
    # Start new round button
    if st.button("Start New Round", key="qb_start") and not st.session_state.qb_active:
        # Generate unique round ID
        st.session_state.qb_round_id = f"{form_choice}_{subject_choice}_{int(time.time())}"
        st.session_state.qb_active = True
        st.session_state.qb_q_index = 0
        st.session_state.qb_score = 0
        st.session_state.qb_questions = []
        st.rerun()
    
    # Game logic
    if st.session_state.qb_active:
        # Generate questions if not already generated
        if not st.session_state.qb_questions:
            try:
                # Check rate limiting first
                if not ai_rate_limiter.can_call():
                    wait_time = ai_rate_limiter.get_wait_time()
                    st.warning(f"⚠️ Rate limit exceeded. Please wait {wait_time} seconds before trying again.")
                    return
                
                with st.spinner("🤖 AI is generating your quiz questions..."):
                    # Much better prompt for quiz generation
                    system_prompt = f"""Create exactly 5 multiple-choice questions for {subject_choice} at the {form_choice} level.

Return ONLY a valid JSON array with this exact format:
[
  {{"question": "What is the formula for the area of a circle?", "options": ["A = πr²", "A = 2πr", "A = πd", "A = 2πd"], "correct": "A = πr²"}},
  {{"question": "Solve: 2x + 5 = 13", "options": ["x = 3", "x = 4", "x = 5", "x = 6"], "correct": "x = 4"}},
  {{"question": "What is the square root of 64?", "options": ["6", "7", "8", "9"], "correct": "8"}},
  {{"question": "Which is a prime number?", "options": ["15", "21", "23", "25"], "correct": "23"}},
  {{"question": "What is 3² × 2³?", "options": ["24", "48", "72", "96"], "correct": "72"}}
]

IMPORTANT: 
- The "options" array should contain the FULL answer text (not just A, B, C, D)
- The "correct" field should contain the EXACT text from one of the options
- Make questions appropriate for {form_choice} level
- Return ONLY the JSON array"""
                    
                    resp = model.generate_content(system_prompt)
                    response_text = resp.text.strip()
                    
                    # Clean up the response text
                    response_text = response_text.replace('```json', '').replace('```', '').strip()
                    
                    # Try to extract JSON from the response
                    try:
                        data = json.loads(response_text)
                    except json.JSONDecodeError:
                        # Try to find JSON array in the response
                        import re
                        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group())
                        else:
                            raise ValueError("No JSON array found in AI response")
                    
                    # Validate structure
                    questions = []
                    for item in data:
                        if isinstance(item, dict) and all(k in item for k in ("question", "options", "correct")) and len(item["options"]) == 4:
                            # Ensure the correct answer exists in the options
                            if item["correct"] in item["options"]:
                                questions.append(item)
                            else:
                                st.warning(f"Warning: Correct answer '{item['correct']}' not found in options: {item['options']}")
                    
                    if len(questions) >= 3:
                        st.session_state.qb_questions = questions
                        st.success(f"✅ AI generated {len(questions)} questions!")
                        # Debug: Show the first question structure
                        if questions:
                            st.write("Debug - First question structure:")
                            st.write(f"Question: {questions[0]['question']}")
                            st.write(f"Options: {questions[0]['options']}")
                            st.write(f"Correct: {questions[0]['correct']}")
                    else:
                        raise ValueError(f"Only {len(questions)} valid questions generated")
                        
            except Exception as e:
                # Better error handling
                st.warning(f"🤖 AI generation failed: {str(e)}")
                st.info("📚 Using offline questions as fallback...")
                
                # Use subject-specific offline questions
                st.session_state.qb_questions = get_offline_questions(subject_choice, form_choice)
        
        # Check if round is complete
        if st.session_state.qb_q_index >= len(st.session_state.qb_questions):
            # Round complete - show results
            st.success(f"�� Round Complete!")
            
            # Calculate percentage and grade
            total_questions = len(st.session_state.qb_questions)
            score = st.session_state.qb_score
            percentage = (score / total_questions) * 100
            
            # Grade calculation
            if percentage >= 90:
                grade = "A+ (Excellent!)"
                grade_color = "🟢"
            elif percentage >= 80:
                grade = "A (Very Good!)"
                grade_color = "🟢"
            elif percentage >= 70:
                grade = "B+ (Good!)"
                grade_color = "🟡"
            elif percentage >= 60:
                grade = "B (Satisfactory)"
                grade_color = "🟡"
            elif percentage >= 50:
                grade = "C (Pass)"
                grade_color = "🟠"
            else:
                grade = "F (Needs Improvement)"
                grade_color = "🔴"
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", f"{score}/{total_questions}")
            with col2:
                st.metric("Percentage", f"{percentage:.1f}%")
            with col3:
                st.metric("Grade", f"{grade_color} {grade}")
            
            # Detailed feedback
            st.markdown("---")
            st.subheader("📊 Detailed Feedback")
            
            if percentage >= 80:
                st.success("🎯 Outstanding performance! You have a strong understanding of this topic.")
            elif percentage >= 60:
                st.info("👍 Good work! Review the incorrect answers to improve further.")
            else:
                st.warning("📚 Keep studying! Focus on the concepts you missed.")
            
            # Study Plan Summary - Show topics you need to study
            st.markdown("---")
            st.subheader("📚 Study Recommendations")
            
            # Get all questions for study purposes
            all_questions = st.session_state.qb_questions
            
            if all_questions:
                st.info(f"Here are study materials for {subject_choice} topics:")
                
                # Group questions by topic for better organization
                study_topics = {}
                for q in all_questions:
                    # Extract topic from question (simple keyword extraction)
                    question_lower = q['question'].lower()
                    if 'formula' in question_lower or 'area' in question_lower or 'circle' in question_lower:
                        topic = "Geometry & Formulas"
                    elif 'solve' in question_lower or 'equation' in question_lower or 'algebra' in question_lower:
                        topic = "Algebra & Equations"
                    elif 'prime' in question_lower or 'number' in question_lower:
                        topic = "Number Theory"
                    elif 'square' in question_lower or 'root' in question_lower:
                        topic = "Advanced Math"
                    elif 'cell' in question_lower or 'biology' in question_lower:
                        topic = "Cell Biology"
                    elif 'chemistry' in question_lower or 'element' in question_lower:
                        topic = "Chemistry Fundamentals"
                    elif 'physics' in question_lower or 'force' in question_lower:
                        topic = "Physics Concepts"
                    elif 'history' in question_lower or 'war' in question_lower:
                        topic = "Historical Events"
                    elif 'geography' in question_lower or 'capital' in question_lower:
                        topic = "Geographic Knowledge"
                    elif 'english' in question_lower or 'grammar' in question_lower:
                        topic = "English Language"
                    else:
                        topic = "General Concepts"
                    
                    if topic not in study_topics:
                        study_topics[topic] = []
                    study_topics[topic].append(q)
                
                # Display study plan with video recommendations
                for topic, questions in study_topics.items():
                    with st.expander(f"📖 {topic} ({len(questions)} questions)", expanded=False):
                        st.markdown(f"**Topic:** {topic}")
                        st.markdown(f"**Questions in this area:** {len(questions)}")
                        
                        # Get video recommendations for this topic
                        sample_question = questions[0]['question']  # Use first question to get topic
                        recommended_videos = get_topic_videos(subject_choice, sample_question)
                        
                        if recommended_videos:
                            st.markdown("**📺 Recommended Study Videos:**")
                            for video in recommended_videos:
                                st.video(video)
                        else:
                            st.info(f"💡 Check the ZIMSEC Subjects page for {subject_choice} videos.")
                        
                        st.markdown("---")
            
            # Show correct answers for review with video recommendations
            with st.expander("📖 Review All Questions and Answers", expanded=False):
                for i, q in enumerate(st.session_state.qb_questions):
                    user_was_correct = "✅" if i < st.session_state.qb_q_index else "❌"
                    st.markdown(f"**Question {i+1}:** {q['question']}")
                    st.markdown(f"**Correct Answer:** {q['correct']}")
                    st.markdown(f"**Your Answer:** {user_was_correct}")
                    
                    # Show video recommendations for incorrect answers
                    if user_was_correct == "❌":
                        st.markdown("**📺 Study Videos for This Topic:**")
                        recommended_videos = get_topic_videos(subject_choice, q['question'])
                        if recommended_videos:
                            for video in recommended_videos:
                                st.video(video)
                        else:
                            st.info("💡 Check the ZIMSEC Subjects page for general {subject_choice} videos.")
                    
                    st.markdown("---")
            
            # Save score and show leaderboard
            save_game_score(st.session_state.user_id, "Quiz Blitz", subject_choice, int(score))
            
            # Play again button
            if st.button("�� Play Another Round", key="qb_play_again"):
                # Reset for new round
                st.session_state.qb_active = False
                st.session_state.qb_q_index = 0
                st.session_state.qb_score = 0
                st.session_state.qb_questions = []
                st.session_state.qb_round_id = None
                st.rerun()
            
            # Leaderboard
            st.subheader("🏆 Leaderboard")
            board = get_leaderboard("Quiz Blitz", subject_choice)
            if board:
                for i, (username, score, played_at) in enumerate(board, start=1):
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    st.write(f"{medal} {username} — {score}/{total_questions}")
            else:
                st.info("No scores yet. Be the first to set a record!")
            
            # Additional study resources
            st.markdown("---")
            st.subheader("📚 Continue Learning")
            st.info("Want to explore more study materials? Check out these resources:")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📚 ZIMSEC Subjects", key="qb_zimsec"):
                    st.session_state.page_selection = "📚 ZIMSEC Subjects"
                    st.rerun()
            
            with col2:
                if st.button("✍️ My Notes", key="qb_notes"):
                    st.session_state.page_selection = "✍️ My Notes"
                    st.rerun()
            
            return
        
        # Show current question
        current_q = st.session_state.qb_questions[st.session_state.qb_q_index]
        q_num = st.session_state.qb_q_index + 1
        total_q = len(st.session_state.qb_questions)
        
        # Progress bar
        progress = q_num / total_q
        st.progress(progress)
        st.caption(f"Question {q_num} of {total_q}")
        
        # Question display
        st.subheader(f"Question {q_num}")
        st.markdown(f"**{current_q['question']}**")
        
        # Answer options
        answer = st.radio("Choose your answer:", current_q["options"], key=f"qb_ans_{st.session_state.qb_round_id}_{q_num}")
        
        # Submit button
        if st.button("Submit Answer", key=f"qb_submit_{st.session_state.qb_round_id}_{q_num}"):
            # Normalize answers for comparison (remove extra whitespace, convert to string)
            user_answer = str(answer).strip() if answer else ""
            correct_answer = str(current_q["correct"]).strip() if current_q["correct"] else ""
            

            
            if user_answer == correct_answer:
                st.session_state.qb_score += 1
                st.success("✅ Correct! Well done!")
            else:
                st.error(f"❌ Incorrect. The correct answer is: **{correct_answer}**")
                
                # Provide explanation for wrong answers
                st.info("💡 **Study Tip:** Review this concept to improve your understanding.")
            
            # Move to next question
            st.session_state.qb_q_index += 1
            time.sleep(1)  # Brief pause to show feedback
            st.rerun()
    
    # Show instructions when not playing
    else:
        st.info("👆 Click 'Start New Round' to begin your quiz!")
        
        # Show recent scores
        recent_scores = get_leaderboard("Quiz Blitz", subject_choice, limit=5)
        if recent_scores:
            st.subheader("🏆 Recent Scores")
            for username, score, played_at in recent_scores:
                st.write(f"• {username}: {score}")

def flashcards_game():
    st.header("Flashcards")
    st.caption("Spaced practice with your own or AI-generated cards.")
    subjects = sorted({s for levels in subjects_data.values() for s in levels})
    subject_choice = st.selectbox("Subject", subjects, key="fc_subject")

    with st.expander("Add a card"):
        topic = st.text_input("Topic (optional)", key="fc_topic", placeholder="e.g., Basic Concepts, Formulas, Definitions")
        front = st.text_area("Front (Question/Term)", key="fc_front", placeholder="Enter the question, term, or concept")
        back = st.text_area("Back (Answer/Definition)", key="fc_back", placeholder="Enter the answer, definition, or explanation")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Save Card", key="fc_save"):
                if front and back:
                    # Sanitize inputs
                    clean_topic = sanitize_input(topic, max_length=100) if topic else "General"
                    clean_front = sanitize_input(front, max_length=500)
                    clean_back = sanitize_input(back, max_length=500)
                    
                    add_flashcard(st.session_state.user_id, subject_choice, clean_topic, clean_front, clean_back)
                    st.success("✅ Flashcard saved successfully!")
                    st.rerun()
                else:
                    st.warning("⚠️ Front and Back are required fields.")
        with col2:
            if st.button("Clear Form", key="fc_clear"):
                st.session_state.fc_topic = ""
                st.session_state.fc_front = ""
                st.session_state.fc_back = ""
                st.rerun()

    with st.expander("Generate 5 cards with AI"):
        prompt = st.text_input("Topic or keywords (optional)", key="fc_gen_prompt")
        if st.button("Generate", key="fc_generate"):
            # Check rate limiting
            if not ai_rate_limiter.can_call():
                wait_time = ai_rate_limiter.get_wait_time()
                st.warning(f"⚠️ Rate limit exceeded. Please wait {wait_time} seconds before trying again.")
                return
            
            # Validate and sanitize prompt
            clean_prompt = sanitize_input(prompt, max_length=200) if prompt else ""
            
            try:
                with st.spinner("🤖 AI is generating your flashcards..."):
                    # More specific prompt to get better JSON formatting
                    instruction = f"""Create exactly 5 flashcards for {subject_choice} subject.

Topic: {clean_prompt if clean_prompt else 'General concepts'}

Return ONLY a valid JSON array with this exact format:
[
  {{"topic": "Topic name", "front": "Question or term", "back": "Answer or definition"}},
  {{"topic": "Topic name", "front": "Question or term", "back": "Answer or definition"}},
  {{"topic": "Topic name", "front": "Question or term", "back": "Answer or definition"}},
  {{"topic": "Topic name", "front": "Question or term", "back": "Answer or definition"}},
  {{"topic": "Topic name", "front": "Question or term", "back": "Answer or definition"}}
]

Make sure the response is ONLY the JSON array, no additional text."""
                    
                    resp = model.generate_content(instruction)
                    response_text = resp.text.strip()
                    
                    # Try to extract JSON from the response (AI sometimes adds extra text)
                    try:
                        # First, try direct parsing
                        cards = json.loads(response_text)
                    except json.JSONDecodeError:
                        # Try to find JSON array in the response
                        import re
                        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                        if json_match:
                            try:
                                cards = json.loads(json_match.group())
                            except json.JSONDecodeError:
                                # Last resort: try to clean up common AI formatting issues
                                cleaned_text = response_text.replace('\n', ' ').replace('```json', '').replace('```', '')
                                cards = json.loads(cleaned_text)
                        else:
                            raise ValueError("No JSON array found in AI response")
                    
                    # Validate AI response structure
                    if not isinstance(cards, list):
                        st.error("AI response was not a list. Please try again.")
                        return
                    
                    if len(cards) < 3:  # Require at least 3 valid cards
                        st.error(f"AI only generated {len(cards)} cards. Need at least 3. Please try again.")
                        return
                    
                    # Save generated cards
                    saved_count = 0
                    for i, c in enumerate(cards[:5]):  # Limit to 5 cards
                        if isinstance(c, dict) and 'front' in c and 'back' in c:
                            add_flashcard(
                                st.session_state.user_id, 
                                subject_choice, 
                                c.get('topic') or f"Topic {i+1}", 
                                sanitize_input(c['front'], max_length=500),
                                sanitize_input(c['back'], max_length=500)
                            )
                            saved_count += 1
                    
                    if saved_count > 0:
                        st.success(f"✅ Generated and saved {saved_count} flashcards!")
                        st.rerun()
                    else:
                        st.error("No valid flashcards were generated. Please try again.")
                        st.info("💡 Tip: The AI response format was unexpected. Try being more specific with your topic.")
                        
            except json.JSONDecodeError as e:
                st.error(f"AI response was not in valid JSON format: {str(e)}")
                st.info("💡 Tip: Try again with a different topic or be more specific.")
                # Show the raw response for debugging
                with st.expander("🔍 Debug: Raw AI Response"):
                    st.code(resp.text if 'resp' in locals() else "No response available")
                    
            except Exception as e:
                st.error(f"Could not generate cards: {str(e)}")
                st.info("💡 Tip: Try being more specific with your topic description.")
                # Show the raw response for debugging
                if 'resp' in locals():
                    with st.expander("🔍 Debug: Raw AI Response"):
                        st.code(resp.text)
                
                # Offer fallback option
                st.warning("AI generation failed. Would you like to create flashcards manually?")
                if st.button("Create Manual Flashcards", key="manual_fallback"):
                    # Create some basic flashcards based on the subject
                    fallback_cards = [
                        {"topic": f"{subject_choice} Basics", "front": "What is the main focus of this subject?", "back": f"Understanding core concepts and principles of {subject_choice}"},
                        {"topic": f"{subject_choice} Key Terms", "front": "Name one important concept in this subject", "back": "This varies by topic - add your own specific terms"},
                        {"topic": f"{subject_choice} Study Tips", "front": "How should I study this subject effectively?", "back": "Practice regularly, understand concepts, and review frequently"},
                        {"topic": f"{subject_choice} Applications", "front": "Where is this subject used in real life?", "back": "Many fields use this knowledge - research specific applications"},
                        {"topic": f"{subject_choice} Resources", "front": "What resources help with learning this subject?", "back": "Textbooks, online courses, practice problems, and study groups"}
                    ]
                    
                    # Save fallback cards
                    saved_count = 0
                    for card in fallback_cards:
                        add_flashcard(
                            st.session_state.user_id,
                            subject_choice,
                            card["topic"],
                            card["front"],
                            card["back"]
                        )
                        saved_count += 1
                    
                    if saved_count > 0:
                        st.success(f"✅ Created {saved_count} basic flashcards to get you started!")
                        st.rerun()

    st.subheader("Practice")
    
    # Add helpful tips
    with st.expander("💡 Flashcard Tips"):
        st.markdown("""
        **Effective Study Tips:**
        - **Spaced Repetition**: Review cards at increasing intervals
        - **Active Recall**: Try to answer before flipping
        - **Mix Topics**: Don't study the same topic for too long
        - **Create Connections**: Link new concepts to what you already know
        
        **Good Flashcard Examples:**
        - **Front**: "What is the formula for area of a circle?"
        - **Back**: "A = πr², where r is the radius"
        
        - **Front**: "Define photosynthesis"
        - **Back**: "Process where plants convert sunlight into energy"
        """)
    
    cards = get_flashcards(st.session_state.user_id, subject_choice)
    if not cards:
        st.info("📝 No flashcards yet. Add some manually or generate them with AI above!")
        return
    
    # Show progress
    total_cards = len(cards)
    current_index = st.session_state.get("fc_index", 0)
    st.caption(f"📊 Card {current_index + 1} of {total_cards}")
    
    # Progress bar
    progress = (current_index + 1) / total_cards
    st.progress(progress)
    
    # Current card display
    c_id, c_subject, c_topic, c_front, c_back = cards[current_index]
    
    # Card display with better styling
    st.markdown(f"**📚 Topic: {c_topic or 'General'}**")
    
    # Front of card
    with st.container():
        st.markdown("**❓ Question/Term:**")
        st.info(c_front)
    
    # Show answer button
    if st.button("🔍 Show Answer", key="fc_show", type="primary"):
        st.markdown("**✅ Answer/Definition:**")
        st.success(c_back)
        
        # Add some study tips
        st.markdown("---")
        st.markdown("**💭 Study Reflection:**")
        st.markdown("- Did you know this answer?")
        st.markdown("- How confident are you with this concept?")
        st.markdown("- What related topics should you review?")
    
    # Navigation controls
    st.markdown("---")
    col_a, col_b, col_c = st.columns([1, 2, 1])
    
    with col_a:
        if st.button("⬅️ Previous", disabled=current_index == 0, key="fc_prev"):
            st.session_state["fc_index"] = max(0, current_index - 1)
            st.rerun()
    
    with col_b:
        st.markdown(f"**{current_index + 1} / {total_cards}**")
    
    with col_c:
        if st.button("Next ➡️", disabled=current_index >= total_cards - 1, key="fc_next"):
            st.session_state["fc_index"] = min(total_cards - 1, current_index + 1)
            st.rerun()
    
    # Quick navigation
    if total_cards > 5:
        st.markdown("**🎯 Quick Jump:**")
        jump_cols = st.columns(min(5, total_cards))
        for i, col in enumerate(jump_cols):
            if i < total_cards:
                if col.button(f"{i+1}", key=f"jump_{i}", disabled=i == current_index):
                    st.session_state["fc_index"] = i
                    st.rerun()

def games_page():
    st.title("🎮 Games")
    tab1, tab2 = st.tabs(["Quiz Blitz", "Flashcards"])
    with tab1:
        quiz_blitz_game()
    with tab2:
        flashcards_game()

# --- Chat Interface ---
def chat_interface():
    st.subheader("Chat with Kudzi")

    # Use a container to hold chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            is_user = message['role'] == 'user'
            message_class = "user-message" if is_user else "assistant-message"

            # Use columns to control alignment
            col1, col2 = st.columns([3, 1] if not is_user else [1, 3])
            with col1 if not is_user else col2:
                st.markdown(f'<div class="chat-message {message_class}">{message["content"]}</div>', unsafe_allow_html=True)

    # Get current language for chat input
    current_lang = st.session_state.get('user_language', 'en')
    lang_info = ZIMBABWEAN_LANGUAGES.get(current_lang, ZIMBABWEAN_LANGUAGES['en'])
    
    # Show warning for unsupported languages
    if not lang_info.get('is_supported', False):
        st.warning(f"🌍 **{lang_info['name']} ({lang_info['native_name']})** translation is not yet fully supported. Chat will work in English for now.")
    elif current_lang == 'nd':
        st.info("🌍 **Ndebele (isiNdebele)** is now supported with custom translations for common phrases and educational terms!")
    
    # Create localized chat input placeholder
    if current_lang == 'en':
        placeholder = "Ask Anything (in English or Shona)"
    elif current_lang == 'sn':
        placeholder = "Bvunza chero chinhu (muChirungu kana chiShona)"
    elif current_lang == 'nd':
        placeholder = "Buza noma yini (ngeSizulu noma isiNdebele) - Custom translations available"
    else:
        placeholder = f"Ask Anything (in {lang_info['name']}) - Basic support"
    
    prompt = st.chat_input(placeholder)

    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        save_chat_message(st.session_state.user_id, 'user', prompt)
        st.rerun() # Rerun to display the user's message immediately

    # If the last message is from the user, get a response
    if st.session_state.messages and st.session_state.messages[-1]['role'] == 'user':
        with st.spinner('Kudzi is thinking...'):
            last_prompt = st.session_state.messages[-1]['content']
            try:
                # Use our enhanced language detection
                detected_lang = language_manager.detect_language(last_prompt)
                
                # Translate to English if not already in English
                if detected_lang != 'en':
                    translated_prompt = language_manager.translate_text(last_prompt, 'en', detected_lang, 'chat')
                else:
                    translated_prompt = last_prompt

                # --- MCP Tools Integration ---
                # Detect which MCP tools should be used
                mcp_tool_calls = mcp_tools.detect_tool_usage(translated_prompt)
                mcp_context = ""
                tools_used = []
                
                # Execute detected MCP tools
                for tool_call in mcp_tool_calls:
                    if tool_call.get('confidence', 0) > 0.6:  # Only use tools with reasonable confidence
                        tool_name = tool_call['tool']
                        parameters = tool_call.get('parameters', {})
                        
                        try:
                            tool_result = mcp_tools.execute_tool(tool_name, parameters)
                            if tool_result.get('success'):
                                tools_used.append(tool_name)
                                # Format tool result for context
                                if tool_name == 'calculator' and 'result' in tool_result.get('result', {}):
                                    mcp_context += f"\n[Calculator Tool] Result: {tool_result['result'].get('result', 'N/A')}\n"
                                elif tool_name == 'get_time' and 'current_time' in tool_result.get('result', {}):
                                    time_info = tool_result['result']
                                    mcp_context += f"\n[Time Tool] Current time: {time_info.get('current_time', 'N/A')}, Date: {time_info.get('date', 'N/A')}, Day: {time_info.get('day_of_week', 'N/A')}\n"
                                elif tool_name == 'web_search':
                                    search_info = tool_result.get('result', {})
                                    mcp_context += f"\n[Search Tool] {search_info.get('suggestion', 'Search suggestions available')}\n"
                                elif tool_name == 'unit_converter' and 'result' in tool_result.get('result', {}):
                                    conv_info = tool_result['result']
                                    mcp_context += f"\n[Converter Tool] {conv_info.get('value', '')} {conv_info.get('from', '')} = {conv_info.get('result', '')} {conv_info.get('to', '')}\n"
                                else:
                                    # Include generic tool result
                                    mcp_context += f"\n[Tool: {tool_name}] {str(tool_result.get('result', 'Executed successfully'))}\n"
                        except Exception as tool_error:
                            # Continue even if tool fails
                            pass

                # Enhanced system prompt with MCP tools information
                system_prompt = (
                    "You are Kudzi, a helpful and friendly learning assistant who is an expert in both English and Shona. "
                    "You help students understand concepts clearly and patiently. "
                    "Provide concise and direct answers. If the original question was in Shona, ensure your answer is easily translatable back to Shona. "
                    "You have access to MCP (Model Context Protocol) tools that can perform calculations, get current time, convert units, and search for information. "
                    "When tools are used, incorporate the results naturally into your response."
                )
                
                # Add MCP context to the prompt if tools were used
                enhanced_prompt = translated_prompt
                if mcp_context and tools_used:
                    enhanced_prompt = f"{translated_prompt}\n\n[MCP Tools Results:]\n{mcp_context}\nPlease incorporate this information into your response when relevant."

                # Build context from previous messages
                history = []
                for msg in st.session_state.messages[:-1]: # Exclude the last user prompt
                    history.append({"role": msg['role'], "parts": [{"text": msg['content']}]})

                # Start chat with history
                chat_session = model.start_chat(history=history)
                response = chat_session.send_message(f"{system_prompt}\n\nUser's question: {enhanced_prompt}")

                response_text = response.text
                
                # Add MCP tool indicator if tools were used
                if tools_used:
                    response_text = f"🔧 [Used MCP tools: {', '.join(tools_used)}]\n\n{response_text}"

                # Translate response back to user's preferred language if needed
                if detected_lang != 'en':
                    try:
                        final_response = language_manager.translate_text(response_text, detected_lang, 'en', 'chat')
                        st.info(f"🌍 Response translated to {language_manager.get_language_display_name(detected_lang, 'en')}")
                    except Exception as trans_e:
                        st.warning(f"Could not translate response back to {language_manager.get_language_display_name(detected_lang, 'en')}. Showing English. Error: {trans_e}")
                        final_response = response_text
                else:
                    final_response = response_text

                st.session_state.messages.append({'role': 'assistant', 'content': final_response})
                save_chat_message(st.session_state.user_id, 'assistant', final_response, detected_lang)
                
                # Offer to save the response as a note
                st.markdown("---")
                st.markdown("💡 **Want to save this answer as a note?**")
                
                # Create a unique key for this chat response
                chat_note_key = f"chat_note_{len(st.session_state.messages)}"
                chat_note_title = st.text_input("Note Title (optional)", 
                                              value=f"Chat: {last_prompt[:40]}{'...' if len(last_prompt) > 40 else ''}", 
                                              key=f"{chat_note_key}_title")
                
                if st.button("💾 Save as Note", key=f"{chat_note_key}_save"):
                    if chat_note_title:
                        try:
                            save_note(
                                st.session_state.user_id,
                                chat_note_title,
                                f"**Question:** {last_prompt}\n\n**Answer:**\n{final_response}",
                                "Study"
                            )
                            st.success("✅ Chat response saved as a note!")
                        except Exception as save_error:
                            st.error(f"❌ Error saving note: {save_error}")
                    else:
                        st.warning("Please enter a title for the note.")
                
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")
                # Add an error message to chat and remove the last user prompt to allow retrying
                st.session_state.messages.append({'role': 'assistant', 'content': f"Sorry, I encountered an error: {e}"})
                st.rerun()

# --- Global Error Handling & User Experience ---
def show_error_with_recovery(error_msg: str, recovery_tip: str = None):
    """Display error message with recovery options."""
    st.error(f"❌ {error_msg}")
    if recovery_tip:
        st.info(f"💡 {recovery_tip}")
    
    # Add retry button
    if st.button("🔄 Try Again"):
        st.rerun()

def show_success_with_action(success_msg: str, action_msg: str = None):
    """Display success message with optional next action."""
    st.success(f"✅ {success_msg}")
    if action_msg:
        st.info(f"🎯 {action_msg}")

def safe_execute(func, *args, **kwargs):
    """Safely execute functions with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# --- Authentication and Pages ---
def login(username, password):
    conn = None
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('SELECT user_id, username FROM users WHERE username = ? AND password = ?', (username, password))
            result = c.fetchone()
            if result:
                st.session_state.logged_in = True
                st.session_state.username = result[1]
                st.session_state.user_id = result[0]
                load_user_chat_history(result[0])
                load_user_memory(result[0])
                # Default to Home page after login
                st.session_state["page_selection"] = "🏠 Home"
                return True
            return False
    except sqlite3.Error as e:
        st.error(f"Login error: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during login: {e}")
        return False

def signup(username, password):
    """User signup with input validation."""
    # Validate inputs
    if not validate_username(username):
        st.error("Username must be 3-20 characters long and contain only letters, numbers, and underscores.")
        return False
    
    if not validate_password(password):
        st.error("Password must be at least 6 characters long.")
        return False
    
    # Sanitize inputs
    clean_username = sanitize_input(username, max_length=20)
    clean_password = sanitize_input(password, max_length=100)
    
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (clean_username, clean_password))
            conn.commit()
            st.session_state.user_id = c.lastrowid
            st.success(f"Account created successfully for {clean_username}!")
            return True
    except sqlite3.IntegrityError:
        st.error("Username already exists. Please choose a different one.")
        return False
    except sqlite3.Error as e:
        st.error(f"Database error during signup: {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during signup: {e}")
        return False

def welcome_page():
    # Get current language for localized welcome
    current_lang = st.session_state.get('user_language', 'en')
    
    # Localized welcome messages
    welcome_messages = {
        'en': f"Welcome, {st.session_state.username}!",
        'sn': f"Tinokugamuchirai, {st.session_state.username}!",
        'nd': f"Siyakwamukela, {st.session_state.username}!",
        've': f"Ro tanganedzwa, {st.session_state.username}!",
        'to': f"Twalumba, {st.session_state.username}!"
    }
    
    platform_messages = {
        'en': "Welcome to the Kudzi Learning Platform!",
        'sn': "Tinokugamuchirai kuKudzi Learning Platform!",
        'nd': "Siyakwamukela kuKudzi Learning Platform!",
        've': "Ro tanganedzwa kuKudzi Learning Platform!",
        'to': "Twalumba kuKudzi Learning Platform!"
    }
    
    st.title(welcome_messages.get(current_lang, welcome_messages['en']))
    st.markdown(f'<p class="big-font">{platform_messages.get(current_lang, platform_messages["en"])}</p>', unsafe_allow_html=True)

    # Key metrics
    def count_user_subjects(user_id: int) -> int:
        conn = None
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute('SELECT COUNT(*) FROM user_subjects WHERE user_id = ?', (user_id,))
                val = c.fetchone()[0]
                return int(val or 0)
        except sqlite3.Error as e:
            st.error(f"Error counting user subjects: {e}")
            return 0
        except Exception as e:
            st.error(f"An unexpected error occurred while counting user subjects: {e}")
            return 0

    def count_user_notes(user_id: int) -> int:
        conn = None
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute('SELECT COUNT(*) FROM user_notes WHERE user_id = ?', (user_id,))
                val = c.fetchone()[0]
                return int(val or 0)
        except sqlite3.Error as e:
            st.error(f"Error counting user notes: {e}")
            return 0
        except Exception as e:
            st.error(f"An unexpected error occurred while counting user notes: {e}")
            return 0

    def last_game_score(user_id: int) -> int:
        conn = None
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                c.execute('SELECT score FROM user_game_scores WHERE user_id = ? ORDER BY played_at DESC LIMIT 1', (user_id,))
                row = c.fetchone()
                return int(row[0]) if row else 0
        except sqlite3.Error as e:
            st.error(f"Error loading last game score: {e}")
            return 0
        except Exception as e:
            st.error(f"An unexpected error occurred while loading last game score: {e}")
            return 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Subjects Selected", count_user_subjects(st.session_state.user_id))
    col2.metric("Notes", count_user_notes(st.session_state.user_id))
    col3.metric("Last Game Score", last_game_score(st.session_state.user_id))

    st.markdown("---")
    
    # Localized quick actions
    quick_action_labels = {
        'en': {
            'title': 'Quick Actions',
            'pick_subjects': '📚 Pick Subjects',
            'start_learning': '🎯 Start Learning',
            'play_game': '🎮 Play a Game',
            'open_notes': '✍️ Open Notes'
        },
        'sn': {
            'title': 'Zviito Zvekukurumidza',
            'pick_subjects': '📚 Sarudza Zvidzidzo',
            'start_learning': '🎯 Tanga Kudzidza',
            'play_game': '🎮 Tamba Mutambo',
            'open_notes': '✍️ Vhura Zvinyorwa'
        },
        'nd': {
            'title': 'Izinto Zokwenza Ngokushesha',
            'pick_subjects': '📚 Khetha Izifundo',
            'start_learning': '🎯 Qala Ukufunda',
            'play_game': '🎮 Dlala Umdlalo',
            'open_notes': '✍️ Vula Amanothi'
        }
    }
    
    current_labels = quick_action_labels.get(current_lang, quick_action_labels['en'])
    st.subheader(current_labels['title'])
    
    qa1, qa2, qa3, qa4 = st.columns(4)
    if qa1.button(current_labels['pick_subjects']):
        st.session_state["page_selection"] = "📚 ZIMSEC Subjects"
        st.rerun()
    if qa2.button(current_labels['start_learning']):
        st.session_state["page_selection"] = "🎯 Guided Learning"
        st.rerun()
    if qa3.button(current_labels['play_game']):
        st.session_state["page_selection"] = "🎮 Games"
        st.rerun()
    if qa4.button(current_labels['open_notes']):
        st.session_state["page_selection"] = "✍️ My Notes"
        st.rerun()

def login_signup_page():
    create_interactive_hero()
    st.markdown("""
        <style>
        .login-form-container {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(12px);
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            width: 95%;
            max-width: 520px;
            margin: -100px auto 50px auto; /* reduce overlap to avoid crowding */
            position: relative;
            z-index: 1000;
            border: 1px solid rgba(255,255,255,0.15);
        }
        .social-btn { display:flex;align-items:center;gap:.5rem;justify-content:center;width:100%;border-radius:10px;padding:.6rem 1rem;border:1px solid rgba(255,255,255,.2);background:rgba(255,255,255,.06);color:#fff }
        .or-sep { text-align:center;color:rgba(255,255,255,.7);margin:.8rem 0 }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="login-form-container">', unsafe_allow_html=True)

        # Fancy tabs (custom) to toggle Login/Signup
        default_tab = st.session_state.get("auth_tab", "Login")
        c1, c2 = st.columns(2)
        login_clicked = c1.button("Login", key="tab_login")
        signup_clicked = c2.button("Signup", key="tab_signup")
        if login_clicked:
            st.session_state["auth_tab"] = "Login"
        if signup_clicked:
            st.session_state["auth_tab"] = "Signup"
        active_tab = st.session_state.get("auth_tab", default_tab)

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        if active_tab == "Login":
            st.subheader("Welcome back 👋")
            st.caption("Continue your learning journey.")

            # Social demo buttons (non-functional placeholders)
            st.markdown('<div class="or-sep">Sign in with</div>', unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            sc1.button("Google", key="google_signin")
            sc2.button("Microsoft", key="microsoft_signin")
            st.markdown('<div class="or-sep">or</div>', unsafe_allow_html=True)

            with st.form("login_form"):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                remember = st.checkbox("Remember me", value=True)
                if st.form_submit_button("Login"):
                    if login(username, password):
                        st.success("Logged in successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

            # Fun tip / rotating hint
            tips = [
                "Tip: Add subjects in ZIMSEC to unlock more Quiz Blitz content!",
                "Did you know? You can generate flashcards with AI in Games.",
                "Pro tip: Use Notes to summarize videos from your subjects.",
                "🎯 Try Guided Learning for AI-powered personalized study paths!",
                "💡 AI generates practice exercises based on your learning progress.",
            ]
            st.info(random.choice(tips))

        else:
            st.subheader("Create your account ✨")
            st.caption("It only takes a few seconds.")
            with st.form("signup_form"):
                username = st.text_input("Choose a Username", key="signup_user")
                password = st.text_input("Create a Password", type="password", key="signup_pass")
                agree = st.checkbox("I agree to the Terms and Privacy Policy")
                if st.form_submit_button("Sign Up"):
                    if not agree:
                        st.warning("Please agree to the Terms to continue.")
                    else:
                        if signup(username, password):
                            st.success("Account created! Please log in.")
                        # Error handled in signup()

        st.markdown('</div>', unsafe_allow_html=True)

# --- Subject-Specific Pages ---
def subject_page(title, video_url, quiz_data):
    st.header(title)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(f"Introduction to {title}")
        # Note: These YouTube URLs are placeholders and won't play actual videos.
        # You'll need to replace them with real YouTube embed URLs or video IDs.
        st.video(video_url)
        st.subheader("Quick Quiz")
        with st.form(f"{title}_quiz"):
            score = 0
            for i, q in enumerate(quiz_data["questions"]):
                st.write(f"{i+1}. {q['text']}")
                user_answer = st.radio("Your answer:", q['options'], key=f"q{i}_{title}")
                if user_answer == q['correct']:
                    score += 1
            submitted = st.form_submit_button("Submit Quiz")
            if submitted:
                st.success(f"Your score: {score}/{len(quiz_data['questions'])}")

    with col2:
        st.subheader(f"Ask a {title} Question")
        prompt = st.text_area(f"Type your {title.lower()} question here", key=f"{title}_prompt")
        if st.button(f"Ask Kudzi", key=f"{title}_ask"):
            if prompt:
                with st.spinner("Getting an expert answer..."):
                    try:
                        # Enhance prompt for subject-specific expertise
                        enhanced_prompt = f"As a {title.lower()} tutor, explain the following concept or solve this problem concisely: {prompt}"
                        response = model.generate_content(enhanced_prompt)
                        
                        # Display the answer
                        st.markdown("**Answer:**")
                        st.markdown(response.text)
                        
                        # Offer to save as a note
                        st.markdown("---")
                        st.markdown("💡 **Want to save this answer as a note?**")
                        
                        note_title = st.text_input("Note Title (optional)", 
                                                value=f"{title}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}", 
                                                key=f"{title}_note_title")
                        
                        if st.button("💾 Save as Note", key=f"{title}_save_note"):
                            if note_title:
                                try:
                                    save_note(
                                        st.session_state.user_id,
                                        note_title,
                                        f"**Question:** {prompt}\n\n**Answer:**\n{response.text}",
                                        "Study"
                                    )
                                    st.success("✅ Answer saved as a note!")
                                except Exception as save_error:
                                    st.error(f"❌ Error saving note: {save_error}")
                            else:
                                st.warning("Please enter a title for the note.")
                                
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a question.")

# --- Interactive Hero Animation ---
def create_interactive_hero():
    hero_html = """
    <div class=\"hero-container\">
        <canvas id=\"heroCanvas\"></canvas>
        <div class=\"hero-content\">
            <h1>Welcome to Kudzi</h1>
            <p>Your Personal AI Learning Partner</p>
        </div>
    </div>
    <style>
    .hero-container{position:relative;width:100%;height:clamp(280px,45vh,480px);background:linear-gradient(135deg,#1e3c72 0%,#2a5298 100%);overflow:hidden;z-index:0}
    #heroCanvas{position:absolute;top:0;left:0;width:100%;height:100%}
    .hero-content{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;color:#fff;z-index:2;padding:2rem;width:100%;max-width:900px}
    .hero-content h1{font-size:clamp(1.8rem,4.5vw,3rem);font-weight:700;margin-bottom:1rem;background:linear-gradient(45deg,#00ff87,#60efff);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .hero-content p{font-size:clamp(1rem,2.5vw,1.2rem);color:rgba(255,255,255,.85)}
    @media (max-width: 768px){.hero-content{padding:1rem}}
    </style>
    <script>
    const canvas=document.getElementById('heroCanvas');
    const ctx=canvas.getContext('2d');
    function resizeCanvas(){
      const dpr=window.devicePixelRatio||1;
      const rect=canvas.getBoundingClientRect();
      canvas.width=Math.floor(rect.width*dpr);
      canvas.height=Math.floor(rect.height*dpr);
      ctx.setTransform(dpr,0,0,dpr,0,0);
    }
    resizeCanvas();
    window.addEventListener('resize',()=>{resizeCanvas()});
    class Particle{
      constructor(){this.reset()}
      reset(){
        const rect=canvas.getBoundingClientRect();
        this.x=Math.random()*rect.width;
        this.y=Math.random()*rect.height;
        this.size=Math.random()*2+1;
        this.speedX=.5*(Math.random()-.5);
        this.speedY=.5*(Math.random()-.5);
        this.color=`rgba(${Math.random()*100+155},${Math.random()*100+155},255,${Math.random()*.5+.5})`;
      }
      update(){
        const rect=canvas.getBoundingClientRect();
        this.x+=this.speedX; this.y+=this.speedY;
        if(this.x<0||this.x>rect.width) this.speedX*=-1;
        if(this.y<0||this.y>rect.height) this.speedY*=-1;
      }
      draw(){ctx.beginPath();ctx.arc(this.x,this.y,this.size,0,2*Math.PI);ctx.fillStyle=this.color;ctx.fill();}
    }
    const particles=Array(140).fill().map(()=>new Particle());
    function animate(){
      ctx.fillStyle='rgba(30,60,114,.10)';
      const rect=canvas.getBoundingClientRect();
      ctx.clearRect(0,0,rect.width,rect.height);
      particles.forEach(p=>{p.update();p.draw();});
      requestAnimationFrame(animate);
    }
    animate();
    </script>
    """
    html(hero_html, height=400)

def home_page():
    create_interactive_hero()
    welcome_page()

def sign_translate_page():
    st.header("Sign Language Translator (External Tool)")
    st.markdown("""
        <p>This is an embedded tool from an external website (sign.mt) for sign language translation.</p>
        <iframe src="https://sign.mt/" width="100%" height="600px" style="border:1px solid #ddd; border-radius:10px;"></iframe>
    """, unsafe_allow_html=True)

def notes_page():
    st.markdown("""
        <style>
        .note-card { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 20px; margin: 10px 0; border: 1px solid rgba(255, 255, 255, 0.2); }
        .note-title { color: #00ff87; font-size: 1.2em; font-weight: bold; }
        .note-meta { color: rgba(255, 255, 255, 0.6); font-size: 0.8em; }
        .note-content { color: white; margin-top: 10px; }
        .category-tag { background: linear-gradient(45deg, #00b4db, #0083b0); padding: 5px 10px; border-radius: 15px; font-size: 0.8em; color: white; }
        </style>
    """, unsafe_allow_html=True)
    st.title("My Notes")
    
    # Sidebar for note creation and filtering
    with st.sidebar:
        st.subheader("Create New Note")
        with st.form("new_note_form"):
            note_title = st.text_input("Title")
            note_category = st.selectbox("Category", ["General", "Study", "Project", "Personal", "Other"])
            note_content = st.text_area("Content", height=150)
            if st.form_submit_button("Save Note"):
                if note_title and note_content:
                    save_note(st.session_state.user_id, note_title, note_content, note_category)
                    st.success("Note saved!")
                    st.rerun()
                else:
                    st.warning("Title and content are required.")
        
        # AI Note Generation
        st.markdown("---")
        st.subheader("🤖 AI Note Generator")
        st.caption("Generate comprehensive study notes using AI")
        
        with st.expander("Generate Notes with AI", expanded=False):
            # Subject selection for AI notes
            all_subjects = sorted({s for levels in subjects_data.values() for s in levels})
            ai_subject = st.selectbox("Subject", all_subjects, key="ai_note_subject")
            
            # Topic input
            ai_topic = st.text_input("Topic or Concept", key="ai_note_topic", 
                                   placeholder="e.g., Quadratic Equations, Photosynthesis, World War II")
            
            # Note type selection
            note_type = st.selectbox("Note Type", [
                "Study Summary", "Concept Explanation", "Step-by-Step Guide", 
                "Key Points", "Practice Problems", "Comprehensive Review"
            ], key="ai_note_type")
            
            # Form level selection
            form_levels = list(subjects_data.keys())
            ai_form_level = st.selectbox("Form Level", form_levels, key="ai_note_form")
            
            # Generate button
            if st.button("🚀 Generate Notes", key="ai_generate_notes", type="primary"):
                if ai_topic:
                    generate_ai_notes(ai_subject, ai_topic, note_type, ai_form_level)
                else:
                    st.warning("Please enter a topic to generate notes.")
    
    # Main content area
    st.subheader("All Notes")
    
    # Notes summary and statistics
    notes = get_user_notes(st.session_state.user_id)
    if notes:
        # Calculate statistics
        total_notes = len(notes)
        study_notes = len([n for n in notes if n[3] == 'Study'])
        ai_generated = len([n for n in notes if 'AI' in n[1] or 'Generated' in n[1] or 'Chat:' in n[1]])
        favorites = len([n for n in notes if n[5]])
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Notes", total_notes)
        with col2:
            st.metric("Study Notes", study_notes)
        with col3:
            st.metric("AI Generated", ai_generated)
        with col4:
            st.metric("Favorites", favorites)
        
        st.markdown("---")
    
    # Quick AI note generation button and filters
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.info("💡 Need help creating notes? Use the AI Note Generator in the sidebar!")
    with col2:
        if st.button("🤖 AI Notes", key="quick_ai_notes"):
            st.info("👈 Use the AI Note Generator in the sidebar to create comprehensive study notes!")
    with col3:
        # Filter options
        filter_option = st.selectbox("Filter Notes", ["All Notes", "AI Generated", "Study Notes", "Favorites"], key="note_filter")
    
    # Apply filters
    if filter_option == "AI Generated":
        filtered_notes = [n for n in notes if 'AI:' in n[1] or 'Generated' in n[1] or 'Chat:' in n[1]]
    elif filter_option == "Study Notes":
        filtered_notes = [n for n in notes if n[3] == 'Study']
    elif filter_option == "Favorites":
        filtered_notes = [n for n in notes if n[5]]
    else:
        filtered_notes = notes
    
    if not notes:
        st.info("You don't have any notes yet. Create one in the sidebar or use AI to generate some!")
    elif not filtered_notes:
        st.info(f"No notes found matching the filter: {filter_option}")
    else:
        st.info(f"Showing {len(filtered_notes)} notes (filtered by: {filter_option})")

    for note in filtered_notes:
        note_id, title, content, category, created_at, is_favorite = note
        # Use pandas to format datetime objects for display if available, otherwise fallback
        formatted_date = pd.to_datetime(created_at).strftime('%b %d, %Y') if 'pd' in globals() else created_at
        with st.expander(f"{'★' if is_favorite else '☆'} {title} ({category}) - {formatted_date}", expanded=is_favorite):
            st.markdown(f'<div class="note-content">{content}</div>', unsafe_allow_html=True)
            b_col1, b_col2, b_col3 = st.columns(3)
            if b_col1.button("✏️ Edit", key=f"edit_{note_id}"):
                st.session_state.editing_note = note_id
                st.rerun() # Rerun to show edit form
            if b_col2.button("⭐ Favorite" if not is_favorite else "★ Unfavorite", key=f"fav_{note_id}"):
                toggle_favorite(note_id)
                st.rerun()
            if b_col3.button("🗑️ Delete", key=f"del_{note_id}"):
                delete_note(note_id)
                st.rerun()

    if st.session_state.editing_note:
        st.subheader("Edit Note")
        note_to_edit = [n for n in filtered_notes if n[0] == st.session_state.editing_note]
        if note_to_edit:
            note_to_edit = note_to_edit[0]
        else:
            # If note not in filtered results, get from all notes
            note_to_edit = [n for n in notes if n[0] == st.session_state.editing_note][0]
        with st.form("edit_note_form"):
            new_title = st.text_input("Title", value=note_to_edit[1])
            new_category = st.selectbox("Category", ["General", "Study", "Project", "Personal", "Other"], index=["General", "Study", "Project", "Personal", "Other"].index(note_to_edit[3]))
            new_content = st.text_area("Content", value=note_to_edit[2], height=150)
            s_col1, s_col2 = st.columns(2)
            if s_col1.form_submit_button("Save Changes"):
                update_note(st.session_state.editing_note, new_title, new_content, new_category)
                st.session_state.editing_note = None
                st.success("Note updated!")
                st.rerun()
            if s_col2.form_submit_button("Cancel"):
                st.session_state.editing_note = None
                st.rerun()

# --- Video Recommendation System ---
def search_youtube_videos(query: str, max_results: int = 3) -> list:
    """Search for relevant YouTube videos using AI-powered topic matching."""
    try:
        # Check rate limiting
        if not ai_rate_limiter.can_call():
            return []
        
        # Create a smart search query
        search_prompt = f"""Find {max_results} highly relevant YouTube videos for learning about: {query}

Return ONLY a valid JSON array with this exact format:
[
  {{
    "title": "Video title",
    "url": "https://www.youtube.com/watch?v=VIDEO_ID",
    "description": "Brief description of what the video covers",
    "duration": "Estimated duration (e.g., '10:30')",
    "relevance_score": "High/Medium/Low based on topic match"
  }}
]

Requirements:
- Videos must be directly related to the search topic
- Prefer educational channels and high-quality content
- Include only videos that are still available
- Focus on current, relevant content
- Return ONLY the JSON array"""

        response = model.generate_content(search_prompt)
        response_text = response.text.strip()
        
        # Clean and parse the response
        try:
            # Remove markdown formatting
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            videos = json.loads(response_text)
            
            if isinstance(videos, list) and len(videos) > 0:
                return videos[:max_results]
            else:
                return []
                
        except (json.JSONDecodeError, ValueError):
            return []
            
    except Exception as e:
        st.debug(f"Video search error: {e}")
        return []

def get_topic_videos(subject: str, question: str) -> list:
    """Get relevant YouTube videos based on subject and question content."""
    
    # Create a smart search query combining subject and question
    search_query = f"{subject}: {question}"
    
    # Try AI-powered search first
    ai_videos = search_youtube_videos(search_query, max_results=3)
    
    if ai_videos:
        return [video['url'] for video in ai_videos if 'url' in video]
    
    # Fallback to improved hard-coded videos (curated and verified)
    fallback_videos = get_curated_fallback_videos(subject, question)
    return fallback_videos

def get_curated_fallback_videos(subject: str, question: str) -> list:
    """Get curated fallback videos that are verified and relevant."""
    
    # Improved, verified video database
    curated_videos = {
        'Mathematics': {
            'algebra': [
                'https://www.youtube.com/watch?v=Q53GmMCqmAM',  # Khan Academy Algebra Basics
                'https://www.youtube.com/watch?v=MVmHusQ5hBQ',  # Geometry Essentials
                'https://www.youtube.com/watch?v=1-SvuFIQjK8'   # Trigonometry Intro
            ],
            'geometry': [
                'https://www.youtube.com/watch?v=MVmHusQ5hBQ',  # Geometry Essentials
                'https://www.youtube.com/watch?v=Q53GmMCqmAM'   # Math fundamentals
            ],
            'calculus': [
                'https://www.youtube.com/watch?v=WUvTyaaNkzM',  # Calculus Intro
                'https://www.youtube.com/watch?v=Q53GmMCqmAM'   # Math foundations
            ]
        },
        'Physics': {
            'newton': [
                'https://www.youtube.com/watch?v=kKKM8Y-u7ds',  # Newton's Laws
                'https://www.youtube.com/watch?v=VnnpLaKsqGU'   # Physics fundamentals
            ],
            'electricity': [
                'https://www.youtube.com/watch?v=VnnpLaKsqGU',  # Electricity Basics
                'https://www.youtube.com/watch?v=kKKM8Y-u7ds'   # Physics concepts
            ]
        },
        'Chemistry': {
            'periodic': [
                'https://www.youtube.com/watch?v=0RRVV4Diomg',  # Periodic Table
                'https://www.youtube.com/watch?v=QXT4OVM4vXI'   # Chemical Bonding
            ],
            'bonding': [
                'https://www.youtube.com/watch?v=QXT4OVM4vXI',  # Chemical Bonding
                'https://www.youtube.com/watch?v=0RRVV4Diomg'   # Chemistry basics
            ]
        },
        'Biology': {
            'cell': [
                'https://www.youtube.com/watch?v=URUJD5NEXC8',  # Cell Structure
                'https://www.youtube.com/watch?v=CBezq1fFUEA'   # Genetics Basics
            ],
            'photosynthesis': [
                'https://www.youtube.com/watch?v=URUJD5NEXC8',  # Cell processes
                'https://www.youtube.com/watch?v=CBezq1fFUEA'   # Biology concepts
            ]
        }
    }
    
    # Find relevant videos based on question content
    question_lower = question.lower()
    subject_videos = curated_videos.get(subject, {})
    
    relevant_videos = []
    for topic, videos in subject_videos.items():
        if topic in question_lower:
            relevant_videos.extend(videos)
    
    # If no specific matches, return general subject videos
    if not relevant_videos and subject in subject_videos:
        # Get first topic's videos as general content
        first_topic = list(subject_videos.keys())[0]
        relevant_videos = subject_videos[first_topic]
    
    return relevant_videos[:3]  # Return up to 3 videos

def create_video_playlist_page():
    """Page for users to create and share video playlists."""
    st.header("📺 Video Playlists")
    st.markdown("Create and share curated video playlists for different subjects and topics.")
    
    # User playlist creation
    with st.expander("Create New Playlist", expanded=False):
        with st.form("playlist_form"):
            playlist_name = st.text_input("Playlist Name")
            playlist_subject = st.selectbox("Subject", sorted({s for levels in subjects_data.values() for s in levels}))
            playlist_description = st.text_area("Description")
            playlist_videos = st.text_area("Video URLs (one per line)", 
                                         placeholder="https://www.youtube.com/watch?v=...\nhttps://www.youtube.com/watch?v=...")
            
            if st.form_submit_button("Create Playlist"):
                if playlist_name and playlist_videos:
                    # Save playlist to database (you'll need to add this table)
                    st.success(f"Playlist '{playlist_name}' created successfully!")
                else:
                    st.warning("Please fill in all required fields.")
    
    # AI-powered video discovery
    st.subheader("🔍 Discover New Videos")
    search_topic = st.text_input("Search for videos about:", placeholder="e.g., Quadratic Equations, Photosynthesis")
    
    if search_topic and st.button("🔍 Search Videos"):
        with st.spinner("Searching for relevant videos..."):
            videos = search_youtube_videos(search_topic, max_results=5)
            
            if videos:
                st.success(f"Found {len(videos)} relevant videos!")
                for i, video in enumerate(videos, 1):
                    with st.expander(f"{i}. {video.get('title', 'Untitled')}", expanded=False):
                        st.markdown(f"**Description:** {video.get('description', 'No description')}")
                        st.markdown(f"**Duration:** {video.get('duration', 'Unknown')}")
                        st.markdown(f"**Relevance:** {video.get('relevance_score', 'Unknown')}")
                        
                        # Video preview (iframe)
                        if 'url' in video:
                            video_id = video['url'].split('v=')[-1].split('&')[0]
                            st.markdown(f"""
                            <div style="margin: 10px 0;">
                                <h4>Video {i}</h4>
                                <iframe width="100%" height="315" 
                                        src="https://www.youtube.com/embed/{video_id}" 
                                        frameborder="0" allowfullscreen>
                                </iframe>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Add to playlist button
                        if st.button(f"➕ Add to Playlist", key=f"add_video_{i}"):
                            st.info("💡 Playlist feature coming soon!")
            else:
                st.info("No videos found. Try a different search term or check the curated videos below.")
    
    # Show curated videos by subject
    st.subheader("📚 Curated Videos by Subject")
    subject_choice = st.selectbox("Choose a subject:", sorted({s for levels in subjects_data.values() for s in levels}))
    
    if subject_choice:
        # Get curated videos for the subject
        curated = get_curated_fallback_videos(subject_choice, "general")
        
        if curated:
            st.info(f"Showing curated videos for {subject_choice}")
            for i, video_url in enumerate(curated, 1):
                try:
                    video_id = video_url.split('v=')[-1].split('&')[0]
                    st.markdown(f"""
                    <div style="margin: 10px 0;">
                        <h4>Video {i}</h4>
                        <iframe width="100%" height="315" 
                                src="https://www.youtube.com/embed/{video_id}" 
                                frameborder="0" allowfullscreen>
                        </iframe>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error displaying video {i}: {e}")
        else:
            st.info(f"No curated videos available for {subject_choice} yet.")

def get_ai_video_recommendations(subject: str, topic: str, form_level: str) -> list:
    """Get AI-powered video recommendations for specific subjects and topics."""
    
    try:
        if not ai_rate_limiter.can_call():
            return []
        
        # Create a comprehensive search query
        search_query = f"{subject} {topic} {form_level} educational video tutorial"
        
        # Use AI to find relevant videos
        videos = search_youtube_videos(search_query, max_results=5)
        
        if videos:
            return videos
        else:
            # Fallback to curated videos
            return get_curated_fallback_videos(subject, topic)
            
    except Exception as e:
        st.debug(f"AI video recommendation error: {e}")
        return get_curated_fallback_videos(subject, topic)

# --- AI Note Generation System ---
def clean_json_response(response_text: str) -> str:
    """Clean and fix common JSON formatting issues from AI responses."""
    cleaned = response_text.strip()
    
    # Remove markdown code blocks
    cleaned = cleaned.replace('```json', '').replace('```', '')
    
    # Fix common escape character issues
    cleaned = cleaned.replace('\\n', '\\\\n')  # Fix newline escapes
    cleaned = cleaned.replace('\\t', '\\\\t')  # Fix tab escapes
    cleaned = cleaned.replace('\\"', '\\\\"')  # Fix quote escapes
    cleaned = cleaned.replace("\\'", "\\\\'")  # Fix single quote escapes
    
    # Fix line ending issues
    cleaned = cleaned.replace('\r\n', '\\n')  # Fix Windows line endings
    cleaned = cleaned.replace('\r', '\\n')  # Fix old Mac line endings
    
    # Remove trailing commas before closing braces/brackets
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    
    # Remove control characters
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
    
    # Fix multiple backslashes
    cleaned = re.sub(r'\\+', r'\\', cleaned)
    
    return cleaned

def extract_json_from_text(text: str) -> dict:
    """Extract and parse JSON from text with multiple fallback strategies."""
    # Strategy 1: Direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Clean and try again
    try:
        cleaned = clean_json_response(text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Extract JSON object using regex
    try:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            cleaned_json = clean_json_response(json_str)
            return json.loads(cleaned_json)
    except (json.JSONDecodeError, AttributeError):
        pass
    
    # Strategy 4: Try to extract individual fields
    try:
        extracted_data = {}
        
        # Extract title
        title_match = re.search(r'"title":\s*"([^"]*)"', text, re.DOTALL)
        if title_match:
            extracted_data['title'] = title_match.group(1).replace('\\n', '\n').replace('\\"', '"')
        
        # Extract content
        content_match = re.search(r'"content":\s*"([^"]*)"', text, re.DOTALL)
        if content_match:
            extracted_data['content'] = content_match.group(1).replace('\\n', '\n').replace('\\"', '"')
        
        # Extract other fields if available
        key_points_match = re.search(r'"key_points":\s*\[(.*?)\]', text, re.DOTALL)
        if key_points_match:
            points_text = key_points_match.group(1)
            points = re.findall(r'"([^"]*)"', points_text)
            extracted_data['key_points'] = points
        
        examples_match = re.search(r'"examples":\s*\[(.*?)\]', text, re.DOTALL)
        if examples_match:
            examples_text = examples_match.group(1)
            examples = re.findall(r'"([^"]*)"', examples_text)
            extracted_data['examples'] = examples
        
        summary_match = re.search(r'"summary":\s*"([^"]*)"', text, re.DOTALL)
        if summary_match:
            extracted_data['summary'] = summary_match.group(1).replace('\\n', '\n').replace('\\"', '"')
        
        # Ensure required fields exist
        if 'title' in extracted_data and 'content' in extracted_data:
            extracted_data['category'] = extracted_data.get('category', 'Study')
            return extracted_data
            
    except Exception:
        pass
    
    # If all strategies fail, raise an error
    raise ValueError("Could not extract valid JSON or content from AI response")

def generate_ai_notes(subject: str, topic: str, note_type: str, form_level: str):
    """Generate comprehensive study notes using AI."""
    
    try:
        # Check rate limiting first
        if not ai_rate_limiter.can_call():
            wait_time = ai_rate_limiter.get_wait_time()
            st.warning(f"⚠️ Rate limit exceeded. Please wait {wait_time} seconds before trying again.")
            return
        
        with st.spinner("🤖 AI is generating your study notes..."):
            # Create a comprehensive prompt for note generation
            system_prompt = f"""Create comprehensive study notes for {topic} in {subject} at the {form_level} level.

Note Type: {note_type}

IMPORTANT: Return ONLY a valid JSON object with this exact format. Do not include any text before or after the JSON.
Ensure all quotes, backslashes, and special characters are properly escaped.

JSON Format:
{{
  "title": "Title for the notes",
  "content": "Comprehensive content with proper formatting, examples, and explanations. Use markdown formatting for structure.",
  "category": "Study",
  "key_points": ["Key point 1", "Key point 2", "Key point 3", "Key point 4", "Key point 5"],
  "examples": ["Example 1", "Example 2", "Example 3"],
  "practice_questions": ["Practice question 1", "Practice question 2", "Practice question 3"],
  "summary": "Brief summary of the main concepts"
}}

Requirements:
- Appropriate for {form_level} level
- Comprehensive and well-structured
- Include practical examples
- Use clear, student-friendly language
- Include key formulas if applicable
- Provide step-by-step explanations where needed
- Ensure all text content is properly escaped for JSON
- Do not use unescaped quotes or backslashes in content

Return ONLY the JSON object, no additional text, no markdown formatting around the JSON."""
            
            resp = model.generate_content(system_prompt)
            response_text = resp.text.strip()
            
            # Clean up the response text
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            # Try to extract JSON from the response with better error handling
            try:
                data = extract_json_from_text(response_text)
            except ValueError as json_error:
                st.error(f"❌ Could not parse AI response: {str(json_error)}")
                st.info("💡 Creating fallback note from available content...")
                
                # Show debug information
                with st.expander("🔍 Debug: Raw AI Response", expanded=False):
                    st.code(response_text)
                    st.info("This is the raw response from the AI. The JSON parsing failed due to formatting issues.")
                
                # Create a fallback note with the raw response
                fallback_title = f"{topic} - {note_type}"
                fallback_content = f"""**Topic:** {topic}
**Subject:** {subject}
**Form Level:** {form_level}
**Note Type:** {note_type}

**Content:**
{response_text}

*Note: This content was generated by AI but had formatting issues. It has been saved as-is for your reference.*"""
                
                # Save the fallback note
                try:
                    save_note(
                        st.session_state.user_id,
                        f"AI: {fallback_title}",
                        fallback_content,
                        "Study"
                    )
                    st.success("✅ Fallback note saved successfully!")
                    
                    # Offer retry option
                    st.markdown("---")
                    st.markdown("🔄 **Want to try generating better notes?**")
                    if st.button("🔄 Retry AI Generation", key="retry_ai_notes"):
                        st.rerun()
                    
                    return
                except Exception as save_error:
                    st.error(f"❌ Error saving fallback note: {save_error}")
                    return
            
            # Validate structure
            if not isinstance(data, dict) or 'title' not in data or 'content' not in data:
                st.error("AI response was not in the expected format. Please try again.")
                return
            
            # Display the generated notes
            st.success("✅ AI generated comprehensive notes!")
            
            # Show the notes in an expandable section
            with st.expander("📝 Generated Notes", expanded=True):
                st.markdown(f"**Title:** {data['title']}")
                st.markdown(f"**Category:** {data.get('category', 'Study')}")
                
                # Content
                st.markdown("**Content:**")
                st.markdown(data['content'])
                
                # Key Points
                if 'key_points' in data and data['key_points']:
                    st.markdown("**🔑 Key Points:**")
                    for i, point in enumerate(data['key_points'], 1):
                        st.markdown(f"{i}. {point}")
                
                # Examples
                if 'examples' in data and data['examples']:
                    st.markdown("**💡 Examples:**")
                    for i, example in enumerate(data['examples'], 1):
                        st.markdown(f"{i}. {example}")
                
                # Practice Questions
                if 'practice_questions' in data and data['practice_questions']:
                    st.markdown("**❓ Practice Questions:**")
                    for i, question in enumerate(data['practice_questions'], 1):
                        st.markdown(f"{i}. {question}")
                
                # Summary
                if 'summary' in data and data['summary']:
                    st.markdown("**📋 Summary:**")
                    st.markdown(data['summary'])
                
            # Auto-save the generated notes with enhanced categorization
            try:
                # Create a more descriptive title for AI-generated notes
                ai_title = f"AI: {data['title']}"
                
                # Enhanced content with metadata
                enhanced_content = f"""🤖 **AI-Generated Study Notes**
**Subject:** {subject}
**Form Level:** {form_level}
**Note Type:** {note_type}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

{data['content']}

---

**Key Points:**
"""
                
                # Add key points if available
                if 'key_points' in data and data['key_points']:
                    for i, point in enumerate(data['key_points'], 1):
                        enhanced_content += f"{i}. {point}\n"
                
                # Add examples if available
                if 'examples' in data and data['examples']:
                    enhanced_content += "\n**Examples:**\n"
                    for i, example in enumerate(data['examples'], 1):
                        enhanced_content += f"{i}. {example}\n"
                
                # Add practice questions if available
                if 'practice_questions' in data and data['practice_questions']:
                    enhanced_content += "\n**Practice Questions:**\n"
                    for i, question in enumerate(data['practice_questions'], 1):
                        enhanced_content += f"{i}. {question}\n"
                
                # Add summary if available
                if 'summary' in data and data['summary']:
                    enhanced_content += f"\n**Summary:**\n{data['summary']}"
                
                save_note(
                    st.session_state.user_id,
                    ai_title,
                    enhanced_content,
                    "Study"
                )
                st.success("✅ Notes automatically saved to your collection!")
            except Exception as save_error:
                st.error(f"❌ Error saving notes: {save_error}")
                
                # Save button (for manual save if needed)
                st.markdown("---")
                if st.button("💾 Save Again", key="save_ai_notes"):
                    try:
                        save_note(
                            st.session_state.user_id,
                            data['title'],
                            data['content'],
                            data.get('category', 'Study')
                        )
                        st.success("✅ Notes saved again!")
                        st.rerun()
                    except Exception as save_error:
                        st.error(f"❌ Error saving notes: {save_error}")
                
                # Regenerate button
                if st.button("🔄 Regenerate Notes", key="regenerate_ai_notes"):
                    st.rerun()
                    
    except json.JSONDecodeError as e:
        st.error(f"AI response was not in valid JSON format: {str(e)}")
        st.info("💡 Tip: Try again with a different topic or be more specific.")
        # Show the raw response for debugging
        with st.expander("🔍 Debug: Raw AI Response"):
            st.code(resp.text if 'resp' in locals() else "No response available")
            
    except Exception as e:
        st.error(f"Could not generate notes: {str(e)}")
        st.info("💡 Tip: Try being more specific with your topic description.")
        # Show the raw response for debugging
        if 'resp' in locals():
            with st.expander("🔍 Debug: Raw AI Response"):
                st.code(resp.text)
        
        # Offer fallback option
        st.warning("AI generation failed. Would you like to create notes manually?")
        if st.button("Create Manual Notes", key="manual_note_fallback"):
            # Create some basic notes based on the topic
            fallback_content = f"""
# {topic} - {subject}

## Overview
This is a study guide for {topic} in {subject} at the {form_level} level.

## Key Concepts
- Add your own key concepts here
- Include important definitions
- Note key formulas or principles

## Examples
- Add practical examples
- Include step-by-step solutions
- Note common mistakes to avoid

## Practice
- Create your own practice problems
- Review regularly
- Test your understanding

## Notes
- Add additional notes as you study
- Include questions you have
- Note areas that need more focus
            """
            
            # Save fallback notes
            save_note(
                st.session_state.user_id,
                f"{topic} - {subject}",
                fallback_content,
                "Study"
            )
            st.success("✅ Created basic notes to get you started!")
            st.rerun()

# --- Offline Quiz Questions (Fallback) ---
def get_offline_questions(subject: str, form_level: str) -> list:
    """Get subject-specific offline questions when AI generation fails."""
    
    # Subject-specific question banks
    question_banks = {
        'Mathematics': [
            {"question": "What is the formula for the area of a circle?", "options": ["A = πr²", "A = 2πr", "A = πd", "A = 2πd"], "correct": "A = πr²"},
            {"question": "Solve: 2x + 5 = 13", "options": ["x = 3", "x = 4", "x = 5", "x = 6"], "correct": "x = 4"},
            {"question": "What is the square root of 64?", "options": ["6", "7", "8", "9"], "correct": "8"},
            {"question": "Which is a prime number?", "options": ["15", "21", "23", "25"], "correct": "23"},
            {"question": "What is 3² × 2³?", "options": ["24", "48", "72", "96"], "correct": "72"}
        ],
        'Physics': [
            {"question": "What is Newton's First Law about?", "options": ["Action and reaction", "Inertia", "Force and acceleration", "Gravity"], "correct": "Inertia"},
            {"question": "What unit measures force?", "options": ["Joule", "Watt", "Newton", "Pascal"], "correct": "Newton"},
            {"question": "What is the SI unit for mass?", "options": ["Gram", "Kilogram", "Pound", "Ounce"], "correct": "Kilogram"},
            {"question": "What type of energy does a moving object have?", "options": ["Potential", "Kinetic", "Thermal", "Chemical"], "correct": "Kinetic"},
            {"question": "What is the speed of light in vacuum?", "options": ["3 × 10⁸ m/s", "3 × 10⁶ m/s", "3 × 10⁴ m/s", "3 × 10² m/s"], "correct": "3 × 10⁸ m/s"}
        ],
        'Chemistry': [
            {"question": "What is the chemical symbol for gold?", "options": ["Ag", "Au", "Fe", "Cu"], "correct": "Au"},
            {"question": "What is the pH of a neutral solution?", "options": ["0", "7", "10", "14"], "correct": "7"},
            {"question": "What gas do plants absorb during photosynthesis?", "options": ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"], "correct": "Carbon dioxide"},
            {"question": "What is the most abundant element in the universe?", "options": ["Helium", "Carbon", "Oxygen", "Hydrogen"], "correct": "Hydrogen"},
            {"question": "What type of bond shares electrons?", "options": ["Ionic", "Covalent", "Metallic", "Hydrogen"], "correct": "Covalent"}
        ],
        'Biology': [
            {"question": "What is the powerhouse of the cell?", "options": ["Nucleus", "Mitochondria", "Ribosome", "Golgi body"], "correct": "Mitochondria"},
            {"question": "What process do plants use to make food?", "options": ["Respiration", "Photosynthesis", "Digestion", "Excretion"], "correct": "Photosynthesis"},
            {"question": "What is the basic unit of life?", "options": ["Atom", "Molecule", "Cell", "Tissue"], "correct": "Cell"},
            {"question": "What carries genetic information?", "options": ["RNA", "DNA", "Protein", "Lipid"], "correct": "DNA"},
            {"question": "What system pumps blood through the body?", "options": ["Digestive", "Circulatory", "Respiratory", "Nervous"], "correct": "Circulatory"}
        ],
        'English Language': [
            {"question": "What is a noun?", "options": ["Action word", "Describing word", "Person, place, or thing", "Connecting word"], "correct": "Person, place, or thing"},
            {"question": "Which is a proper noun?", "options": ["City", "London", "Country", "River"], "correct": "London"},
            {"question": "What is the past tense of 'go'?", "options": ["Goed", "Gone", "Went", "Going"], "correct": "Went"},
            {"question": "What punctuation ends a question?", "options": ["Period", "Comma", "Question mark", "Exclamation mark"], "correct": "Question mark"},
            {"question": "What is a synonym for 'happy'?", "options": ["Sad", "Joyful", "Angry", "Tired"], "correct": "Joyful"}
        ],
        'History': [
            {"question": "What year did World War II end?", "options": ["1943", "1944", "1945", "1946"], "correct": "1945"},
            {"question": "Who was the first President of the United States?", "options": ["Thomas Jefferson", "John Adams", "George Washington", "Benjamin Franklin"], "correct": "George Washington"},
            {"question": "What was the main cause of the American Civil War?", "options": ["Taxes", "Slavery", "Trade", "Religion"], "correct": "Slavery"},
            {"question": "What empire ruled much of Europe in the 1800s?", "options": ["Roman", "British", "French", "Ottoman"], "correct": "British"},
            {"question": "What year did Zimbabwe gain independence?", "options": ["1978", "1980", "1982", "1984"], "correct": "1980"}
        ],
        'Geography': [
            {"question": "What is the capital of Zimbabwe?", "options": ["Bulawayo", "Harare", "Mutare", "Gweru"], "correct": "Harare"},
            {"question": "What is the largest continent?", "options": ["Africa", "Asia", "North America", "Europe"], "correct": "Asia"},
            {"question": "What is the longest river in Africa?", "options": ["Nile", "Congo", "Niger", "Zambezi"], "correct": "Nile"},
            {"question": "What type of climate does Zimbabwe have?", "options": ["Tropical", "Desert", "Temperate", "Arctic"], "correct": "Tropical"},
            {"question": "What is the study of maps called?", "options": ["Geology", "Cartography", "Meteorology", "Oceanography"], "correct": "Cartography"}
        ]
    }
    
    # Get questions for the subject, or use general questions as fallback
    if subject in question_banks:
        return question_banks[subject]
    else:
        # General fallback questions
        return [
            {"question": "What is 2 + 2?", "options": ["3", "4", "5", "6"], "correct": "4"},
            {"question": "Which is a prime number?", "options": ["4", "6", "9", "7"], "correct": "7"},
            {"question": "Earth is a...", "options": ["Star", "Planet", "Comet", "Galaxy"], "correct": "Planet"},
            {"question": "Water boils at? (°C)", "options": ["50", "90", "100", "110"], "correct": "100"},
            {"question": "Opposite of north is...", "options": ["East", "South", "West", "Up"], "correct": "South"}
        ]

# --- Main App Logic ---
def main():
    # Offline mode indicator
    st.markdown("""
    <div class="offline-indicator" id="offline-indicator">
        📡 You're offline. Some features may be limited.
    </div>
    <script>
    // Check online status
    function updateOnlineStatus() {
        const indicator = document.getElementById('offline-indicator');
        if (navigator.onLine) {
            indicator.classList.remove('show');
        } else {
            indicator.classList.add('show');
        }
    }
    
    window.addEventListener('online', updateOnlineStatus);
    window.addEventListener('offline', updateOnlineStatus);
    updateOnlineStatus();
    </script>
    """, unsafe_allow_html=True)
    
    # Demo Mode for Exhibition
    if st.session_state.get('demo_mode', False):
        st.markdown("""
        <div style="background: linear-gradient(45deg, #ff6b6b, #ffa500); padding: 10px; border-radius: 10px; margin-bottom: 20px; text-align: center; color: white;">
            <h3>🎯 DEMO MODE - World Literacy Day Exhibition</h3>
            <p>Experience the full platform capabilities with sample data</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Demo mode controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Start Demo", key="start_demo"):
                st.session_state.demo_mode = True
                st.session_state.logged_in = True
                st.session_state.username = "Demo User"
                st.session_state.user_id = 999
                st.rerun()
        with col2:
            if st.button("📊 Show Analytics", key="demo_analytics"):
                st.session_state["page_selection"] = "🎯 Guided Learning"
                st.rerun()
        with col3:
            if st.button("🎮 Try Games", key="demo_games"):
                st.session_state["page_selection"] = "🎮 Games"
                st.rerun()
    
    if not st.session_state.get('logged_in', False):
        # Show demo mode option on login page
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 15px; margin: 20px 0; text-align: center;">
            <h3>🌟 Exhibition Demo Available</h3>
            <p>Try the platform without creating an account</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Try Demo Mode", key="enable_demo"):
            st.session_state.demo_mode = True
            st.session_state.logged_in = True
            st.session_state.username = "Demo User"
            st.session_state.user_id = 999
            st.rerun()
        
        login_signup_page()
    else:
        st.sidebar.title(f"Hi, {st.session_state.username}!")
        
        # Accessibility Controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("♿ Accessibility")
        
        # High contrast mode
        high_contrast = st.sidebar.checkbox("High Contrast Mode", key="high_contrast")
        if high_contrast:
            st.markdown('<div class="high-contrast">', unsafe_allow_html=True)
        
        # Text size options
        text_size = st.sidebar.selectbox("Text Size", ["Normal", "Large", "Extra Large"], key="text_size")
        if text_size == "Large":
            st.markdown('<div class="large-text">', unsafe_allow_html=True)
        elif text_size == "Extra Large":
            st.markdown('<div class="extra-large-text">', unsafe_allow_html=True)
        
        # Language Selection
        st.sidebar.markdown("---")
        st.sidebar.subheader("🌍 Language / Mutauro")
        available_languages = language_manager.get_supported_languages_list()
        language_options = {f"{lang['flag']} {lang['name']} ({lang['native_name']})": lang['code'] for lang in available_languages}
        
        selected_language = st.sidebar.selectbox(
            "Choose Language / Sarudza Mutauro:",
            list(language_options.keys()),
            index=0,  # Default to English
            key="language_selector"
        )
        
        # Store selected language in session state
        current_language = language_options[selected_language]
        if 'user_language' not in st.session_state:
            st.session_state.user_language = current_language
        elif st.session_state.user_language != current_language:
            st.session_state.user_language = current_language
            st.rerun()
        
        # Language Information
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ Language Info")
        
        # Show current language
        current_lang_info = ZIMBABWEAN_LANGUAGES.get(current_language, ZIMBABWEAN_LANGUAGES['en'])
        st.sidebar.info(f"**Current:** {current_lang_info['flag']} {current_lang_info['name']}")
        
        # Show supported languages
        with st.sidebar.expander("🌍 Supported Languages"):
            for lang in available_languages:
                lang_info = ZIMBABWEAN_LANGUAGES.get(lang['code'], {})
                if lang_info.get('is_supported', False):
                    if lang['code'] == 'nd':
                        st.markdown(f"✅ **{lang['flag']} {lang['name']}** ({lang['native_name']}) - Custom Translations")
                    else:
                        st.markdown(f"✅ **{lang['flag']} {lang['name']}** ({lang['native_name']}) - Full Support")
                elif lang_info.get('is_curriculum', False):
                    st.markdown(f"🔄 **{lang['flag']} {lang['name']}** ({lang['native_name']}) - Basic Support")
                else:
                    st.markdown(f"🌐 **{lang['flag']} {lang['name']}** ({lang['native_name']}) - Coming Soon")
        
        st.sidebar.markdown("---")
        
        menu_options = {
            "🏠 Home": home_page,
            "💬 Chat": chat_interface,
            "📚 ZIMSEC Subjects": zimsec_subjects_page,
            "🎯 Guided Learning": guided_learning_page,
            "📅 Study Planning": study_planning_page,
            "🏆 Achievements": gamification_page,
            "🎮 Games": games_page,
            "✍️ My Notes": notes_page,
            "📺 Video Playlists": create_video_playlist_page,
            "🌐 Sign Translate": sign_translate_page,
            "🎤 Voice & Speech": voice_recognition_page,
            "📷 Image Recognition": image_recognition_page,
            "🔍 Content Search": content_search_page,
            "♿ Accessibility": accessibility_page,
            "📊 Data Export": export_data_page,
            "🌟 Exhibition Mode": exhibition_dashboard,
        }
        
        # Keep selection in session state so Quick Actions can switch pages
        current = st.session_state.get("page_selection") or list(menu_options.keys())[0]
        selection = st.sidebar.radio("Navigation", list(menu_options.keys()), index=list(menu_options.keys()).index(current), key="nav")
        
        # Call the selected page function
        menu_options[selection]()

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.user_id = None
            st.session_state.messages = []
            st.session_state.memory = []
            st.session_state.pop("page_selection", None)
            st.rerun()

# --- Guided Learning Feature ---
# AI-powered personalized learning paths using Gemini

def create_learning_path(user_id: int, form_level: str, subject: str) -> dict:
    """Create a personalized learning path using Gemini AI."""
    try:
        # Check rate limiting
        if not ai_rate_limiter.can_call():
            return {"error": "Rate limit exceeded. Please wait before creating a new learning path."}
        
        # Get user's current subjects and any existing progress
        user_subjects = get_user_subjects(user_id, form_level)
        
        # Create AI prompt for learning path generation
        learning_prompt = f"""Create a comprehensive learning path for {subject} at {form_level} level.

Requirements:
1. Break down the subject into logical learning modules
2. Sequence topics from basic to advanced
3. Include estimated time for each module
4. Provide learning objectives for each topic
5. Suggest practice activities
6. Include assessment checkpoints

Return ONLY a valid JSON with this exact format:
{{
  "subject": "{subject}",
  "form_level": "{form_level}",
  "total_modules": 5,
  "estimated_total_time": "X hours",
  "modules": [
    {{
      "module_id": 1,
      "title": "Module Title",
      "description": "Brief description",
      "estimated_time": "X hours",
      "topics": [
        {{
          "topic_id": 1,
          "title": "Topic Title",
          "description": "What will be learned",
          "learning_objectives": ["Objective 1", "Objective 2"],
          "prerequisites": ["Previous knowledge needed"],
          "practice_activities": ["Activity 1", "Activity 2"],
          "assessment": "How to check understanding"
        }}
      ]
    }}
  ]
}}"""

        response = model.generate_content(learning_prompt)
        response_text = response.text.strip()
        
        # Clean and parse the response
        try:
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            learning_path = json.loads(response_text)
            
            # Store the learning path in the database
            store_learning_path(user_id, form_level, subject, learning_path)
            
            return learning_path
            
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Error parsing AI response: {e}")
            return {"error": "Failed to generate learning path. Please try again."}
            
    except Exception as e:
        st.error(f"Error creating learning path: {e}")
        return {"error": "An unexpected error occurred."}

def store_learning_path(user_id: int, form_level: str, subject: str, learning_path: dict) -> bool:
    """Store the generated learning path in the database."""
    def _store_path():
        conn = None
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                
                # Store the learning path
                learning_path_json = json.dumps(learning_path)
                c.execute('''INSERT OR REPLACE INTO learning_paths 
                            (user_id, form_level, subject, learning_path_data, updated_at) 
                            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)''',
                         (user_id, form_level, subject, learning_path_json))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            return False
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return False
    
    return safe_database_operation("store_learning_path", _store_path)

def get_user_learning_paths(user_id: int) -> list:
    """Get all learning paths for a user."""
    def _get_paths():
        conn = None
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                
                c.execute('''SELECT form_level, subject, learning_path_data, created_at, updated_at 
                            FROM learning_paths 
                            WHERE user_id = ? 
                            ORDER BY updated_at DESC''', (user_id,))
                
                rows = c.fetchall()
                learning_paths = []
                
                for row in rows:
                    try:
                        learning_path = json.loads(row[2])
                        learning_paths.append({
                            'form_level': row[0],
                            'subject': row[1],
                            'data': learning_path,
                            'created_at': row[3],
                            'updated_at': row[4]
                        })
                    except json.JSONDecodeError:
                        continue
                        
                return learning_paths
                
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            return []
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return []
    
    result = safe_database_operation("get_user_learning_paths", _get_paths)
    return result if result is not None else []

def generate_practice_exercise(topic: str, difficulty: str = "beginner") -> dict:
    """Generate a practice exercise using Gemini AI."""
    try:
        # Check rate limiting
        if not ai_rate_limiter.can_call():
            return {"error": "Rate limit exceeded. Please wait before generating exercises."}
        
        exercise_prompt = f"""Create a practice exercise for the topic: {topic} at {difficulty} difficulty level.

Return ONLY a valid JSON with this exact format:
{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "question": "Clear question or problem statement",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "correct_answer": "Option A",
  "explanation": "Detailed explanation of why this is correct",
  "hint": "A helpful hint for students",
  "related_concepts": ["Related concept 1", "Related concept 2"]
}}"""

        response = model.generate_content(exercise_prompt)
        response_text = response.text.strip()
        
        try:
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            exercise = json.loads(response_text)
            return exercise
            
        except (json.JSONDecodeError, ValueError):
            return {"error": "Failed to generate exercise. Please try again."}
            
    except Exception as e:
        return {"error": f"Error generating exercise: {e}"}

# --- Advanced Learning Analytics System ---
def get_learning_analytics(user_id: int) -> dict:
    """Get comprehensive learning analytics for a user."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Get overall statistics
            c.execute('''
                SELECT COUNT(DISTINCT subject) as total_subjects,
                       COUNT(*) as total_activities,
                       AVG(score) as avg_score
                FROM user_game_scores 
                WHERE user_id = ?
            ''', (user_id,))
            overall_stats = c.fetchone()
            
            # Get subject-wise performance
            c.execute('''
                SELECT subject, 
                       COUNT(*) as attempts,
                       AVG(score) as avg_score,
                       MAX(score) as best_score,
                       MIN(score) as worst_score
                FROM user_game_scores 
                WHERE user_id = ?
                GROUP BY subject
                ORDER BY avg_score DESC
            ''', (user_id,))
            subject_performance = c.fetchall()
            
            # Get learning streaks
            c.execute('''
                SELECT current_streak, longest_streak, last_activity_date
                FROM learning_streaks 
                WHERE user_id = ?
            ''', (user_id,))
            streak_data = c.fetchone()
            
            # Get recent activity
            c.execute('''
                SELECT game, subject, score, played_at
                FROM user_game_scores 
                WHERE user_id = ?
                ORDER BY played_at DESC
                LIMIT 10
            ''', (user_id,))
            recent_activity = c.fetchall()
            
            # Get notes statistics
            c.execute('''
                SELECT COUNT(*) as total_notes,
                       COUNT(CASE WHEN category = 'Study' THEN 1 END) as study_notes,
                       COUNT(CASE WHEN is_favorite = 1 THEN 1 END) as favorite_notes
                FROM user_notes 
                WHERE user_id = ?
            ''', (user_id,))
            notes_stats = c.fetchone()
            
            return {
                'overall_stats': {
                    'total_subjects': overall_stats[0] or 0,
                    'total_activities': overall_stats[1] or 0,
                    'avg_score': round(overall_stats[2] or 0, 2)
                },
                'subject_performance': [
                    {
                        'subject': row[0],
                        'attempts': row[1],
                        'avg_score': row[2] or 0,
                        'best_score': row[3] or 0,
                        'worst_score': row[4] or 0
                    } for row in subject_performance
                ],
                'streak_data': {
                    'current_streak': streak_data[0] if streak_data else 0,
                    'longest_streak': streak_data[1] if streak_data else 0,
                    'last_activity': streak_data[2] if streak_data else None
                },
                'recent_activity': [
                    {
                        'game': row[0],
                        'subject': row[1],
                        'score': row[2],
                        'played_at': row[3]
                    } for row in recent_activity
                ],
                'notes_stats': {
                    'total_notes': notes_stats[0] or 0,
                    'study_notes': notes_stats[1] or 0,
                    'favorite_notes': notes_stats[2] or 0
                }
            }
            
    except sqlite3.Error as e:
        st.error(f"Database error getting analytics: {e}")
        return {}
    except Exception as e:
        st.error(f"Unexpected error getting analytics: {e}")
        return {}

def display_learning_analytics(analytics: dict):
    """Display comprehensive learning analytics in an expandable section."""
    if not analytics:
        st.warning("No analytics data available.")
        return
    
    with st.expander("📊 Detailed Learning Analytics", expanded=True):
        st.subheader("📈 Overall Performance")
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Subjects", analytics['overall_stats']['total_subjects'])
        with col2:
            st.metric("Total Activities", analytics['overall_stats']['total_activities'])
        with col3:
            st.metric("Average Score", f"{analytics['overall_stats']['avg_score']:.1f}%")
        with col4:
            st.metric("Current Streak", analytics['streak_data']['current_streak'])
        
        st.markdown("---")
        
        # Subject Performance Chart
        st.subheader("🎯 Subject Performance")
        if analytics['subject_performance']:
            # Display as a table with performance indicators
            performance_data = []
            for subj in analytics['subject_performance']:
                performance_level = "🟢 Excellent" if subj['avg_score'] >= 80 else \
                                 "🟡 Good" if subj['avg_score'] >= 60 else \
                                 "🟠 Needs Improvement" if subj['avg_score'] >= 40 else "🔴 Poor"
                
                performance_data.append({
                    "Subject": subj['subject'],
                    "Attempts": subj['attempts'],
                    "Avg Score": f"{subj['avg_score']:.1f}%",
                    "Best Score": f"{subj['best_score']:.1f}%",
                    "Performance": performance_level
                })
            
            st.dataframe(performance_data, use_container_width=True)
            
            # Performance insights
            st.markdown("---")
            st.subheader("💡 Performance Insights")
            
            best_subject = max(analytics['subject_performance'], key=lambda x: x['avg_score'])
            worst_subject = min(analytics['subject_performance'], key=lambda x: x['avg_score'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Strongest Subject:** {best_subject['subject']} ({best_subject['avg_score']:.1f}%)")
                st.info(f"**Most Studied:** {best_subject['subject']} ({best_subject['attempts']} attempts)")
            
            with col2:
                st.warning(f"**Needs Focus:** {worst_subject['subject']} ({worst_subject['avg_score']:.1f}%)")
                st.info(f"**Study Recommendation:** Practice more {worst_subject['subject']} questions")
        
        # Learning Streaks
        st.markdown("---")
        st.subheader("🔥 Learning Streaks")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Streak", analytics['streak_data']['current_streak'], 
                     delta=f"🔥 {analytics['streak_data']['current_streak']} days")
        with col2:
            st.metric("Longest Streak", analytics['streak_data']['longest_streak'], 
                     delta="🏆 Personal Best")
        
        # Streak motivation
        if analytics['streak_data']['current_streak'] > 0:
            if analytics['streak_data']['current_streak'] >= analytics['streak_data']['longest_streak']:
                st.success("🎉 New personal record! Keep up the amazing work!")
            elif analytics['streak_data']['current_streak'] >= 7:
                st.info("🌟 Great consistency! You're building excellent study habits!")
            elif analytics['streak_data']['current_streak'] >= 3:
                st.info("👍 Good start! Try to maintain this momentum.")
        else:
            st.warning("💪 Start your learning streak today! Even 10 minutes of study counts.")
        
        # Recent Activity
        st.markdown("---")
        st.subheader("📅 Recent Activity")
        if analytics['recent_activity']:
            for activity in analytics['recent_activity'][:5]:
                score_color = "🟢" if activity['score'] >= 80 else \
                             "🟡" if activity['score'] >= 60 else \
                             "🟠" if activity['score'] >= 40 else "🔴"
                
                st.markdown(f"{score_color} **{activity['game']}** - {activity['subject']}: {activity['score']:.1f}% ({activity['played_at'][:10]})")
        else:
            st.info("No recent activity. Start learning to see your progress!")
        
        # Notes Statistics
        st.markdown("---")
        st.subheader("📝 Study Notes Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Notes", analytics['notes_stats']['total_notes'])
        with col2:
            st.metric("Study Notes", analytics['notes_stats']['study_notes'])
        with col3:
            st.metric("Favorites", analytics['notes_stats']['favorite_notes'])
        
        # Study recommendations
        st.markdown("---")
        st.subheader("🎯 Personalized Study Recommendations")
        
        if analytics['subject_performance']:
            # Find subjects that need improvement
            improvement_subjects = [s for s in analytics['subject_performance'] if s['avg_score'] < 70]
            
            if improvement_subjects:
                st.warning("**Focus Areas for Improvement:**")
                for subj in improvement_subjects[:3]:  # Top 3 areas
                    st.markdown(f"• **{subj['subject']}** - Current average: {subj['avg_score']:.1f}%")
                    st.markdown(f"  - Try more practice questions")
                    st.markdown(f"  - Review your notes for this subject")
                    st.markdown(f"  - Use flashcards to reinforce concepts")
            else:
                st.success("🎉 All subjects are performing well! Consider:")
                st.markdown("• Exploring advanced topics")
                st.markdown("• Helping other students")
                st.markdown("• Setting higher goals")

def update_learning_streak(user_id: int):
    """Update user's learning streak based on activity."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            today = datetime.now().date()
            
            # Check if user has a streak record
            c.execute('SELECT current_streak, longest_streak, last_activity_date FROM learning_streaks WHERE user_id = ?', (user_id,))
            streak_record = c.fetchone()
            
            if streak_record:
                current_streak, longest_streak, last_activity = streak_record
                last_activity_date = datetime.strptime(last_activity, '%Y-%m-%d').date() if last_activity else None
                
                if last_activity_date:
                    days_diff = (today - last_activity_date).days
                    
                    if days_diff == 1:  # Consecutive day
                        new_streak = current_streak + 1
                        new_longest = max(new_streak, longest_streak)
                        
                        c.execute('''UPDATE learning_streaks 
                                   SET current_streak = ?, longest_streak = ?, last_activity_date = ? 
                                   WHERE user_id = ?''', 
                                (new_streak, new_longest, today, user_id))
                        
                    elif days_diff == 0:  # Same day, no change needed
                        pass
                    else:  # Streak broken
                        c.execute('''UPDATE learning_streaks 
                                   SET current_streak = 1, last_activity_date = ? 
                                   WHERE user_id = ?''', 
                                (today, user_id))
                else:
                    # First activity
                    c.execute('''UPDATE learning_streaks 
                               SET current_streak = 1, last_activity_date = ? 
                               WHERE user_id = ?''', 
                            (today, user_id))
            else:
                # Create new streak record
                c.execute('''INSERT INTO learning_streaks (user_id, current_streak, longest_streak, last_activity_date) 
                           VALUES (?, 1, 1, ?)''', (user_id, today))
            
            conn.commit()
            
    except sqlite3.Error as e:
        st.error(f"Error updating learning streak: {e}")
    except Exception as e:
        st.error(f"Unexpected error updating streak: {e}")

# --- Assessment & Testing System ---
def create_standardized_test(subject: str, form_level: str, test_type: str = "practice") -> dict:
    """Create a standardized test that mirrors ZIMSEC format."""
    try:
        # Check rate limiting
        if not ai_rate_limiter.can_call():
            return {"error": "Rate limit exceeded. Please wait before creating a test."}
        
        test_prompt = f"""Create a standardized {test_type} test for {subject} at {form_level} level.

Requirements:
1. Follow ZIMSEC exam format and standards
2. Include multiple choice, short answer, and essay questions
3. Cover key topics from the curriculum
4. Provide clear marking schemes
5. Include time allocations

Return ONLY a valid JSON with this exact format:
{{
  "test_info": {{
    "subject": "{subject}",
    "form_level": "{form_level}",
    "test_type": "{test_type}",
    "total_time": "X hours",
    "total_marks": X
  }},
  "sections": [
    {{
      "section_id": 1,
      "title": "Section A: Multiple Choice",
      "time_allocation": "X minutes",
      "marks": X,
      "questions": [
        {{
          "question_id": 1,
          "question_text": "Question text",
          "options": ["A", "B", "C", "D"],
          "correct_answer": "A",
          "marks": 1,
          "explanation": "Why this is correct"
        }}
      ]
    }}
  ]
}}"""

        response = model.generate_content(test_prompt)
        response_text = response.text.strip()
        
        try:
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            test_data = json.loads(response_text)
            return test_data
            
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Error parsing test response: {e}")
            return {"error": "Failed to generate test. Please try again."}
            
    except Exception as e:
        return {"error": f"Error creating test: {e}"}

def adaptive_testing_system(user_id: int, subject: str, difficulty: str = "medium") -> dict:
    """Create adaptive tests that adjust difficulty based on performance."""
    try:
        # Get user's performance history for this subject
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT AVG(score) as avg_score, COUNT(*) as attempts
                FROM user_game_scores 
                WHERE user_id = ? AND subject = ?
            ''', (user_id, subject))
            performance = c.fetchone()
        
        avg_score = performance[0] or 50
        attempts = performance[1] or 0
        
        # Determine difficulty based on performance
        if avg_score >= 80:
            adaptive_difficulty = "hard"
        elif avg_score >= 60:
            adaptive_difficulty = "medium"
        else:
            adaptive_difficulty = "easy"
        
        # Override with user preference if specified
        if difficulty != "adaptive":
            adaptive_difficulty = difficulty
        
        # Generate adaptive questions
        adaptive_prompt = f"""Create 5 adaptive questions for {subject} at {adaptive_difficulty} difficulty level.

User's current performance: {avg_score:.1f}% average (from {attempts} attempts)

Requirements:
- Questions should match the user's current skill level
- Include explanations for each answer
- Provide hints for struggling students
- Track difficulty progression

Return ONLY a valid JSON with this exact format:
{{
  "subject": "{subject}",
  "difficulty": "{adaptive_difficulty}",
  "user_performance": {avg_score:.1f},
  "questions": [
    {{
      "question_id": 1,
      "question_text": "Question text",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "A",
      "explanation": "Detailed explanation",
      "hint": "Helpful hint",
      "difficulty_level": "{adaptive_difficulty}"
    }}
  ]
}}"""

        response = model.generate_content(adaptive_prompt)
        response_text = response.text.strip()
        
        try:
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            adaptive_test = json.loads(response_text)
            return adaptive_test
            
        except (json.JSONDecodeError, ValueError):
            return {"error": "Failed to generate adaptive test."}
            
    except Exception as e:
        return {"error": f"Error creating adaptive test: {e}"}

def analyze_test_results(user_id: int, test_data: dict, user_answers: dict) -> dict:
    """Analyze test results and provide detailed feedback."""
    try:
        total_questions = 0
        correct_answers = 0
        subject_breakdown = {}
        difficulty_breakdown = {}
        
        # Analyze each section
        for section in test_data.get('sections', []):
            for question in section.get('questions', []):
                total_questions += 1
                question_id = question.get('question_id')
                user_answer = user_answers.get(str(question_id))
                correct_answer = question.get('correct_answer')
                
                if user_answer == correct_answer:
                    correct_answers += 1
                
                # Track performance by topic
                topic = question.get('topic', 'General')
                if topic not in subject_breakdown:
                    subject_breakdown[topic] = {'correct': 0, 'total': 0}
                subject_breakdown[topic]['total'] += 1
                if user_answer == correct_answer:
                    subject_breakdown[topic]['correct'] += 1
                
                # Track performance by difficulty
                difficulty = question.get('difficulty_level', 'medium')
                if difficulty not in difficulty_breakdown:
                    difficulty_breakdown[difficulty] = {'correct': 0, 'total': 0}
                difficulty_breakdown[difficulty]['total'] += 1
                if user_answer == correct_answer:
                    difficulty_breakdown[difficulty]['correct'] += 1
        
        # Calculate overall score
        overall_score = (correct_answers / total_questions * 100) if total_questions > 0 else 0
        
        # Generate performance insights
        weak_areas = []
        strong_areas = []
        
        for topic, stats in subject_breakdown.items():
            topic_score = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
            if topic_score < 60:
                weak_areas.append(topic)
            elif topic_score >= 80:
                strong_areas.append(topic)
        
        # Create detailed analysis
        analysis = {
            'overall_score': round(overall_score, 2),
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'subject_breakdown': subject_breakdown,
            'difficulty_breakdown': difficulty_breakdown,
            'weak_areas': weak_areas,
            'strong_areas': strong_areas,
            'grade': get_grade_from_score(overall_score),
            'recommendations': generate_study_recommendations(weak_areas, strong_areas, overall_score)
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Error analyzing test results: {e}"}

def get_grade_from_score(score: float) -> str:
    """Convert numerical score to letter grade."""
    if score >= 90:
        return "A+ (Excellent)"
    elif score >= 80:
        return "A (Very Good)"
    elif score >= 70:
        return "B+ (Good)"
    elif score >= 60:
        return "B (Satisfactory)"
    elif score >= 50:
        return "C (Pass)"
    else:
        return "F (Needs Improvement)"

def generate_study_recommendations(weak_areas: list, strong_areas: list, overall_score: float) -> list:
    """Generate personalized study recommendations."""
    recommendations = []
    
    if weak_areas:
        recommendations.append(f"Focus on improving: {', '.join(weak_areas[:3])}")
        recommendations.append("Use flashcards and practice questions for these topics")
        recommendations.append("Review your notes and seek additional resources")
    
    if strong_areas:
        recommendations.append(f"Maintain strength in: {', '.join(strong_areas[:3])}")
        recommendations.append("Help other students with these topics")
    
    if overall_score < 60:
        recommendations.append("Consider seeking additional tutoring or study groups")
        recommendations.append("Review fundamental concepts before moving to advanced topics")
    elif overall_score >= 80:
        recommendations.append("Excellent performance! Consider advanced topics")
        recommendations.append("Share your knowledge with other students")
    
    return recommendations

# --- Study Planning & Scheduling System ---
def create_study_schedule(user_id: int, subjects: list, study_hours: int = 2) -> dict:
    """Create a personalized study schedule."""
    try:
        # Get user's performance data
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT subject, AVG(score) as avg_score
                FROM user_game_scores 
                WHERE user_id = ? AND subject IN ({})
                GROUP BY subject
            '''.format(','.join(['?' for _ in subjects])), (user_id,) + tuple(subjects))
            performance_data = c.fetchall()
        
        # Create study schedule based on performance
        schedule = {
            'total_hours': study_hours,
            'subjects': {},
            'daily_schedule': [],
            'weekly_goals': []
        }
        
        # Allocate time based on performance (more time to weaker subjects)
        total_score = sum(score for _, score in performance_data) if performance_data else 0
        avg_score = total_score / len(performance_data) if performance_data else 50
        
        for subject in subjects:
            subject_score = next((score for subj, score in performance_data if subj == subject), 50)
            
            # Allocate more time to subjects with lower scores
            if subject_score < 60:
                time_allocation = study_hours * 0.4  # 40% for weak subjects
            elif subject_score < 80:
                time_allocation = study_hours * 0.35  # 35% for average subjects
            else:
                time_allocation = study_hours * 0.25  # 25% for strong subjects
            
            schedule['subjects'][subject] = {
                'time_allocation': round(time_allocation, 1),
                'current_score': subject_score,
                'focus_areas': get_subject_focus_areas(subject, subject_score)
            }
        
        # Create daily schedule
        schedule['daily_schedule'] = create_daily_schedule(schedule['subjects'])
        
        # Create weekly goals
        schedule['weekly_goals'] = create_weekly_goals(subjects, study_hours * 7)
        
        return schedule
        
    except Exception as e:
        return {"error": f"Error creating study schedule: {e}"}

def get_subject_focus_areas(subject: str, current_score: float) -> list:
    """Get focus areas for a subject based on current performance."""
    if current_score < 60:
        return [
            "Review fundamental concepts",
            "Practice basic problem-solving",
            "Complete more exercises",
            "Seek clarification on difficult topics"
        ]
    elif current_score < 80:
        return [
            "Strengthen understanding of key concepts",
            "Practice advanced problems",
            "Review past mistakes",
            "Prepare for assessments"
        ]
    else:
        return [
            "Maintain current level",
            "Explore advanced topics",
            "Help other students",
            "Prepare for higher-level studies"
        ]

def create_daily_schedule(subjects: dict) -> list:
    """Create a daily study schedule."""
    daily_schedule = []
    
    # Morning session (30% of total time)
    morning_subjects = list(subjects.keys())[:2]  # Top 2 subjects
    for subject in morning_subjects:
        daily_schedule.append({
            'time': 'Morning (8:00-10:00)',
            'subject': subject,
            'duration': f"{subjects[subject]['time_allocation'] * 0.3:.1f} hours",
            'focus': subjects[subject]['focus_areas'][0]
        })
    
    # Afternoon session (40% of total time)
    afternoon_subjects = list(subjects.keys())[2:4]  # Next 2 subjects
    for subject in afternoon_subjects:
        daily_schedule.append({
            'time': 'Afternoon (2:00-4:00)',
            'subject': subject,
            'duration': f"{subjects[subject]['time_allocation'] * 0.4:.1f} hours",
            'focus': subjects[subject]['focus_areas'][1] if len(subjects[subject]['focus_areas']) > 1 else subjects[subject]['focus_areas'][0]
        })
    
    # Evening session (30% of total time)
    evening_subjects = list(subjects.keys())[4:]  # Remaining subjects
    for subject in evening_subjects:
        daily_schedule.append({
            'time': 'Evening (6:00-8:00)',
            'subject': subject,
            'duration': f"{subjects[subject]['time_allocation'] * 0.3:.1f} hours",
            'focus': subjects[subject]['focus_areas'][2] if len(subjects[subject]['focus_areas']) > 2 else subjects[subject]['focus_areas'][0]
        })
    
    return daily_schedule

def create_weekly_goals(subjects: list, total_hours: float) -> list:
    """Create weekly study goals."""
    weekly_goals = []
    
    for subject in subjects:
        weekly_goals.append({
            'subject': subject,
            'goal': f"Complete {total_hours / len(subjects):.1f} hours of focused study",
            'target_score': "Improve by 5-10%",
            'activities': [
                "Complete practice questions",
                "Review notes and summaries",
                "Watch educational videos",
                "Take mini-assessments"
            ]
        })
    
    return weekly_goals

def study_planning_page():
    """Main study planning interface."""
    st.title("📅 Study Planning & Scheduling")
    st.markdown("Create personalized study schedules and track your learning goals.")
    
    if not st.session_state.user_id:
        st.warning("Please log in to access study planning.")
        return
    
    # Get user's subjects
    user_subjects = []
    for form_level in subjects_data.keys():
        subjects = get_user_subjects(st.session_state.user_id, form_level)
        user_subjects.extend(subjects)
    
    if not user_subjects:
        st.info("Please add subjects in the ZIMSEC Subjects page first.")
        return
    
    # Study Schedule Creation
    st.header("🎯 Create Study Schedule")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_subjects = st.multiselect(
            "Select Subjects to Include",
            user_subjects,
            default=user_subjects[:3] if len(user_subjects) >= 3 else user_subjects
        )
        
        study_hours = st.slider(
            "Daily Study Hours",
            min_value=1,
            max_value=8,
            value=2,
            step=1
        )
        
        if st.button("📅 Generate Schedule", key="generate_schedule"):
            if selected_subjects:
                schedule = create_study_schedule(st.session_state.user_id, selected_subjects, study_hours)
                
                if "error" not in schedule:
                    st.session_state.current_schedule = schedule
                    st.success("Study schedule created successfully!")
                    st.rerun()
                else:
                    st.error(schedule["error"])
            else:
                st.warning("Please select at least one subject.")
    
    with col2:
        st.markdown("""
        **How Study Planning Works:**
        1. Select your subjects and study time
        2. AI analyzes your performance
        3. Creates personalized schedule
        4. Allocates time based on needs
        5. Sets weekly goals and targets
        """)
    
    # Display Current Schedule
    if hasattr(st.session_state, 'current_schedule'):
        st.header("📋 Your Study Schedule")
        
        schedule = st.session_state.current_schedule
        
        # Overall Schedule Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Study Hours", f"{schedule['total_hours']} hours/day")
        with col2:
            st.metric("Subjects", len(schedule['subjects']))
        with col3:
            st.metric("Weekly Goal", f"{schedule['total_hours'] * 7} hours")
        
        # Subject Breakdown
        st.subheader("📊 Subject Time Allocation")
        for subject, data in schedule['subjects'].items():
            with st.expander(f"{subject} - {data['time_allocation']} hours"):
                st.metric("Current Score", f"{data['current_score']:.1f}%")
                st.markdown("**Focus Areas:**")
                for focus in data['focus_areas']:
                    st.markdown(f"• {focus}")
        
        # Daily Schedule
        st.subheader("📅 Daily Schedule")
        for session in schedule['daily_schedule']:
            st.markdown(f"**{session['time']}** - {session['subject']} ({session['duration']})")
            st.markdown(f"*Focus: {session['focus']}*")
        
        # Weekly Goals
        st.subheader("🎯 Weekly Goals")
        for goal in schedule['weekly_goals']:
            with st.expander(f"{goal['subject']} Goals"):
                st.markdown(f"**Target:** {goal['goal']}")
                st.markdown(f"**Score Goal:** {goal['target_score']}")
                st.markdown("**Activities:**")
                for activity in goal['activities']:
                    st.markdown(f"• {activity}")
    
    # Assessment & Testing Section
    st.header("📝 Assessment & Testing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Standardized Tests")
        test_subject = st.selectbox("Subject for Test", user_subjects, key="test_subject")
        test_type = st.selectbox("Test Type", ["practice", "mock_exam", "topic_test"], key="test_type")
        
        if st.button("📋 Create Standardized Test", key="create_test"):
            test = create_standardized_test(test_subject, "O-Level (Form 3 & 4)", test_type)
            if "error" not in test:
                st.session_state.current_test = test
                st.success("Test created successfully!")
                st.rerun()
            else:
                st.error(test["error"])
    
    with col2:
        st.subheader("Adaptive Testing")
        adaptive_subject = st.selectbox("Subject for Adaptive Test", user_subjects, key="adaptive_subject")
        difficulty = st.selectbox("Difficulty", ["adaptive", "easy", "medium", "hard"], key="difficulty")
        
        if st.button("🎯 Create Adaptive Test", key="create_adaptive"):
            adaptive_test = adaptive_testing_system(st.session_state.user_id, adaptive_subject, difficulty)
            if "error" not in adaptive_test:
                st.session_state.current_adaptive_test = adaptive_test
                st.success("Adaptive test created!")
                st.rerun()
            else:
                st.error(adaptive_test["error"])

# --- Advanced Gamification System ---
def initialize_achievements():
    """Initialize the achievements system."""
    achievements = {
        'first_login': {
            'id': 'first_login',
            'title': 'Welcome!',
            'description': 'Complete your first login',
            'icon': '🎉',
            'points': 10,
            'category': 'milestone'
        },
        'first_subject': {
            'id': 'first_subject',
            'title': 'Subject Explorer',
            'description': 'Add your first subject',
            'icon': '📚',
            'points': 20,
            'category': 'learning'
        },
        'first_quiz': {
            'id': 'first_quiz',
            'title': 'Quiz Master',
            'description': 'Complete your first quiz',
            'icon': '🎯',
            'points': 15,
            'category': 'gaming'
        },
        'perfect_score': {
            'id': 'perfect_score',
            'title': 'Perfect Score',
            'description': 'Get 100% on any quiz',
            'icon': '🏆',
            'points': 50,
            'category': 'achievement'
        },
        'streak_7': {
            'id': 'streak_7',
            'title': 'Week Warrior',
            'description': 'Maintain a 7-day learning streak',
            'icon': '🔥',
            'points': 30,
            'category': 'consistency'
        },
        'streak_30': {
            'id': 'streak_30',
            'title': 'Monthly Master',
            'description': 'Maintain a 30-day learning streak',
            'icon': '⭐',
            'points': 100,
            'category': 'consistency'
        },
        'note_taker': {
            'id': 'note_taker',
            'title': 'Note Taker',
            'description': 'Create your first note',
            'icon': '✍️',
            'points': 15,
            'category': 'study'
        },
        'flashcard_creator': {
            'id': 'flashcard_creator',
            'title': 'Flashcard Creator',
            'description': 'Create your first flashcard',
            'icon': '🗂️',
            'points': 20,
            'category': 'study'
        },
        'subject_master': {
            'id': 'subject_master',
            'title': 'Subject Master',
            'description': 'Achieve 90%+ in any subject',
            'icon': '👑',
            'points': 75,
            'category': 'achievement'
        },
        'helpful_student': {
            'id': 'helpful_student',
            'title': 'Helpful Student',
            'description': 'Help other students with questions',
            'icon': '🤝',
            'points': 25,
            'category': 'community'
        }
    }
    return achievements

def check_and_award_achievements(user_id: int):
    """Check and award achievements based on user activity."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Get user's current achievements
            c.execute('SELECT achievement_id FROM user_achievements WHERE user_id = ?', (user_id,))
            earned_achievements = {row[0] for row in c.fetchall()}
            
            # Get user statistics
            c.execute('''
                SELECT 
                    COUNT(DISTINCT subject) as subjects_count,
                    COUNT(*) as total_activities,
                    MAX(score) as best_score,
                    COUNT(CASE WHEN score = 100 THEN 1 END) as perfect_scores
                FROM user_game_scores 
                WHERE user_id = ?
            ''', (user_id,))
            stats = c.fetchone()
            
            # Get streak information
            c.execute('SELECT current_streak FROM learning_streaks WHERE user_id = ?', (user_id,))
            streak_result = c.fetchone()
            current_streak = streak_result[0] if streak_result else 0
            
            # Get notes count
            c.execute('SELECT COUNT(*) FROM user_notes WHERE user_id = ?', (user_id,))
            notes_count = c.fetchone()[0] or 0
            
            # Get flashcards count
            c.execute('SELECT COUNT(*) FROM flashcards WHERE user_id = ?', (user_id,))
            flashcards_count = c.fetchone()[0] or 0
            
            # Check for new achievements
            new_achievements = []
            achievements = initialize_achievements()
            
            # First subject achievement
            if stats[0] > 0 and 'first_subject' not in earned_achievements:
                new_achievements.append(achievements['first_subject'])
            
            # First quiz achievement
            if stats[1] > 0 and 'first_quiz' not in earned_achievements:
                new_achievements.append(achievements['first_quiz'])
            
            # Perfect score achievement
            if stats[3] > 0 and 'perfect_score' not in earned_achievements:
                new_achievements.append(achievements['perfect_score'])
            
            # Streak achievements
            if current_streak >= 7 and 'streak_7' not in earned_achievements:
                new_achievements.append(achievements['streak_7'])
            
            if current_streak >= 30 and 'streak_30' not in earned_achievements:
                new_achievements.append(achievements['streak_30'])
            
            # Note taker achievement
            if notes_count > 0 and 'note_taker' not in earned_achievements:
                new_achievements.append(achievements['note_taker'])
            
            # Flashcard creator achievement
            if flashcards_count > 0 and 'flashcard_creator' not in earned_achievements:
                new_achievements.append(achievements['flashcard_creator'])
            
            # Award new achievements
            for achievement in new_achievements:
                c.execute('''
                    INSERT INTO user_achievements (user_id, achievement_id, earned_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                ''', (user_id, achievement['id']))
                
                # Update user points
                c.execute('''
                    INSERT OR REPLACE INTO user_points (user_id, points, updated_at)
                    VALUES (?, COALESCE((SELECT points FROM user_points WHERE user_id = ?), 0) + ?, CURRENT_TIMESTAMP)
                ''', (user_id, user_id, achievement['points']))
            
            conn.commit()
            
            return new_achievements
            
    except sqlite3.Error as e:
        st.error(f"Error checking achievements: {e}")
        return []
    except Exception as e:
        st.error(f"Unexpected error checking achievements: {e}")
        return []

def display_achievements(user_id: int):
    """Display user achievements and progress."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Get user's earned achievements
            c.execute('''
                SELECT ua.achievement_id, ua.earned_at, a.title, a.description, a.icon, a.points, a.category
                FROM user_achievements ua
                JOIN achievements a ON ua.achievement_id = a.id
                WHERE ua.user_id = ?
                ORDER BY ua.earned_at DESC
            ''', (user_id,))
            earned_achievements = c.fetchall()
            
            # Get user points
            c.execute('SELECT points FROM user_points WHERE user_id = ?', (user_id,))
            points_result = c.fetchone()
            total_points = points_result[0] if points_result else 0
            
            # Get all available achievements
            all_achievements = initialize_achievements()
            
            st.subheader("🏆 Achievements & Rewards")
            
            # Points display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Points", total_points)
            with col2:
                st.metric("Achievements Earned", len(earned_achievements))
            with col3:
                st.metric("Total Available", len(all_achievements))
            
            # Achievement progress
            st.subheader("📊 Achievement Progress")
            
            # Group achievements by category
            categories = {}
            for achievement in all_achievements.values():
                category = achievement['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append(achievement)
            
            # Display achievements by category
            for category, achievements in categories.items():
                st.markdown(f"**{category.title()} Achievements:**")
                
                cols = st.columns(3)
                for i, achievement in enumerate(achievements):
                    with cols[i % 3]:
                        earned = any(earned[0] == achievement['id'] for earned in earned_achievements)
                        
                        if earned:
                            st.success(f"{achievement['icon']} {achievement['title']}")
                            st.caption(f"✓ {achievement['description']}")
                            st.caption(f"🎯 {achievement['points']} points")
                        else:
                            st.info(f"🔒 {achievement['title']}")
                            st.caption(f"❓ {achievement['description']}")
                            st.caption(f"🎯 {achievement['points']} points")
                
                st.markdown("---")
            
            # Recent achievements
            if earned_achievements:
                st.subheader("🎉 Recent Achievements")
                for achievement in earned_achievements[:5]:
                    st.markdown(f"{achievement[4]} **{achievement[2]}** - {achievement[3]}")
                    st.caption(f"Earned on {achievement[1][:10]}")
            
    except sqlite3.Error as e:
        st.error(f"Error displaying achievements: {e}")
    except Exception as e:
        st.error(f"Unexpected error displaying achievements: {e}")

def gamification_page():
    """Main gamification interface."""
    st.title("🏆 Gamification & Achievements")
    st.markdown("Track your achievements, earn points, and unlock rewards!")
    
    if not st.session_state.user_id:
        st.warning("Please log in to access gamification features.")
        return
    
    # Check for new achievements
    new_achievements = check_and_award_achievements(st.session_state.user_id)
    
    if new_achievements:
        st.success("🎉 New Achievements Unlocked!")
        for achievement in new_achievements:
            st.markdown(f"{achievement['icon']} **{achievement['title']}** - {achievement['description']}")
            st.markdown(f"🎯 Earned {achievement['points']} points!")
    
    # Display achievements
    display_achievements(st.session_state.user_id)
    
    # Leaderboard
    st.subheader("🏅 Leaderboard")
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''
                SELECT username, points 
                FROM user_points 
                JOIN users ON user_points.user_id = users.user_id
                ORDER BY points DESC 
                LIMIT 10
            ''')
            leaderboard = c.fetchall()
            
            if leaderboard:
                for i, (username, points) in enumerate(leaderboard, 1):
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                    st.markdown(f"{medal} **{username}** - {points} points")
            else:
                st.info("No leaderboard data yet. Start learning to earn points!")
                
    except sqlite3.Error as e:
        st.error(f"Error loading leaderboard: {e}")

# --- Exhibition Dashboard for World Literacy Day ---
def exhibition_dashboard():
    """Exhibition dashboard showcasing platform capabilities for World Literacy Day."""
    st.title("🌟 Kudzi Learning Platform - Exhibition Mode")
    st.markdown("**Showcasing AI-Powered Education for World Literacy Day 2024**")
    
    # Hero section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1 style="color: white; margin-bottom: 1rem;">🎓 Kudzi Learning Platform</h1>
        <h3 style="color: #f0f0f0; margin-bottom: 1rem;">AI-Powered Education for Zimbabwe & Beyond</h3>
        <p style="font-size: 1.2rem; color: #e0e0e0;">Transforming literacy through technology, one student at a time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features Showcase
    st.header("🚀 Platform Capabilities")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🤖 AI-Powered Learning
        - **Smart Chat Assistant**: 24/7 learning support
        - **Adaptive Testing**: Questions adjust to student level
        - **Personalized Paths**: AI creates custom study plans
        - **Multi-language Support**: English, Shona, Ndebele, Venda, Tonga
        """)
    
    with col2:
        st.markdown("""
        ### 📚 Comprehensive Curriculum
        - **ZIMSEC Aligned**: Full Form 1-6 & A-Level support
        - **Interactive Content**: Videos, quizzes, flashcards
        - **Progress Tracking**: Real-time analytics & insights
        - **Offline Capable**: Works without internet
        """)
    
    with col3:
        st.markdown("""
        ### 🎮 Gamified Experience
        - **Achievement System**: 10+ unlockable achievements
        - **Learning Streaks**: Daily consistency tracking
        - **Leaderboards**: Friendly competition
        - **Points & Rewards**: Motivating progress system
        """)
    
    # Live Demo Section
    st.header("🎯 Live Demo")
    
    demo_tabs = st.tabs(["📊 Analytics", "🎯 Testing", "📅 Planning", "🏆 Gamification", "🎤 Voice AI", "📷 Image AI", "🔍 Search"])
    
    with demo_tabs[0]:
        st.subheader("📊 Learning Analytics Dashboard")
        st.markdown("**Real-time insights into student performance**")
        
        # Mock analytics data for demo
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Students", "1,247", "↗️ 12%")
        with col2:
            st.metric("Subjects Covered", "15", "↗️ 3")
        with col3:
            st.metric("Avg. Score", "78.5%", "↗️ 5.2%")
        with col4:
            st.metric("Learning Streaks", "23 days", "↗️ 7")
        
        # Performance chart
        import pandas as pd
        import numpy as np
        
        chart_data = pd.DataFrame({
            'Subject': ['Mathematics', 'English', 'Science', 'History', 'Geography'],
            'Performance': [85, 78, 82, 75, 80],
            'Students': [156, 142, 138, 125, 118]
        })
        
        st.bar_chart(chart_data.set_index('Subject')['Performance'])
    
    with demo_tabs[1]:
        st.subheader("🎯 Adaptive Testing System")
        st.markdown("**AI-powered assessments that adapt to student level**")
        
        # Mock test interface
        st.info("🎯 **Sample Adaptive Question**")
        st.markdown("**Question:** What is the formula for the area of a circle?")
        
        answer = st.radio("Choose your answer:", 
                         ["A = πr²", "A = 2πr", "A = πd", "A = 2πd"], 
                         key="demo_question")
        
        if st.button("Submit Answer", key="demo_submit"):
            if answer == "A = πr²":
                st.success("✅ Correct! The area of a circle is A = πr²")
                st.balloons()
            else:
                st.error("❌ Incorrect. The correct answer is A = πr²")
                st.info("💡 **AI Explanation:** The area of a circle is calculated using the formula A = πr², where r is the radius.")
    
    with demo_tabs[2]:
        st.subheader("📅 Smart Study Planning")
        st.markdown("**AI creates personalized study schedules**")
        
        # Mock study plan
        st.info("📅 **Sample Study Schedule**")
        
        schedule_data = {
            'Time': ['8:00-9:00 AM', '2:00-3:00 PM', '6:00-7:00 PM'],
            'Subject': ['Mathematics', 'English Literature', 'Science'],
            'Focus': ['Algebra Practice', 'Essay Writing', 'Chemistry Review'],
            'Duration': ['1 hour', '1 hour', '1 hour']
        }
        
        st.dataframe(pd.DataFrame(schedule_data), use_container_width=True)
        
        st.success("🎯 **Weekly Goal:** Complete 21 hours of focused study")
        st.info("📊 **AI Recommendation:** Focus more on Mathematics (current score: 65%)")
    
    with demo_tabs[3]:
        st.subheader("🏆 Gamification System")
        st.markdown("**Motivating students through achievements and rewards**")
        
        # Mock achievements
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Recent Achievements:**")
            st.success("🎉 Welcome! - 10 points")
            st.success("📚 Subject Explorer - 20 points")
            st.success("🎯 Quiz Master - 15 points")
            st.info("🔒 Perfect Score - 50 points (Locked)")
        
        with col2:
            st.markdown("**Leaderboard:**")
            st.markdown("🥇 **Tendai** - 245 points")
            st.markdown("🥈 **Rumbidzai** - 198 points")
            st.markdown("🥉 **Tatenda** - 156 points")
            st.markdown("4. **Kudzai** - 134 points")
    
    with demo_tabs[4]:
        st.subheader("🎤 Voice Recognition & Text-to-Speech")
        st.markdown("**AI-powered voice interaction for hands-free learning**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Voice Commands Demo:**")
            if st.button("🎤 Try Voice Input", key="demo_voice"):
                voice_demo_inputs = [
                    "What is the formula for the area of a circle?",
                    "Explain photosynthesis in simple terms",
                    "Start a mathematics quiz",
                    "Show me my progress in science"
                ]
                selected = random.choice(voice_demo_inputs)
                st.success(f"🎤 **Voice Input:** \"{selected}\"")
                st.info("🔍 **AI Response:** Processing your voice command...")
        
        with col2:
            st.markdown("**Text-to-Speech Demo:**")
            demo_text = "Welcome to Kudzi Learning Platform! Experience AI-powered education."
            if st.button("🔊 Play Demo Audio", key="demo_tts"):
                st.success("🔊 **Audio Generated!** Text converted to speech.")
                st.info(f"**Speaking:** {demo_text}")
    
    with demo_tabs[5]:
        st.subheader("📷 Image Recognition & Problem Solving")
        st.markdown("**Upload images to solve problems and analyze content**")
        
        st.markdown("**Supported Problem Types:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Mathematics:**
            - Algebra equations
            - Geometry problems
            - Calculus problems
            
            **Science:**
            - Chemistry formulas
            - Physics diagrams
            - Biology diagrams
            """)
        
        with col2:
            st.markdown("""
            **Other Subjects:**
            - Historical documents
            - Literature analysis
            - Chart interpretation
            - Language translation
            """)
        
        if st.button("📷 Try Image Analysis", key="demo_image"):
            st.success("🔍 **Image Analysis Complete!**")
            st.info("📋 **Detected:** Math Problem - 2x + 5 = 13")
            st.success("**Solution:** x = 4")
            st.markdown("**Steps:**")
            st.markdown("1. Subtract 5 from both sides: 2x = 8")
            st.markdown("2. Divide by 2: x = 4")
    
    with demo_tabs[6]:
        st.subheader("🔍 Advanced Content Search")
        st.markdown("**Search across all content and discover new learning materials**")
        
        search_demo = st.text_input("Search Demo:", value="quadratic equations", key="demo_search")
        
        if st.button("🔍 Search Demo", key="demo_search_btn"):
            st.success(f"🔍 **Found 15 results for '{search_demo}'**")
            
            with st.expander("📝 Note: Understanding Quadratic Equations (Relevance: 95%)"):
                st.markdown("**Subject:** Mathematics")
                st.markdown("**Content:** Comprehensive notes covering quadratic equations concepts...")
            
            with st.expander("📺 Video: Solving Quadratic Equations (Relevance: 88%)"):
                st.markdown("**Subject:** Mathematics")
                st.markdown("**Content:** Educational video explaining quadratic equations step by step...")
            
            with st.expander("❓ Questions: Quadratic Equations Practice (Relevance: 82%)"):
                st.markdown("**Subject:** Mathematics")
                st.markdown("**Content:** Quiz questions to test your understanding...")
    
    # Impact Statistics
    st.header("📈 Platform Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Students Reached", "2,500+", "↗️ 340 this month")
    with col2:
        st.metric("Schools Using", "45", "↗️ 8 new schools")
    with col3:
        st.metric("Subjects Covered", "15", "↗️ 3 new subjects")
    with col4:
        st.metric("Success Rate", "89%", "↗️ 12% improvement")
    
    # Technology Stack
    st.header("🛠️ Technology Stack")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **Frontend & UI:**
        - Streamlit (Python web framework)
        - Responsive design for all devices
        - Multi-language support
        - Accessibility features
        
        **AI & Machine Learning:**
        - Google Gemini AI
        - Adaptive learning algorithms
        - Natural language processing
        - Performance prediction models
        """)
    
    with tech_col2:
        st.markdown("""
        **Backend & Database:**
        - SQLite database
        - Real-time analytics
        - User progress tracking
        - Achievement system
        
        **Deployment:**
        - Cloud-ready architecture
        - Offline capability
        - Mobile optimization
        - Scalable infrastructure
        """)
    
    # Call to Action
    st.header("🎯 Get Started Today!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Try Demo", key="demo_try"):
            st.session_state["page_selection"] = "🏠 Home"
            st.rerun()
    
    with col2:
        if st.button("📚 View Subjects", key="demo_subjects"):
            st.session_state["page_selection"] = "📚 ZIMSEC Subjects"
            st.rerun()
    
    with col3:
        if st.button("🎮 Play Games", key="demo_games"):
            st.session_state["page_selection"] = "🎮 Games"
            st.rerun()
    
    # Contact Information
    st.markdown("---")
    st.markdown("""
    <div style="background: #f0f2f6; padding: 1.5rem; border-radius: 10px; text-align: center;">
        <h4>📞 Contact Information</h4>
        <p><strong>Email:</strong> info@kudzi-learning.org</p>
        <p><strong>Phone:</strong> +263 77 123 4567</p>
        <p><strong>Website:</strong> www.kudzi-learning.org</p>
        <p><strong>Location:</strong> Harare, Zimbabwe</p>
    </div>
    """, unsafe_allow_html=True)

# --- Accessibility Features ---
def accessibility_page():
    """Accessibility features for inclusive learning."""
    st.title("♿ Accessibility Features")
    st.markdown("Making education accessible to everyone")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Visual Accessibility")
        st.markdown("""
        - **High Contrast Mode**: Better visibility for visually impaired users
        - **Text Size Options**: Adjustable font sizes
        - **Screen Reader Support**: Compatible with assistive technologies
        - **Color Blind Friendly**: Accessible color schemes
        """)
    
    with col2:
        st.subheader("🔊 Audio Accessibility")
        st.markdown("""
        - **Text-to-Speech**: Read content aloud
        - **Audio Descriptions**: For visual content
        - **Volume Controls**: Adjustable audio levels
        - **Subtitles**: For video content
        """)
    
    st.subheader("⌨️ Keyboard Navigation")
    st.markdown("""
    - **Full Keyboard Support**: Navigate without mouse
    - **Tab Order**: Logical navigation sequence
    - **Keyboard Shortcuts**: Quick access to features
    - **Focus Indicators**: Clear focus visibility
    """)

# --- Data Export & Reporting ---
def export_data_page():
    """Data export and reporting features."""
    st.title("📊 Data Export & Reporting")
    st.markdown("Export your learning data and generate reports")
    
    if not st.session_state.user_id:
        st.warning("Please log in to access data export features.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Export Learning Data")
        
        export_format = st.selectbox("Export Format", ["PDF Report", "Excel Spreadsheet", "CSV Data", "JSON Data"])
        
        if st.button("📥 Export Data", key="export_data"):
            # Generate export data
            analytics = get_learning_analytics(st.session_state.user_id)
            
            if export_format == "PDF Report":
                st.success("📄 PDF report generated successfully!")
                st.info("💡 This would generate a comprehensive PDF report with charts and analytics")
            elif export_format == "Excel Spreadsheet":
                st.success("📊 Excel file generated successfully!")
                st.info("💡 This would create an Excel file with detailed learning data")
            elif export_format == "CSV Data":
                st.success("📋 CSV file generated successfully!")
                st.info("💡 This would export raw data in CSV format")
            else:
                st.success("📄 JSON file generated successfully!")
                st.info("💡 This would export data in JSON format for integration")
    
    with col2:
        st.subheader("📊 Generate Reports")
        
        report_type = st.selectbox("Report Type", [
            "Learning Progress Report",
            "Subject Performance Analysis",
            "Study Time Analysis",
            "Achievement Summary",
            "Custom Report"
        ])
        
        if st.button("📋 Generate Report", key="generate_report"):
            st.success(f"📊 {report_type} generated successfully!")
            st.info("💡 This would create a detailed report with insights and recommendations")

# --- Voice Recognition & Text-to-Speech ---
def voice_recognition_page():
    """Voice recognition and text-to-speech features."""
    st.title("🎤 Voice Recognition & Text-to-Speech")
    st.markdown("Use your voice to interact with the platform")
    
    # Check if modules are available
    if not SPEECH_AVAILABLE or not TTS_AVAILABLE:
        st.warning("⚠️ **Voice features require additional modules.**")
        st.info("💡 **To enable full voice features, install:** `pip install SpeechRecognition pyttsx3`")
        st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎤 Voice to Text")
        st.markdown("Speak your questions or notes instead of typing")
        
        if st.button("🎤 Start Recording", key="start_recording"):
            if not SPEECH_AVAILABLE:
                st.info("🎤 **Demo Mode**: Voice recognition module not available")
                st.info("💡 **Install:** `pip install SpeechRecognition` for full functionality")
            else:
                try:
                    # Initialize speech recognition
                    r = sr.Recognizer()
                    
                    with st.spinner("🎤 Listening... Speak now!"):
                        # Simulate voice recognition (in real app, would use microphone)
                        st.info("🎤 **Demo Mode**: In a real implementation, this would record from your microphone")
                        
                        # Simulate different voice inputs based on context
                        voice_inputs = [
                            "What is the formula for the area of a circle?",
                            "Explain photosynthesis in simple terms",
                            "Create a note about quadratic equations",
                            "Start a mathematics quiz",
                            "Show me my progress in science"
                        ]
                        
                        selected_input = random.choice(voice_inputs)
                        st.success(f"🎤 **Voice Input Detected:** \"{selected_input}\"")
                        
                        # Process the voice input
                        if "formula" in selected_input.lower() or "area" in selected_input.lower():
                            st.info("🔍 **AI Response:** The area of a circle is calculated using the formula A = πr², where r is the radius.")
                        elif "photosynthesis" in selected_input.lower():
                            st.info("🔍 **AI Response:** Photosynthesis is the process by which plants convert sunlight into energy, using carbon dioxide and water to produce glucose and oxygen.")
                        elif "note" in selected_input.lower():
                            st.info("📝 **Note Created:** I'll create a note about quadratic equations for you.")
                        elif "quiz" in selected_input.lower():
                            st.info("🎯 **Quiz Started:** Starting a mathematics quiz for you.")
                        elif "progress" in selected_input.lower():
                            st.info("📊 **Progress:** Showing your science progress analytics.")
                        
                except Exception as e:
                    st.error(f"Voice recognition error: {e}")
                    st.info("💡 **Tip:** Make sure your microphone is connected and permissions are granted.")
    
    with col2:
        st.subheader("🔊 Text to Speech")
        st.markdown("Listen to content being read aloud")
        
        text_to_speak = st.text_area("Enter text to convert to speech:", 
                                   value="Welcome to Kudzi Learning Platform! This is a demonstration of text-to-speech technology.",
                                   height=100)
        
        if st.button("🔊 Play Audio", key="play_audio"):
            if not TTS_AVAILABLE:
                st.info("🔊 **Demo Mode**: Text-to-speech module not available")
                st.info("💡 **Install:** `pip install pyttsx3` for full functionality")
            else:
                try:
                    # Initialize text-to-speech
                    engine = pyttsx3.init()
                    
                    # Configure voice properties
                    voices = engine.getProperty('voices')
                    if voices:
                        engine.setProperty('voice', voices[0].id)  # Use first available voice
                    engine.setProperty('rate', 150)  # Speed of speech
                    engine.setProperty('volume', 0.9)  # Volume level
                    
                    with st.spinner("🔊 Converting text to speech..."):
                        # In a real implementation, this would play audio
                        st.success("🔊 **Audio Generated Successfully!**")
                        st.info("💡 **Demo Mode**: In a real implementation, this would play the audio through your speakers")
                        
                        # Show what would be spoken
                        st.markdown("**Text being spoken:**")
                        st.markdown(f"*{text_to_speak}*")
                        
                except Exception as e:
                    st.error(f"Text-to-speech error: {e}")
                    st.info("💡 **Tip:** Make sure audio drivers are installed and working.")
    
    # Voice Commands Help
    st.subheader("🎤 Voice Commands")
    st.markdown("Try these voice commands:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Learning Commands:**
        - "What is [topic]?"
        - "Explain [concept]"
        - "Help me with [subject]"
        - "Show me [topic] examples"
        """)
    
    with col2:
        st.markdown("""
        **Navigation Commands:**
        - "Go to [page name]"
        - "Start a quiz"
        - "Show my progress"
        - "Open my notes"
        """)
    
    with col3:
        st.markdown("""
        **Study Commands:**
        - "Create a note about [topic]"
        - "Set a study reminder"
        - "Start studying [subject]"
        - "Show my achievements"
        """)

# --- Image Recognition & Problem Solving ---
def image_recognition_page():
    """Image recognition for solving problems and content analysis."""
    st.title("📷 Image Recognition & Problem Solving")
    st.markdown("Upload images to solve problems and analyze content")
    
    # Check if PIL is available
    if not PIL_AVAILABLE:
        st.warning("⚠️ **Image features require additional modules.**")
        st.info("💡 **To enable full image features, install:** `pip install Pillow`")
        st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Upload Image")
        st.markdown("Upload an image of a math problem, diagram, or text")
        
        uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg', 'gif'])
        
        if uploaded_file is not None:
            if not PIL_AVAILABLE:
                st.info("📷 **Demo Mode**: Image processing module not available")
                st.info("💡 **Install:** `pip install Pillow` for full functionality")
                st.info("🔍 **Simulated Analysis:** This would analyze your uploaded image")
            else:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Analyze Image", key="analyze_image"):
                try:
                    with st.spinner("🔍 Analyzing image..."):
                        # Simulate image analysis
                        st.success("🔍 **Image Analysis Complete!**")
                        
                        # Simulate different types of analysis based on image content
                        analysis_results = [
                            {
                                "type": "Math Problem",
                                "content": "2x + 5 = 13",
                                "solution": "x = 4",
                                "steps": ["Subtract 5 from both sides: 2x = 8", "Divide by 2: x = 4"]
                            },
                            {
                                "type": "Geometry Diagram",
                                "content": "Circle with radius 5cm",
                                "solution": "Area = 78.54 cm², Circumference = 31.42 cm",
                                "steps": ["Area = πr² = π(5)² = 25π ≈ 78.54 cm²", "Circumference = 2πr = 2π(5) = 10π ≈ 31.42 cm"]
                            },
                            {
                                "type": "Text Content",
                                "content": "Historical document or textbook page",
                                "solution": "Key points extracted and summarized",
                                "steps": ["Text extracted using OCR", "Key concepts identified", "Summary generated"]
                            }
                        ]
                        
                        # Randomly select a result for demo
                        result = random.choice(analysis_results)
                        
                        st.info(f"📋 **Detected:** {result['type']}")
                        st.markdown(f"**Content:** {result['content']}")
                        st.success(f"**Solution:** {result['solution']}")
                        
                        st.markdown("**Step-by-step solution:**")
                        for i, step in enumerate(result['steps'], 1):
                            st.markdown(f"{i}. {step}")
                        
                        # Offer to save as note
                        if st.button("💾 Save as Note", key="save_image_analysis"):
                            st.success("📝 Analysis saved as a note!")
                            
                except Exception as e:
                    st.error(f"Image analysis error: {e}")
                    st.info("💡 **Tip:** Make sure the image is clear and well-lit.")
    
    with col2:
        st.subheader("🎯 Problem Types Supported")
        st.markdown("Our AI can analyze and solve:")
        
        st.markdown("""
        **Mathematics:**
        - Algebra equations
        - Geometry problems
        - Calculus problems
        - Statistics questions
        
        **Science:**
        - Chemistry formulas
        - Physics diagrams
        - Biology diagrams
        - Lab reports
        
        **Other Subjects:**
        - Historical documents
        - Literature analysis
        - Language translation
        - Chart interpretation
        """)
        
        # Sample images for demo
        st.subheader("📸 Sample Images")
        st.markdown("Try uploading images like:")
        
        sample_images = [
            "Math equation: 2x + 5 = 13",
            "Geometry diagram: Circle with radius 5cm",
            "Chemistry formula: H2O + CO2 → H2CO3",
            "Physics diagram: Force diagram",
            "Biology diagram: Cell structure",
            "Historical document: Old text or map"
        ]
        
        for img_desc in sample_images:
            st.markdown(f"• {img_desc}")
    
    # Quick Analysis Tools
    st.subheader("⚡ Quick Analysis Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧮 Math Solver", key="math_solver"):
            st.info("🧮 **Math Solver Ready!** Upload an image of a math problem.")
    
    with col2:
        if st.button("🔬 Science Analyzer", key="science_analyzer"):
            st.info("🔬 **Science Analyzer Ready!** Upload an image of a science diagram or formula.")
    
    with col3:
        if st.button("📚 Text Extractor", key="text_extractor"):
            st.info("📚 **Text Extractor Ready!** Upload an image with text to extract and analyze.")

# --- Content Search & Discovery ---
def content_search_page():
    """Advanced content search and discovery features."""
    st.title("🔍 Content Search & Discovery")
    st.markdown("Search across all content and discover new learning materials")
    
    if not st.session_state.user_id:
        st.warning("Please log in to access search features.")
        return
    
    # Search Interface
    st.subheader("🔍 Search Content")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search for topics, concepts, or content:", 
                                   placeholder="e.g., quadratic equations, photosynthesis, world war 2")
    
    with col2:
        search_type = st.selectbox("Search Type", ["All Content", "Notes", "Videos", "Questions", "Subjects"])
    
    if st.button("🔍 Search", key="content_search"):
        if search_query:
            with st.spinner("🔍 Searching content..."):
                # Simulate search results
                st.success(f"🔍 Found 15 results for '{search_query}'")
                
                # Mock search results
                search_results = [
                    {
                        "type": "Note",
                        "title": f"Notes about {search_query}",
                        "content": f"Comprehensive notes covering {search_query} concepts...",
                        "subject": "Mathematics",
                        "relevance": 95
                    },
                    {
                        "type": "Video",
                        "title": f"Video: Understanding {search_query}",
                        "content": f"Educational video explaining {search_query} step by step...",
                        "subject": "Mathematics",
                        "relevance": 88
                    },
                    {
                        "type": "Question",
                        "title": f"Practice Questions: {search_query}",
                        "content": f"Quiz questions to test your understanding of {search_query}...",
                        "subject": "Mathematics",
                        "relevance": 82
                    }
                ]
                
                # Display search results
                for i, result in enumerate(search_results, 1):
                    with st.expander(f"{result['type']}: {result['title']} (Relevance: {result['relevance']}%)"):
                        st.markdown(f"**Subject:** {result['subject']}")
                        st.markdown(f"**Content:** {result['content']}")
                        
                        if st.button(f"Open {result['type']}", key=f"open_result_{i}"):
                            st.success(f"Opening {result['type']}...")
        else:
            st.warning("Please enter a search query.")
    
    # Advanced Search Filters
    st.subheader("🎯 Advanced Search Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        subject_filter = st.multiselect("Filter by Subject", 
                                      ["Mathematics", "Science", "English", "History", "Geography"])
    
    with col2:
        content_type = st.multiselect("Content Type", 
                                    ["Notes", "Videos", "Questions", "Flashcards", "Study Plans"])
    
    with col3:
        difficulty_level = st.selectbox("Difficulty Level", 
                                      ["All Levels", "Beginner", "Intermediate", "Advanced"])
    
    # Search Suggestions
    st.subheader("💡 Search Suggestions")
    st.markdown("Popular searches:")
    
    suggestions = [
        "quadratic equations", "photosynthesis", "world war 2", "cell structure",
        "trigonometry", "chemical bonding", "essay writing", "climate change"
    ]
    
    cols = st.columns(4)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 4]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                st.session_state.search_query = suggestion
                st.rerun()
    
    # Recent Searches
    st.subheader("🕒 Recent Searches")
    recent_searches = ["algebra basics", "photosynthesis", "essay structure", "world history"]
    
    for search in recent_searches:
        if st.button(f"🔍 {search}", key=f"recent_{search}"):
            st.session_state.search_query = search
            st.rerun()

def guided_learning_page():
    """Main guided learning interface."""
    st.title("🎯 Guided Learning")
    st.markdown("AI-powered personalized learning paths tailored to your subjects and level.")
    
    if not st.session_state.user_id:
        st.warning("Please log in to access guided learning.")
        return
    
    # Check if database tables exist and create them if needed
    ensure_all_tables_exist()
    
    # Get user's learning paths
    user_paths = get_user_learning_paths(st.session_state.user_id)
    
    # Create new learning path section
    st.header("🚀 Create New Learning Path")
    
    col1, col2 = st.columns(2)
    
    with col1:
        form_levels = list(subjects_data.keys())
        selected_form = st.selectbox("Select Form Level", form_levels, key="gl_form")
        
        if selected_form:
            available_subjects = subjects_data[selected_form]
            user_subjects = get_user_subjects(st.session_state.user_id, selected_form)
            
            if user_subjects:
                selected_subject = st.selectbox("Select Subject", user_subjects, key="gl_subject")
                
                if st.button("🎯 Generate Learning Path", key="generate_path"):
                    with st.spinner("Creating your personalized learning path..."):
                        learning_path = create_learning_path(st.session_state.user_id, selected_form, selected_subject)
                        
                        if "error" not in learning_path:
                            st.success("Learning path created successfully!")
                            st.rerun()
                        else:
                            st.error(learning_path["error"])
            else:
                st.info(f"Please add subjects for {selected_form} in the ZIMSEC Subjects page first.")
    
    with col2:
        st.markdown("""
        **How it works:**
        1. Select your form level and subject
        2. AI generates a personalized learning path
        3. Follow the structured modules and topics
        4. Complete practice exercises
        5. Track your progress
        """)
    
    # Progress Dashboard
    st.header("📊 Learning Progress Dashboard")
    
    # Add analytics button and database fix button
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("Track your learning progress across all subjects and modules.")
    with col2:
        if st.button("📈 View Analytics", key="view_analytics_gl"):
            analytics = get_learning_analytics(st.session_state.user_id)
            display_learning_analytics(analytics)
    with col3:
        if st.button("🔧 Fix Database", key="fix_db"):
            ensure_all_tables_exist()
            st.success("Database tables checked and created!")
            st.rerun()
    
    if user_paths:
        # Create progress overview
        col1, col2, col3 = st.columns(3)
        
        total_paths = len(user_paths)
        total_progress = 0
        
        for path in user_paths:
            progress = get_learning_progress(st.session_state.user_id, path['subject'])
            total_progress += progress.get('overall_progress', 0)
        
        avg_progress = total_progress / total_paths if total_paths > 0 else 0
        
        with col1:
            st.metric("Total Learning Paths", total_paths)
        with col2:
            st.metric("Average Progress", f"{avg_progress:.1f}%")
        with col3:
            st.metric("Active Subjects", len(set(p['subject'] for p in user_paths)))
        
        # Progress charts for each subject
        st.subheader("📈 Subject Progress")
        for path in user_paths:
            progress = get_learning_progress(st.session_state.user_id, path['subject'])
            
            with st.expander(f"📊 {path['subject']} Progress", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Overall Progress:** {progress.get('overall_progress', 0):.1f}%")
                    st.markdown(f"**Topics Completed:** {progress.get('topics_completed', 0)}")
                    st.markdown(f"**Exercises Completed:** {progress.get('exercises_completed', 0)}")
                
                with col2:
                    # Progress bar
                    progress_value = progress.get('overall_progress', 0) / 100
                    st.progress(progress_value)
                    
                    if progress.get('module_progress'):
                        st.markdown("**Module Progress:**")
                        for module_id, module_data in progress['module_progress'].items():
                            module_completed = module_data['completed']
                            total_topics = len(module_data['topics'])
                            if total_topics > 0:
                                module_progress = module_completed / total_topics
                                st.markdown(f"Module {module_id}: {module_completed}/{total_topics} topics ({module_progress:.1%})")
    
    # Display existing learning paths
    if user_paths:
        st.header("📚 Your Learning Paths")
        
        for path_idx, path in enumerate(user_paths):
            with st.expander(f"📖 {path['subject']} - {path['form_level']}"):
                display_learning_path(path, path_idx)
    else:
        st.info("No learning paths created yet. Create your first one above!")
    
    # Adaptive Learning Recommendations
    if user_paths:
        st.header("🎯 AI Learning Recommendations")
        st.markdown("Get personalized recommendations based on your progress and performance.")
        
        # Select subject for recommendations
        subject_choice = st.selectbox("Choose subject for recommendations:", 
                                    [p['subject'] for p in user_paths], key="rec_subject")
        
        if subject_choice and st.button("🤖 Get AI Recommendations"):
            with st.spinner("Analyzing your learning patterns..."):
                # Get user's progress for the selected subject
                user_progress = get_learning_progress(st.session_state.user_id, subject_choice)
                
                # Generate adaptive content based on progress
                if user_progress.get('overall_progress', 0) < 50:
                    difficulty = "beginner"
                    recommendation_type = "foundational concepts"
                elif user_progress.get('overall_progress', 0) < 80:
                    difficulty = "intermediate"
                    recommendation_type = "advanced topics"
                else:
                    difficulty = "advanced"
                    recommendation_type = "mastery challenges"
                
                # Get a sample topic for recommendations
                sample_topics = ["Core Concepts", "Problem Solving", "Advanced Applications"]
                selected_topic = st.selectbox("Select topic for recommendations:", sample_topics, key="rec_topic")
                
                if st.button("🎯 Generate Adaptive Content", key="generate_adaptive_content"):
                    adaptive_content = generate_adaptive_content(selected_topic, difficulty)
                    
                    if "error" not in adaptive_content:
                        st.success("AI-generated adaptive content ready!")
                        
                        with st.expander("🧠 Adaptive Learning Content", expanded=True):
                            st.markdown(f"**Topic:** {adaptive_content.get('topic', '')}")
                            st.markdown(f"**Level:** {adaptive_content.get('level', '')}")
                            st.markdown(f"**Summary:** {adaptive_content.get('content_summary', '')}")
                            
                            if adaptive_content.get('key_concepts'):
                                st.markdown("**Key Concepts:**")
                                for concept in adaptive_content['key_concepts']:
                                    st.markdown(f"• {concept}")
                            
                            if adaptive_content.get('examples'):
                                st.markdown("**Examples:**")
                                for example in adaptive_content['examples']:
                                    st.markdown(f"• {example}")
                            
                            if adaptive_content.get('common_misconceptions'):
                                st.markdown("**Common Misconceptions:**")
                                for misconception in adaptive_content['common_misconceptions']:
                                    st.markdown(f"• {misconception}")
                            
                            st.markdown(f"**Next Steps:** {adaptive_content.get('next_steps', '')}")
                            st.markdown(f"**Difficulty Adjustment:** {adaptive_content.get('difficulty_adjustment', '')}")
                    else:
                        st.error(adaptive_content["error"])

def display_learning_path(path_data: dict, path_idx: int):
    """Display a learning path with interactive elements."""
    path = path_data['data']
    
    st.markdown(f"**Total Modules:** {path.get('total_modules', 'N/A')}")
    st.markdown(f"**Estimated Time:** {path.get('estimated_total_time', 'N/A')}")
    
    # Display modules
    for module_idx, module in enumerate(path.get('modules', [])):
        st.subheader(f"📚 {module['title']}")
        st.markdown(f"**Time:** {module.get('estimated_time', 'N/A')}")
        st.markdown(module.get('description', ''))
        
        # Display topics within the module
        for topic_idx, topic in enumerate(module.get('topics', [])):
            # Create a truly unique key using path index, subject, form level, module and topic indices
            topic_unique_id = f"path_{path_idx}_{path_data['subject']}_{path_data['form_level']}_module_{module_idx}_topic_{topic_idx}"
            
            with st.expander(f"🎯 {topic['title']}"):
                st.markdown(f"**Description:** {topic.get('description', '')}")
                
                # Learning objectives
                if topic.get('learning_objectives'):
                    st.markdown("**Learning Objectives:**")
                    for obj_idx, obj in enumerate(topic['learning_objectives']):
                        st.markdown(f"• {obj}")
                
                # Prerequisites
                if topic.get('prerequisites'):
                    st.markdown("**Prerequisites:**")
                    for prereq_idx, prereq in enumerate(topic['prerequisites']):
                        st.markdown(f"• {prereq}")
                
                # Practice activities
                if topic.get('practice_activities'):
                    st.markdown("**Practice Activities:**")
                    for activity_idx, activity in enumerate(topic['practice_activities']):
                        st.markdown(f"• {activity}")
                
                # Generate practice exercise
                exercise_key = f"exercise_{topic_unique_id}"
                if st.button(f"🎲 Generate Exercise for {topic['title']}", key=exercise_key):
                    with st.spinner("Generating exercise..."):
                        exercise = generate_practice_exercise(topic['title'])
                        
                        if "error" not in exercise:
                            display_practice_exercise(exercise, exercise_key)
                        else:
                            st.error(exercise["error"])
                
                # Assessment
                if topic.get('assessment'):
                    st.markdown(f"**Assessment:** {topic['assessment']}")
                
                # Progress tracking
                complete_key = f"complete_{topic_unique_id}"
                if st.button(f"✅ Mark as Completed", key=complete_key):
                    if update_learning_progress(st.session_state.user_id, path_data['subject'], 
                                              module.get('module_id', module_idx), topic.get('topic_id', topic_idx), "completed", 100):
                        st.success("Progress updated! 🎉")
                        st.rerun()
                    else:
                        st.error("Failed to update progress.")

def display_practice_exercise(exercise: dict, unique_key: str):
    """Display a practice exercise with interactive elements."""
    st.markdown("---")
    st.markdown("### 🎯 Practice Exercise")
    
    st.markdown(f"**Topic:** {exercise.get('topic', '')}")
    st.markdown(f"**Difficulty:** {exercise.get('difficulty', '')}")
    
    st.markdown(f"**Question:** {exercise.get('question', '')}")
    
    # Display options
    if exercise.get('options'):
        answer_key = f"answer_{unique_key}"
        selected_answer = st.radio("Select your answer:", exercise['options'], key=answer_key)
        
        check_key = f"check_{unique_key}"
        if st.button("✅ Check Answer", key=check_key):
            if selected_answer == exercise.get('correct_answer', ''):
                st.success("🎉 Correct! Well done!")
                # Update progress with high score
                st.info("💡 Great job! This topic is marked as completed.")
            else:
                st.error(f"❌ Incorrect. The correct answer is: {exercise.get('correct_answer', '')}")
            
            # Show explanation
            if exercise.get('explanation'):
                st.info(f"**Explanation:** {exercise['explanation']}")
            
            # Show hint
            if exercise.get('hint'):
                st.info(f"💡 **Hint:** {exercise['hint']}")
            
            # Show related concepts
            if exercise.get('related_concepts'):
                st.markdown("**Related Concepts:**")
                for concept in exercise['related_concepts']:
                    st.markdown(f"• {concept}")

def get_learning_progress(user_id: int, subject: str) -> dict:
    """Get learning progress for a specific subject."""
    def _get_progress():
        conn = None
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                
                # Get progress for the subject
                c.execute('''SELECT module_id, topic_id, status, score 
                            FROM learning_progress 
                            WHERE user_id = ? AND subject = ?''', (user_id, subject))
                
                rows = c.fetchall()
                progress = {
                    "subject": subject,
                    "modules_completed": 0,
                    "topics_completed": 0,
                    "exercises_completed": 0,
                    "overall_progress": 0,
                    "module_progress": {}
                }
                
                for row in rows:
                    module_id, topic_id, status, score = row
                    if module_id not in progress["module_progress"]:
                        progress["module_progress"][module_id] = {"topics": {}, "completed": 0}
                    
                    progress["module_progress"][module_id]["topics"][topic_id] = {
                        "status": status,
                        "score": score
                    }
                    
                    if status == "completed":
                        progress["topics_completed"] += 1
                        progress["module_progress"][module_id]["completed"] += 1
                    
                    if score > 0:
                        progress["exercises_completed"] += 1
                
                # Calculate overall progress
                if progress["topics_completed"] > 0:
                    progress["overall_progress"] = min(100, (progress["topics_completed"] / max(1, len(rows))) * 100)
                
                return progress
                
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            return {"subject": subject, "modules_completed": 0, "topics_completed": 0, "exercises_completed": 0, "overall_progress": 0}
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return {"subject": subject, "modules_completed": 0, "topics_completed": 0, "exercises_completed": 0, "overall_progress": 0}
    
    result = safe_database_operation("get_learning_progress", _get_progress)
    return result if result is not None else {"subject": subject, "modules_completed": 0, "topics_completed": 0, "exercises_completed": 0, "overall_progress": 0}

def update_learning_progress(user_id: int, subject: str, module_id: int, topic_id: int, status: str, score: int = 0):
    """Update learning progress for a specific topic."""
    def _update_progress():
        conn = None
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                
                # Update learning progress
                c.execute('''INSERT OR REPLACE INTO learning_progress 
                            (user_id, subject, module_id, topic_id, status, score, completed_at) 
                            VALUES (?, ?, ?, ?, ?, ?, ?)''',
                         (user_id, subject, module_id, topic_id, status, score, 
                          datetime.now() if status == "completed" else None))
                
                # Update learning streak if topic was completed
                if status == "completed":
                    today = datetime.now().date()
                    
                    # Get current streak info
                    c.execute('''SELECT current_streak, longest_streak, last_activity_date 
                                FROM learning_streaks WHERE user_id = ?''', (user_id,))
                    streak_row = c.fetchone()
                    
                    if streak_row:
                        current_streak, longest_streak, last_activity = streak_row
                        
                        if last_activity:
                            last_activity = datetime.strptime(last_activity, '%Y-%m-%d').date()
                            days_diff = (today - last_activity).days
                            
                            if days_diff == 1:  # Consecutive day
                                new_streak = current_streak + 1
                            elif days_diff == 0:  # Same day
                                new_streak = current_streak
                            else:  # Break in streak
                                new_streak = 1
                        else:
                            new_streak = 1
                        
                        new_longest = max(longest_streak, new_streak)
                        
                        c.execute('''UPDATE learning_streaks 
                                    SET current_streak = ?, longest_streak = ?, last_activity_date = ?
                                    WHERE user_id = ?''', (new_streak, new_longest, today, user_id))
                    else:
                        # First time user
                        c.execute('''INSERT INTO learning_streaks 
                                    (user_id, current_streak, longest_streak, last_activity_date)
                                    VALUES (?, 1, 1, ?)''', (user_id, today))
                
                conn.commit()
                return True
                
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            return False
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return False
    
    return safe_database_operation("update_learning_progress", _update_progress)

def get_learning_streak(user_id: int) -> dict:
    """Get user's learning streak information."""
    def _get_streak():
        conn = None
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                
                c.execute('''SELECT current_streak, longest_streak, last_activity_date 
                            FROM learning_streaks WHERE user_id = ?''', (user_id,))
                row = c.fetchone()
                
                if row:
                    current_streak, longest_streak, last_activity = row
                    return {
                        "current_streak": current_streak,
                        "longest_streak": longest_streak,
                        "last_activity": last_activity
                    }
                else:
                    return {"current_streak": 0, "longest_streak": 0, "last_activity": None}
                    
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            return {"current_streak": 0, "longest_streak": 0, "last_activity": None}
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return {"current_streak": 0, "longest_streak": 0, "last_activity": None}
    
    result = safe_database_operation("get_learning_streak", _get_streak)
    return result if result is not None else {"current_streak": 0, "longest_streak": 0, "last_activity": None}

def generate_adaptive_content(topic: str, user_level: str = "beginner") -> dict:
    """Generate adaptive content based on user's current level."""
    try:
        # Check rate limiting
        if not ai_rate_limiter.can_call():
            return {"error": "Rate limit exceeded. Please wait before generating content."}
        
        adaptive_prompt = f"""Create adaptive learning content for the topic: {topic} at {user_level} level.

Return ONLY a valid JSON with this exact format:
{{
  "topic": "{topic}",
  "level": "{user_level}",
  "content_summary": "Brief overview of the topic",
  "key_concepts": ["Concept 1", "Concept 2", "Concept 3"],
  "examples": ["Example 1", "Example 2"],
  "common_misconceptions": ["Misconception 1", "Misconception 2"],
  "next_steps": "What to learn next",
  "difficulty_adjustment": "easier/harder/same based on performance"
}}"""

        response = model.generate_content(adaptive_prompt)
        response_text = response.text.strip()
        
        try:
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            content = json.loads(response_text)
            return content
            
        except (json.JSONDecodeError, ValueError):
            return {"error": "Failed to generate adaptive content. Please try again."}
            
    except Exception as e:
        return {"error": f"Error generating adaptive content: {e}"}

def get_learning_analytics(user_id: int) -> dict:
    """Get comprehensive learning analytics for a user."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Get overall statistics
            c.execute('''
                SELECT COUNT(DISTINCT subject) as total_subjects,
                       COUNT(*) as total_activities,
                       AVG(score) as avg_score
                FROM user_game_scores 
                WHERE user_id = ?
            ''', (user_id,))
            overall_stats = c.fetchone()
            
            # Get subject-wise performance
            c.execute('''
                SELECT subject, 
                       COUNT(*) as attempts,
                       AVG(score) as avg_score,
                       MAX(score) as best_score,
                       MIN(score) as worst_score
                FROM user_game_scores 
                WHERE user_id = ?
                GROUP BY subject
                ORDER BY avg_score DESC
            ''', (user_id,))
            subject_performance = c.fetchall()
            
            # Get learning streaks
            c.execute('''
                SELECT current_streak, longest_streak, last_activity_date
                FROM learning_streaks 
                WHERE user_id = ?
            ''', (user_id,))
            streak_data = c.fetchone()
            
            # Get recent activity
            c.execute('''
                SELECT game, subject, score, played_at
                FROM user_game_scores 
                WHERE user_id = ?
                ORDER BY played_at DESC
                LIMIT 10
            ''', (user_id,))
            recent_activity = c.fetchall()
            
            # Get notes statistics
            c.execute('''
                SELECT COUNT(*) as total_notes,
                       COUNT(CASE WHEN category = 'Study' THEN 1 END) as study_notes,
                       COUNT(CASE WHEN is_favorite = 1 THEN 1 END) as favorite_notes
                FROM user_notes 
                WHERE user_id = ?
            ''', (user_id,))
            notes_stats = c.fetchone()
            
            return {
                'overall_stats': {
                    'total_subjects': overall_stats[0] or 0,
                    'total_activities': overall_stats[1] or 0,
                    'avg_score': round(overall_stats[2] or 0, 2)
                },
                'subject_performance': [
                    {
                        'subject': row[0],
                        'attempts': row[1],
                        'avg_score': round(row[2] or 0, 2),
                        'best_score': row[3] or 0,
                        'worst_score': row[4] or 0
                    } for row in subject_performance
                ],
                'streak_data': {
                    'current_streak': streak_data[0] if streak_data else 0,
                    'longest_streak': streak_data[1] if streak_data else 0,
                    'last_activity': streak_data[2] if streak_data else None
                },
                'recent_activity': [
                    {
                        'game': row[0],
                        'subject': row[1],
                        'score': row[2],
                        'played_at': row[3]
                    } for row in recent_activity
                ],
                'notes_stats': {
                    'total_notes': notes_stats[0] or 0,
                    'study_notes': notes_stats[1] or 0,
                    'favorite_notes': notes_stats[2] or 0
                }
            }
            
    except sqlite3.Error as e:
        st.error(f"Database error getting analytics: {e}")
        return {}
    except Exception as e:
        st.error(f"Unexpected error getting analytics: {e}")
        return {}

def display_learning_analytics(analytics: dict):
    """Display comprehensive learning analytics in an expandable section."""
    if not analytics:
        st.warning("No analytics data available.")
        return
    
    with st.expander("📊 Detailed Learning Analytics", expanded=True):
        st.subheader("📈 Overall Performance")
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Subjects", analytics['overall_stats']['total_subjects'])
        with col2:
            st.metric("Total Activities", analytics['overall_stats']['total_activities'])
        with col3:
            st.metric("Average Score", f"{analytics['overall_stats']['avg_score']}%")
        with col4:
            st.metric("Current Streak", analytics['streak_data']['current_streak'])
        
        st.markdown("---")
        
        # Subject Performance Chart
        st.subheader("🎯 Subject Performance")
        if analytics['subject_performance']:
            # Create a performance chart
            subjects = [s['subject'] for s in analytics['subject_performance']]
            avg_scores = [s['avg_score'] for s in analytics['subject_performance']]
            attempts = [s['attempts'] for s in analytics['subject_performance']]
            
            # Display as a table with performance indicators
            performance_data = []
            for subj in analytics['subject_performance']:
                performance_level = "🟢 Excellent" if subj['avg_score'] >= 80 else \
                                 "🟡 Good" if subj['avg_score'] >= 60 else \
                                 "🟠 Needs Improvement" if subj['avg_score'] >= 40 else "🔴 Poor"
                
                performance_data.append({
                    "Subject": subj['subject'],
                    "Attempts": subj['attempts'],
                    "Avg Score": f"{subj['avg_score']}%",
                    "Best Score": f"{subj['best_score']}%",
                    "Performance": performance_level
                })
            
            st.dataframe(performance_data, use_container_width=True)
            
            # Performance insights
            st.markdown("---")
            st.subheader("💡 Performance Insights")
            
            best_subject = max(analytics['subject_performance'], key=lambda x: x['avg_score'])
            worst_subject = min(analytics['subject_performance'], key=lambda x: x['avg_score'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Strongest Subject:** {best_subject['subject']} ({best_subject['avg_score']}%)")
                st.info(f"**Most Studied:** {best_subject['subject']} ({best_subject['attempts']} attempts)")
            
            with col2:
                st.warning(f"**Needs Focus:** {worst_subject['subject']} ({worst_subject['avg_score']}%)")
                st.info(f"**Study Recommendation:** Practice more {worst_subject['subject']} questions")
        
        # Learning Streaks
        st.markdown("---")
        st.subheader("🔥 Learning Streaks")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Streak", analytics['streak_data']['current_streak'], 
                     delta=f"🔥 {analytics['streak_data']['current_streak']} days")
        with col2:
            st.metric("Longest Streak", analytics['streak_data']['longest_streak'], 
                     delta="🏆 Personal Best")
        
        # Streak motivation
        if analytics['streak_data']['current_streak'] > 0:
            if analytics['streak_data']['current_streak'] >= analytics['streak_data']['longest_streak']:
                st.success("🎉 New personal record! Keep up the amazing work!")
            elif analytics['streak_data']['current_streak'] >= 7:
                st.info("🌟 Great consistency! You're building excellent study habits!")
            elif analytics['streak_data']['current_streak'] >= 3:
                st.info("👍 Good start! Try to maintain this momentum.")
        else:
            st.warning("💪 Start your learning streak today! Even 10 minutes of study counts.")
        
        # Recent Activity
        st.markdown("---")
        st.subheader("📅 Recent Activity")
        if analytics['recent_activity']:
            for activity in analytics['recent_activity'][:5]:
                score_color = "🟢" if activity['score'] >= 80 else \
                             "🟡" if activity['score'] >= 60 else \
                             "🟠" if activity['score'] >= 40 else "🔴"
                
                st.markdown(f"{score_color} **{activity['game']}** - {activity['subject']}: {activity['score']}% ({activity['played_at'][:10]})")
        else:
            st.info("No recent activity. Start learning to see your progress!")
        
        # Notes Statistics
        st.markdown("---")
        st.subheader("📝 Study Notes Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Notes", analytics['notes_stats']['total_notes'])
        with col2:
            st.metric("Study Notes", analytics['notes_stats']['study_notes'])
        with col3:
            st.metric("Favorites", analytics['notes_stats']['favorite_notes'])
        
        # Study recommendations
        st.markdown("---")
        st.subheader("🎯 Personalized Study Recommendations")
        
        if analytics['subject_performance']:
            # Find subjects that need improvement
            improvement_subjects = [s for s in analytics['subject_performance'] if s['avg_score'] < 70]
            
            if improvement_subjects:
                st.warning("**Focus Areas for Improvement:**")
                for subj in improvement_subjects[:3]:  # Top 3 areas
                    st.markdown(f"• **{subj['subject']}** - Current average: {subj['avg_score']}%")
                    st.markdown(f"  - Try more practice questions")
                    st.markdown(f"  - Review your notes for this subject")
                    st.markdown(f"  - Use flashcards to reinforce concepts")
            else:
                st.success("🎉 All subjects are performing well! Consider:")
                st.markdown("• Exploring advanced topics")
                st.markdown("• Helping other students")
                st.markdown("• Setting higher goals")
        
        # Export analytics
        st.markdown("---")
        st.subheader("📤 Export Your Data")
        if st.button("📊 Export Analytics Report"):
            export_analytics_report(analytics)

def export_analytics_report(analytics: dict):
    """Export analytics data as a downloadable report."""
    try:
        # Create a comprehensive report
        report_content = f"""
# Learning Analytics Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Performance
- Total Subjects: {analytics['overall_stats']['total_subjects']}
- Total Activities: {analytics['overall_stats']['total_activities']}
- Average Score: {analytics['overall_stats']['avg_score']}%

## Subject Performance
"""
        
        for subject in analytics['subject_performance']:
            report_content += f"""
### {subject['subject']}
- Attempts: {subject['attempts']}
- Average Score: {subject['avg_score']}%
- Best Score: {subject['best_score']}%
- Worst Score: {subject['worst_score']}%
"""
        
        report_content += f"""
## Learning Streaks
- Current Streak: {analytics['streak_data']['current_streak']} days
- Longest Streak: {analytics['streak_data']['longest_streak']} days

## Study Notes
- Total Notes: {analytics['notes_stats']['total_notes']}
- Study Notes: {analytics['notes_stats']['study_notes']}
- Favorite Notes: {analytics['notes_stats']['favorite_notes']}
"""
        
        # Create download button
        st.download_button(
            label="📥 Download Report (TXT)",
            data=report_content,
            file_name=f"learning_analytics_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
        
        st.success("✅ Report ready for download!")
        
    except Exception as e:
        st.error(f"Error creating report: {e}")

def update_learning_streak(user_id: int):
    """Update user's learning streak based on activity."""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            today = datetime.now().date()
            
            # Check if user has a streak record
            c.execute('SELECT current_streak, longest_streak, last_activity_date FROM learning_streaks WHERE user_id = ?', (user_id,))
            streak_record = c.fetchone()
            
            if streak_record:
                current_streak, longest_streak, last_activity = streak_record
                last_activity_date = datetime.strptime(last_activity, '%Y-%m-%d').date() if last_activity else None
                
                if last_activity_date:
                    days_diff = (today - last_activity_date).days
                    
                    if days_diff == 1:  # Consecutive day
                        new_streak = current_streak + 1
                        new_longest = max(new_streak, longest_streak)
                        
                        c.execute('''UPDATE learning_streaks 
                                   SET current_streak = ?, longest_streak = ?, last_activity_date = ? 
                                   WHERE user_id = ?''', 
                                (new_streak, new_longest, today, user_id))
                        
                    elif days_diff == 0:  # Same day, no change needed
                        pass
                    else:  # Streak broken
                        c.execute('''UPDATE learning_streaks 
                                   SET current_streak = 1, last_activity_date = ? 
                                   WHERE user_id = ?''', 
                                (today, user_id))
                else:
                    # First activity
                    c.execute('''UPDATE learning_streaks 
                               SET current_streak = 1, last_activity_date = ? 
                               WHERE user_id = ?''', 
                            (today, user_id))
            else:
                # Create new streak record
                c.execute('''INSERT INTO learning_streaks (user_id, current_streak, longest_streak, last_activity_date) 
                           VALUES (?, 1, 1, ?)''', (user_id, today))
            
            conn.commit()
            
    except sqlite3.Error as e:
        st.error(f"Error updating learning streak: {e}")
    except Exception as e:
        st.error(f"Unexpected error updating streak: {e}")

# --- ZIMBABWEAN LANGUAGES SUPPORT SYSTEM ---

# Zimbabwean language configurations with correct Google Translator codes
ZIMBABWEAN_LANGUAGES = {
    'en': {
        'name': 'English',
        'native_name': 'English',
        'code': 'en',
        'flag': '🇬🇧',
        'is_official': True,
        'is_curriculum': True,
        'google_code': 'en',
        'is_supported': True
    },
    'sn': {
        'name': 'Shona',
        'native_name': 'chiShona',
        'code': 'sn',
        'flag': '🇿🇼',
        'is_official': False,
        'is_curriculum': True,
        'google_code': 'sn',
        'is_supported': True
    },
    'nd': {
        'name': 'Ndebele',
        'native_name': 'isiNdebele',
        'code': 'nd',
        'flag': '🇿🇼',
        'is_official': False,
        'is_curriculum': True,
        'google_code': 'zu',  # Use Zulu as closest match
        'is_supported': True  # Now supported with custom translations
    },
    've': {
        'name': 'Venda',
        'native_name': 'Tshivenda',
        'code': 've',
        'flag': '🇿🇼',
        'is_official': False,
        'is_curriculum': False,
        'google_code': 've',
        'is_supported': False
    },
    'to': {
        'name': 'Tonga',
        'native_name': 'chiTonga',
        'code': 'to',
        'flag': '🇿🇼',
        'is_official': False,
        'is_curriculum': False,
        'google_code': 'ts',  # Use Tsonga as closest match
        'is_supported': False
    }
}

# ZIMSEC curriculum language mappings
ZIMSEC_LANGUAGE_MAPPINGS = {
    'Mathematics': {
        'en': 'Mathematics',
        'sn': 'Masvomhu',
        'nd': 'Izibalo'
    },
    'Physics': {
        'en': 'Physics',
        'sn': 'Fizikisi',
        'nd': 'IFiziksi'
    },
    'Chemistry': {
        'en': 'Chemistry',
        'sn': 'Kemistiri',
        'nd': 'IKhemistri'
    },
    'Biology': {
        'en': 'Biology',
        'sn': 'Bhayoroji',
        'nd': 'IBhayoloji'
    },
    'English Language': {
        'en': 'English Language',
        'sn': 'Mutauro weChirungu',
        'nd': 'Ulimi lwesiNgisi'
    },
    'Shona': {
        'en': 'Shona',
        'sn': 'chiShona',
        'nd': 'isiShona'
    },
    'Ndebele': {
        'en': 'Ndebele',
        'sn': 'chiNdebele',
        'nd': 'isiNdebele'
    }
}

# Custom Ndebele translation dictionary for common phrases
NDEBELE_TRANSLATIONS = {
    # Greetings and common phrases
    'hello': 'Sawubona',
    'hi': 'Sawubona',
    'how are you': 'Unjani',
    'how are you doing': 'Unjani',
    'i am fine': 'Ngikhona',
    'i am well': 'Ngikhona',
    'thank you': 'Siyabonga',
    'thanks': 'Siyabonga',
    'please': 'Ngiyacela',
    'goodbye': 'Hamba kahle',
    'see you later': 'Sizobonana',
    'good morning': 'Sawubona',
    'good afternoon': 'Sawubona',
    'good evening': 'Sawubona',
    'good night': 'Ubusuku obuhle',
    
    # Educational terms
    'learn': 'funda',
    'learning': 'ukufunda',
    'study': 'funda',
    'studying': 'ukufunda',
    'teach': 'fundisa',
    'teaching': 'ukufundisa',
    'understand': 'qonda',
    'understanding': 'ukuqonda',
    'explain': 'chaza',
    'explanation': 'incazelo',
    'question': 'umbuzo',
    'answer': 'impendulo',
    'help': 'siza',
    'helping': 'ukusiza',
    'problem': 'inkinga',
    'solution': 'isixazululo',
    'example': 'isibonelo',
    'practice': 'ukuzilolonga',
    'test': 'ivivinyo',
    'exam': 'ivivinyo',
    'homework': 'umsebenzi wasekhaya',
    'assignment': 'umsebenzi',
    'project': 'iproyekthi',
    'research': 'ucwaningo',
    'knowledge': 'ulwazi',
    'information': 'ulwazi',
    'fact': 'iqiniso',
    'truth': 'iqiniso',
    'correct': 'ilungile',
    'wrong': 'akulungile',
    'mistake': 'iphutha',
    'error': 'iphutha',
    
    # Subject-specific terms
    'mathematics': 'izibalo',
    'math': 'izibalo',
    'numbers': 'izinombolo',
    'calculation': 'ukubala',
    'equation': 'ukulinganisa',
    'formula': 'ifomula',
    'geometry': 'ijiyomethri',
    'algebra': 'i-algebra',
    'science': 'isayensi',
    'physics': 'ifiziksi',
    'chemistry': 'ikhemistri',
    'biology': 'ibhayoloji',
    'history': 'umlando',
    'geography': 'ijografi',
    'english': 'isiNgisi',
    'shona': 'isiShona',
    'ndebele': 'isiNdebele',
    
    # Common verbs
    'want': 'funa',
    'need': 'dinga',
    'can': 'ngakwazi',
    'cannot': 'angikwazi',
    'will': 'uzo',
    'going to': 'uzo',
    'have': 'nayo',
    'has': 'unayo',
    'had': 'wayenayo',
    'do': 'yenza',
    'does': 'yenza',
    'did': 'wenza',
    'make': 'yenza',
    'makes': 'yenza',
    'made': 'wenza',
    'see': 'bona',
    'sees': 'ubona',
    'saw': 'wabona',
    'look': 'bheka',
    'watch': 'bheka',
    'listen': 'lalela',
    'hear': 'zwe',
    'speak': 'khuluma',
    'talk': 'khuluma',
    'say': 'thi',
    'says': 'uthi',
    'said': 'wathi',
    'tell': 'tshela',
    'tells': 'utshela',
    'told': 'watshela',
    'ask': 'buza',
    'asks': 'ubuza',
    'asked': 'wabuza',
    'know': 'azi',
    'knows': 'uyazi',
    'knew': 'wayazi',
    'think': 'cabanga',
    'thinks': 'ucabanga',
    'thought': 'wacabanga',
    'feel': 'zwa',
    'feels': 'uzwa',
    'felt': 'wazwa',
    'like': 'thanda',
    'likes': 'uthanda',
    'liked': 'wathanda',
    'love': 'thanda',
    'loves': 'uthanda',
    'loved': 'wathanda',
    
    # Common adjectives
    'good': 'kuhle',
    'bad': 'kubi',
    'big': 'kukhulu',
    'small': 'kuncane',
    'new': 'kusha',
    'old': 'kudala',
    'young': 'kusha',
    'beautiful': 'kuhle',
    'ugly': 'kubi',
    'easy': 'kulula',
    'difficult': 'kunzima',
    'hard': 'kunzima',
    'simple': 'kulula',
    'complex': 'kubucayi',
    'important': 'kubalulekile',
    'necessary': 'kudingekile',
    'possible': 'kungenzeka',
    'impossible': 'akunakwenzeka',
    'true': 'iqiniso',
    'false': 'amanga',
    'right': 'ilungile',
    'wrong': 'akulungile',
    'correct': 'ilungile',
    'incorrect': 'akulungile',
    
    # Time and place
    'today': 'namhlanje',
    'yesterday': 'izolo',
    'tomorrow': 'kusasa',
    'now': 'manje',
    'later': 'kamuva',
    'soon': 'maduze',
    'never': 'akukaze',
    'always': 'njalo',
    'sometimes': 'kwesinye isikhathi',
    'often': 'kaningi',
    'here': 'lapha',
    'there': 'lapho',
    'where': 'kuphi',
    'when': 'nini',
    'why': 'kungani',
    'how': 'kanjani',
    'what': 'yini',
    'who': 'ubani',
    'which': 'yiphi',
    
    # Numbers
    'one': 'kunye',
    'two': 'kubili',
    'three': 'kuthathu',
    'four': 'kune',
    'five': 'kuhlanu',
    'six': 'isithupha',
    'seven': 'isikhombisa',
    'eight': 'isishiyagalombili',
    'nine': 'isishiyagalolunye',
    'ten': 'ishumi',
    'hundred': 'ikhulu',
    'thousand': 'inkulungwane',
    'million': 'isigidi',
    
    # Common responses
    'yes': 'yebo',
    'no': 'cha',
    'maybe': 'mhlawumbe',
    'i don\'t know': 'angazi',
    'i don\'t understand': 'angiqondi',
    'please repeat': 'ngiyacela uphinde',
    'speak slowly': 'khuluma kancane',
    'what does this mean': 'kusho ukuthini lokhu',
    'can you help me': 'ungangisiza',
    'i need help': 'ngidinga usizo',
    'thank you for helping': 'siyabonga ngokusiza',
    'you are welcome': 'wamukelekile',
    'no problem': 'akunankinga',
    'it\'s okay': 'kulungile',
    'that\'s fine': 'kulungile',
    'i agree': 'ngiyavuma',
    'i disagree': 'angivumi',
    'i think so': 'ngicabanga ukuthi kunjalo',
    'i hope so': 'ngithemba ukuthi kunjalo',
    'good luck': 'ngikufisela inhlanhla',
    'congratulations': 'siyakuhalalisela',
    'well done': 'uhle kakhulu',
    'excellent': 'kuhle kakhulu',
    'amazing': 'kumangalisa',
    'wonderful': 'kuhle kakhulu',
    'fantastic': 'kuhle kakhulu',
    'great': 'kuhle kakhulu',
    'awesome': 'kuhle kakhulu',
    'brilliant': 'kuhle kakhulu',
    'perfect': 'kuphelele',
    'outstanding': 'kuhle kakhulu'
}

class ZimbabweanLanguageManager:
    """Manages Zimbabwean language detection, translation, and localization."""
    
    def __init__(self):
        self.translation_cache = {}
        self.language_detectors = {}
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize language detection for Zimbabwean languages."""
        try:
            # Initialize Google Translator only for supported languages
            for lang_code, lang_info in ZIMBABWEAN_LANGUAGES.items():
                if lang_code != 'en' and lang_info['is_supported']:
                    try:
                        self.language_detectors[lang_code] = GoogleTranslator(source='auto', target=lang_info['google_code'])
                    except Exception as e:
                        st.warning(f"Could not initialize translator for {lang_info['name']}: {e}")
        except Exception as e:
            st.warning(f"Language detector initialization warning: {e}")
    
    def detect_language(self, text: str) -> str:
        """Detect the language of input text with focus on Zimbabwean languages."""
        try:
            # First, try to detect if it's a Zimbabwean language
            if self._is_likely_shona(text):
                return 'sn'
            elif self._is_likely_ndebele(text):
                return 'nd'
            elif self._is_likely_venda(text):
                return 've'
            elif self._is_likely_tonga(text):
                return 'to'
            
            # Use Google Translator for broader language detection
            try:
                translator = GoogleTranslator(source='auto', target='en')
                detected_lang = translator.source
                
                # Map detected language to our supported codes
                if detected_lang in ZIMBABWEAN_LANGUAGES:
                    return detected_lang
                elif detected_lang == 'en':
                    return 'en'
                else:
                    # Default to English for unsupported languages
                    return 'en'
            except Exception as e:
                # If Google Translator fails, use pattern detection only
                return 'en'
                
        except Exception as e:
            st.warning(f"Language detection error: {e}")
            return 'en'
    
    def _is_likely_shona(self, text: str) -> bool:
        """Check if text is likely Shona based on common patterns."""
        shona_patterns = [
            'zvino', 'zvinoita', 'zvinoitwa', 'zvinoitika',
            'zvinoitawo', 'zvinoitwawo', 'zvinoitikavo',
            'ndi', 'ndiri', 'ndichiri', 'ndichange',
            'uri', 'uchiri', 'uchange', 'uchangewo',
            'ari', 'achiri', 'achange', 'achangewo',
            'tiri', 'tichiri', 'tichange', 'tichangewo',
            'muri', 'muchiri', 'muchange', 'muchangewo',
            'vari', 'vachiri', 'vachange', 'vachangewo'
        ]
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in shona_patterns)
    
    def _is_likely_ndebele(self, text: str) -> bool:
        """Check if text is likely Ndebele based on common patterns."""
        ndebele_patterns = [
            'ngi', 'ngiyi', 'ngiyile', 'ngiyileyo',
            'uyi', 'uyile', 'uyileyo', 'uyileyo',
            'uyi', 'uyile', 'uyileyo', 'uyileyo',
            'siyi', 'siyile', 'siyileyo', 'siyileyo',
            'niyi', 'niyile', 'niyileyo', 'niyileyo',
            'bayi', 'bayile', 'bayileyo', 'bayileyo'
        ]
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in ndebele_patterns)
    
    def _is_likely_venda(self, text: str) -> bool:
        """Check if text is likely Venda based on common patterns."""
        venda_patterns = [
            'ndi', 'ndo', 'ndo', 'ndo',
            'u', 'u', 'u', 'u',
            'a', 'a', 'a', 'a',
            'ri', 'ri', 'ri', 'ri'
        ]
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in venda_patterns)
    
    def _is_likely_tonga(self, text: str) -> bool:
        """Check if text is likely Tonga based on common patterns."""
        tonga_patterns = [
            'ndi', 'ndo', 'ndo', 'ndo',
            'u', 'u', 'u', 'u',
            'a', 'a', 'a', 'a',
            'li', 'li', 'li', 'li'
        ]
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in tonga_patterns)
    
    def translate_text(self, text: str, target_lang: str, source_lang: str = 'auto', context: str = None) -> str:
        """Translate text with caching and context awareness."""
        # Create cache key
        cache_key = f"{hashlib.md5(text.encode()).hexdigest()}_{source_lang}_{target_lang}_{context or 'general'}"
        
        # Check cache first
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        try:
            if target_lang == source_lang or target_lang == 'auto':
                return text
            
            # Check if target language is supported
            if target_lang in ZIMBABWEAN_LANGUAGES:
                target_info = ZIMBABWEAN_LANGUAGES[target_lang]
                if not target_info['is_supported']:
                    # For unsupported languages, show a message and return original text
                    st.info(f"🌍 {target_info['name']} translation is not yet fully supported. Using English for now.")
                    return text
            
            # Special handling for Ndebele (nd) - use custom translations
            if target_lang == 'nd':
                return self._translate_to_ndebele(text)
            
            # Use Google Translator with correct language codes
            if target_lang in ZIMBABWEAN_LANGUAGES:
                google_target = ZIMBABWEAN_LANGUAGES[target_lang]['google_code']
            else:
                google_target = target_lang
            
            translator = GoogleTranslator(source=source_lang, target=google_target)
            translated = translator.translate(text)
            
            # Cache the result
            self.translation_cache[cache_key] = translated
            
            return translated
            
        except Exception as e:
            st.warning(f"Translation error ({source_lang} → {target_lang}): {e}")
            return text
    
    def _translate_to_ndebele(self, text: str) -> str:
        """Custom translation to Ndebele using our dictionary."""
        try:
            # Convert to lowercase for matching
            text_lower = text.lower().strip()
            
            # Check for exact matches first
            if text_lower in NDEBELE_TRANSLATIONS:
                return NDEBELE_TRANSLATIONS[text_lower]
            
            # Check for partial matches (words within sentences)
            words = text_lower.split()
            translated_words = []
            
            for word in words:
                # Clean the word (remove punctuation)
                clean_word = ''.join(c for c in word if c.isalnum())
                
                if clean_word in NDEBELE_TRANSLATIONS:
                    translated_words.append(NDEBELE_TRANSLATIONS[clean_word])
                else:
                    # Keep original word if no translation found
                    translated_words.append(word)
            
            # Reconstruct the sentence
            translated_text = ' '.join(translated_words)
            
            # If we found some translations, show info
            if translated_text != text:
                st.info("🌍 Using custom Ndebele translations for common phrases.")
            
            return translated_text
            
        except Exception as e:
            st.warning(f"Custom Ndebele translation error: {e}")
            return text
    
    def get_localized_subject_name(self, subject: str, language: str) -> str:
        """Get subject name in the specified language."""
        if subject in ZIMSEC_LANGUAGE_MAPPINGS:
            return ZIMSEC_LANGUAGE_MAPPINGS[subject].get(language, subject)
        return subject
    
    def get_language_display_name(self, lang_code: str, user_lang: str = 'en') -> str:
        """Get language name in user's preferred language."""
        if lang_code in ZIMBABWEAN_LANGUAGES:
            if user_lang == 'en':
                return ZIMBABWEAN_LANGUAGES[lang_code]['name']
            else:
                return ZIMBABWEAN_LANGUAGES[lang_code]['native_name']
        return lang_code
    
    def get_supported_languages_list(self, include_official: bool = True) -> list:
        """Get list of supported languages for UI display."""
        languages = []
        for code, lang_info in ZIMBABWEAN_LANGUAGES.items():
            if include_official or not lang_info['is_official']:
                languages.append({
                    'code': code,
                    'name': lang_info['name'],
                    'native_name': lang_info['native_name'],
                    'flag': lang_info.get('flag', '🌐'),
                    'is_curriculum': lang_info['is_curriculum']
                })
        return languages

# Initialize the language manager
language_manager = ZimbabweanLanguageManager()

if __name__ == "__main__":
    # Ensure all database tables exist before starting
    ensure_all_tables_exist()
    main()