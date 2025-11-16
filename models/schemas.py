"""
Data models for the AI Financial Recommendation System
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class UserSegment(str, Enum):
    """User segment types"""
    YOUNG_PROFESSIONAL = "young_professional"
    MIDDLE_AGED_FAMILY = "middle_aged_family"
    RETIRED = "retired"
    STUDENT = "student"
    ENTREPRENEUR = "entrepreneur"


class RiskLevel(str, Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class Transaction(BaseModel):
    """Transaction record"""
    id: str
    user_id: str
    amount: float
    category: str
    description: str
    timestamp: datetime
    merchant: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class UserProfile(BaseModel):
    """User financial profile"""
    user_id: str
    segment: UserSegment
    risk_level: RiskLevel
    age: int
    monthly_income: float
    monthly_expenses: float
    savings: float
    investment_portfolio: Dict[str, float] = Field(default_factory=dict)
    financial_goals: List[Dict[str, Any]] = Field(default_factory=list)
    transaction_history: List[Transaction] = Field(default_factory=list)
    behavior_patterns: Dict[str, Any] = Field(default_factory=dict)
    credit_score: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class DialogueContext(BaseModel):
    """Dialogue context for maintaining conversation state"""
    user_id: str
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list)
    current_intent: Optional[str] = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    context_variables: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class RecommendationRequest(BaseModel):
    """Recommendation request"""
    user_id: str
    context: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)


class RecommendationItem(BaseModel):
    """Individual recommendation item"""
    product_id: str
    product_name: str
    product_type: str
    score: float
    reasoning: List[str]
    risk_level: RiskLevel
    expected_return: Optional[float] = None
    min_investment: Optional[float] = None
    features: Dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    """Recommendation response with explanations"""
    user_id: str
    recommendations: List[RecommendationItem]
    explanation: str
    confidence_score: float
    timestamp: datetime = Field(default_factory=datetime.now)
    strategy_used: str


class QueryRequest(BaseModel):
    """Natural language query request"""
    user_id: str
    query: str
    session_id: Optional[str] = None
    use_voice: bool = False


class QueryResponse(BaseModel):
    """Query response"""
    response: str
    intent: str
    entities: Dict[str, Any]
    suggestions: List[str] = Field(default_factory=list)
    session_id: str
