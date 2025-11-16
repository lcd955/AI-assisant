"""
Dynamic User Financial Profile System
Integrates multi-source data for personalized recommendations
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from models.schemas import UserProfile, UserSegment, RiskLevel, Transaction
from utils.logger import setup_logger
from .gnn_model import UserGraphBuilder

logger = setup_logger()


class UserProfileManager:
    """
    Manages dynamic user financial profiles with multi-source data integration
    """
    
    def __init__(self):
        """Initialize user profile manager"""
        self.profiles: Dict[str, UserProfile] = {}
        self.graph_builder = UserGraphBuilder()
        logger.info("UserProfileManager initialized")
    
    def create_profile(
        self,
        user_id: str,
        age: int,
        monthly_income: float,
        monthly_expenses: float,
        savings: float,
        risk_level: RiskLevel = RiskLevel.MODERATE
    ) -> UserProfile:
        """
        Create a new user profile
        
        Args:
            user_id: User identifier
            age: User age
            monthly_income: Monthly income
            monthly_expenses: Monthly expenses
            savings: Current savings
            risk_level: Risk tolerance level
            
        Returns:
            Created UserProfile
        """
        # Determine user segment based on age and financial metrics
        segment = self._determine_segment(age, monthly_income, savings)
        
        profile = UserProfile(
            user_id=user_id,
            segment=segment,
            risk_level=risk_level,
            age=age,
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            savings=savings
        )
        
        self.profiles[user_id] = profile
        
        # Add to graph
        self.graph_builder.add_user(user_id, {
            "segment": segment.value,
            "risk_level": risk_level.value,
            "age": age
        })
        
        logger.info(f"Created profile for user {user_id}, segment: {segment}")
        return profile
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID"""
        return self.profiles.get(user_id)
    
    def update_profile(
        self,
        user_id: str,
        updates: Dict[str, Any]
    ) -> Optional[UserProfile]:
        """
        Update user profile with new data
        
        Args:
            user_id: User identifier
            updates: Dictionary of fields to update
            
        Returns:
            Updated UserProfile or None if user not found
        """
        profile = self.profiles.get(user_id)
        if not profile:
            logger.warning(f"Profile not found for user {user_id}")
            return None
        
        # Update fields
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        
        profile.updated_at = datetime.now()
        
        # Re-evaluate segment if relevant fields changed
        if any(k in updates for k in ['age', 'monthly_income', 'savings']):
            profile.segment = self._determine_segment(
                profile.age,
                profile.monthly_income,
                profile.savings
            )
        
        logger.info(f"Updated profile for user {user_id}")
        return profile
    
    def add_transaction(
        self,
        user_id: str,
        transaction: Transaction
    ) -> bool:
        """
        Add transaction to user profile
        
        Args:
            user_id: User identifier
            transaction: Transaction record
            
        Returns:
            True if successful
        """
        profile = self.profiles.get(user_id)
        if not profile:
            logger.warning(f"Profile not found for user {user_id}")
            return False
        
        profile.transaction_history.append(transaction)
        
        # Update behavior patterns
        self._update_behavior_patterns(profile, transaction)
        
        logger.debug(f"Added transaction for user {user_id}")
        return True
    
    def analyze_spending_patterns(
        self,
        user_id: str,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze user spending patterns over a period
        
        Args:
            user_id: User identifier
            period_days: Number of days to analyze
            
        Returns:
            Spending analysis
        """
        profile = self.profiles.get(user_id)
        if not profile:
            return {}
        
        # Filter transactions by period
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_transactions = [
            t for t in profile.transaction_history
            if t.timestamp >= cutoff_date
        ]
        
        if not recent_transactions:
            return {"total_spending": 0, "categories": {}}
        
        # Analyze by category
        category_spending = {}
        total_spending = 0
        
        for transaction in recent_transactions:
            category = transaction.category
            amount = abs(transaction.amount)  # Use absolute value for expenses
            
            if category not in category_spending:
                category_spending[category] = 0
            category_spending[category] += amount
            total_spending += amount
        
        # Calculate percentages
        category_percentages = {
            cat: (amount / total_spending * 100) if total_spending > 0 else 0
            for cat, amount in category_spending.items()
        }
        
        # Identify trends
        is_overspending = total_spending > profile.monthly_expenses
        savings_rate = (profile.monthly_income - total_spending) / profile.monthly_income if profile.monthly_income > 0 else 0
        
        return {
            "total_spending": total_spending,
            "categories": category_spending,
            "category_percentages": category_percentages,
            "is_overspending": is_overspending,
            "savings_rate": savings_rate,
            "period_days": period_days
        }
    
    def identify_needs(self, user_id: str) -> List[str]:
        """
        Identify potential financial needs based on profile and behavior
        
        Args:
            user_id: User identifier
            
        Returns:
            List of identified needs
        """
        profile = self.profiles.get(user_id)
        if not profile:
            return []
        
        needs = []
        
        # Analyze based on segment
        if profile.segment == UserSegment.YOUNG_PROFESSIONAL:
            # Check for low savings
            if profile.savings < profile.monthly_income * 3:
                needs.append("emergency_fund")
            
            # Check for spending patterns
            spending = self.analyze_spending_patterns(user_id)
            if spending.get("savings_rate", 0) < 0.1:
                needs.append("auto_savings_plan")
            
            needs.append("low_threshold_investing")
        
        elif profile.segment == UserSegment.MIDDLE_AGED_FAMILY:
            # Education planning
            if any(goal.get("type") == "education" for goal in profile.financial_goals):
                needs.append("education_insurance")
            
            needs.extend(["stable_funds", "family_insurance"])
        
        elif profile.segment == UserSegment.RETIRED:
            needs.extend(["principal_safety", "high_dividend_stocks", "healthcare_insurance"])
        
        # Risk-based needs
        if profile.risk_level == RiskLevel.CONSERVATIVE:
            needs.append("low_risk_products")
        elif profile.risk_level == RiskLevel.AGGRESSIVE:
            needs.append("high_growth_products")
        
        logger.debug(f"Identified needs for user {user_id}: {needs}")
        return list(set(needs))  # Remove duplicates
    
    def _determine_segment(
        self,
        age: int,
        monthly_income: float,
        savings: float
    ) -> UserSegment:
        """Determine user segment based on demographics and financials"""
        
        if age < 30:
            if monthly_income < 10000 or savings < 50000:
                return UserSegment.YOUNG_PROFESSIONAL
            return UserSegment.YOUNG_PROFESSIONAL
        elif age < 50:
            return UserSegment.MIDDLE_AGED_FAMILY
        else:
            return UserSegment.RETIRED
    
    def _update_behavior_patterns(
        self,
        profile: UserProfile,
        transaction: Transaction
    ):
        """Update behavior patterns based on new transaction"""
        
        if "spending_by_category" not in profile.behavior_patterns:
            profile.behavior_patterns["spending_by_category"] = {}
        
        category = transaction.category
        amount = abs(transaction.amount)
        
        if category not in profile.behavior_patterns["spending_by_category"]:
            profile.behavior_patterns["spending_by_category"][category] = []
        
        profile.behavior_patterns["spending_by_category"][category].append({
            "amount": amount,
            "timestamp": transaction.timestamp.isoformat()
        })
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the user-product-scene graph"""
        return self.graph_builder.get_statistics()
