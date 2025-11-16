"""
Personalized Recommendation Engine
Integrates user profiling, GNN, and RL for intelligent product recommendations
"""
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
from models.schemas import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationItem,
    UserSegment,
    RiskLevel
)
from utils.logger import setup_logger
from utils.config_loader import config

logger = setup_logger()


class PersonalizedRecommendationEngine:
    """
    Main recommendation engine that combines multiple strategies
    """
    
    def __init__(
        self,
        user_profile_manager,
        rl_engine: Optional[Any] = None
    ):
        """
        Initialize recommendation engine
        
        Args:
            user_profile_manager: UserProfileManager instance
            rl_engine: RLRecommendationEngine instance (optional)
        """
        self.user_profile_manager = user_profile_manager
        self.rl_engine = rl_engine
        
        # Product catalog (in production, load from database)
        self.product_catalog = self._initialize_product_catalog()
        
        # Strategy configuration
        self.strategies = config.get("recommendation.strategies", {})
        
        logger.info("PersonalizedRecommendationEngine initialized")
    
    def _initialize_product_catalog(self) -> List[Dict[str, Any]]:
        """Initialize financial product catalog"""
        return [
            # Savings products
            {
                "product_id": "SAVE_001",
                "product_name": "智能自动储蓄计划",
                "product_type": "savings",
                "risk_level": RiskLevel.CONSERVATIVE,
                "min_investment": 100.0,
                "expected_return": 0.03,
                "features": {
                    "auto_transfer": True,
                    "flexible_withdrawal": True,
                    "min_amount": 100
                },
                "target_segments": [UserSegment.YOUNG_PROFESSIONAL]
            },
            # Investment funds
            {
                "product_id": "FUND_001",
                "product_name": "稳健型混合基金",
                "product_type": "fund",
                "risk_level": RiskLevel.MODERATE,
                "min_investment": 1000.0,
                "expected_return": 0.06,
                "features": {
                    "asset_allocation": "60% bonds, 30% stocks, 10% cash",
                    "management_fee": 0.012
                },
                "target_segments": [UserSegment.MIDDLE_AGED_FAMILY]
            },
            {
                "product_id": "FUND_002",
                "product_name": "低门槛定投基金",
                "product_type": "fund",
                "risk_level": RiskLevel.MODERATE,
                "min_investment": 100.0,
                "expected_return": 0.08,
                "features": {
                    "auto_invest": True,
                    "min_amount": 100,
                    "investment_period": "monthly"
                },
                "target_segments": [UserSegment.YOUNG_PROFESSIONAL]
            },
            # Insurance products
            {
                "product_id": "INS_001",
                "product_name": "教育金保险",
                "product_type": "insurance",
                "risk_level": RiskLevel.CONSERVATIVE,
                "min_investment": 5000.0,
                "expected_return": 0.04,
                "features": {
                    "education_coverage": True,
                    "guaranteed_return": True,
                    "term": "15 years"
                },
                "target_segments": [UserSegment.MIDDLE_AGED_FAMILY]
            },
            # Fixed income products
            {
                "product_id": "BOND_001",
                "product_name": "国债逆回购",
                "product_type": "bond",
                "risk_level": RiskLevel.CONSERVATIVE,
                "min_investment": 1000.0,
                "expected_return": 0.025,
                "features": {
                    "principal_safety": True,
                    "short_term": True,
                    "liquidity": "high"
                },
                "target_segments": [UserSegment.RETIRED]
            },
            # Equity products
            {
                "product_id": "STOCK_001",
                "product_name": "高股息股票组合",
                "product_type": "stock",
                "risk_level": RiskLevel.MODERATE,
                "min_investment": 10000.0,
                "expected_return": 0.055,
                "features": {
                    "dividend_yield": "4-6%",
                    "blue_chip": True,
                    "defensive": True
                },
                "target_segments": [UserSegment.RETIRED]
            }
        ]
    
    def generate_recommendations(
        self,
        request: RecommendationRequest,
        top_k: int = 5
    ) -> RecommendationResponse:
        """
        Generate personalized recommendations for a user
        
        Args:
            request: RecommendationRequest with user info
            top_k: Number of recommendations to return
            
        Returns:
            RecommendationResponse with recommendations and explanations
        """
        user_id = request.user_id
        
        # Get user profile
        profile = self.user_profile_manager.get_profile(user_id)
        if not profile:
            logger.warning(f"Profile not found for user {user_id}, using defaults")
            return self._generate_default_recommendations(request, top_k)
        
        # Identify user needs
        needs = self.user_profile_manager.identify_needs(user_id)
        
        # Get strategy for user segment
        strategy = self._get_strategy_for_segment(profile.segment)
        
        # Filter and score products
        candidate_products = self._filter_products(profile, needs)
        scored_products = self._score_products(profile, candidate_products, needs)
        
        # Sort by score and select top-k
        scored_products.sort(key=lambda x: x["score"], reverse=True)
        top_products = scored_products[:top_k]
        
        # Convert to RecommendationItems with explanations
        recommendations = []
        for product_data in top_products:
            product = product_data["product"]
            reasoning = self._generate_reasoning(profile, product, needs)
            
            item = RecommendationItem(
                product_id=product["product_id"],
                product_name=product["product_name"],
                product_type=product["product_type"],
                score=product_data["score"],
                reasoning=reasoning,
                risk_level=product["risk_level"],
                expected_return=product.get("expected_return"),
                min_investment=product.get("min_investment"),
                features=product.get("features", {})
            )
            recommendations.append(item)
        
        # Generate overall explanation
        explanation = self._generate_overall_explanation(profile, recommendations, needs)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(profile, recommendations)
        
        response = RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            explanation=explanation,
            confidence_score=confidence_score,
            strategy_used=strategy
        )
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {user_id}")
        return response
    
    def _filter_products(
        self,
        profile,
        needs: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter products based on user profile and needs"""
        candidates = []
        
        for product in self.product_catalog:
            # Check segment match
            if profile.segment not in product.get("target_segments", []):
                continue
            
            # Check risk level compatibility
            if not self._is_risk_compatible(profile.risk_level, product["risk_level"]):
                continue
            
            # Check minimum investment
            if product.get("min_investment", 0) > profile.savings * 0.3:
                continue
            
            candidates.append(product)
        
        return candidates
    
    def _score_products(
        self,
        profile,
        products: List[Dict[str, Any]],
        needs: List[str]
    ) -> List[Dict[str, Any]]:
        """Score products based on user profile and needs"""
        scored = []
        
        for product in products:
            score = 0.0
            
            # Base score from expected return
            score += product.get("expected_return", 0) * 10
            
            # Boost for risk level match
            if product["risk_level"] == profile.risk_level:
                score += 2.0
            
            # Boost for segment match
            if profile.segment in product.get("target_segments", []):
                score += 3.0
            
            # Boost for need alignment
            product_type = product["product_type"]
            if "auto_savings_plan" in needs and product_type == "savings":
                score += 4.0
            if "education_insurance" in needs and product_type == "insurance":
                score += 4.0
            if "high_dividend_stocks" in needs and product_type == "stock":
                score += 4.0
            
            # Use RL engine if available
            if self.rl_engine:
                # In production, convert profile to state vector
                # rl_score = self.rl_engine.recommend(user_state, top_k=1)[0]
                pass
            
            scored.append({
                "product": product,
                "score": score
            })
        
        return scored
    
    def _is_risk_compatible(
        self,
        user_risk: RiskLevel,
        product_risk: RiskLevel
    ) -> bool:
        """Check if product risk is compatible with user risk tolerance"""
        risk_levels = {
            RiskLevel.CONSERVATIVE: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.AGGRESSIVE: 3
        }
        
        user_level = risk_levels[user_risk]
        product_level = risk_levels[product_risk]
        
        # Users can accept products at or below their risk level
        return product_level <= user_level
    
    def _generate_reasoning(
        self,
        profile,
        product: Dict[str, Any],
        needs: List[str]
    ) -> List[str]:
        """Generate reasoning for why a product is recommended"""
        reasoning = []
        
        # Segment-based reasoning
        if profile.segment == UserSegment.YOUNG_PROFESSIONAL:
            reasoning.append(f"适合{profile.age}岁的年轻职场人士")
        elif profile.segment == UserSegment.MIDDLE_AGED_FAMILY:
            reasoning.append("符合中年家庭的稳健理财需求")
        elif profile.segment == UserSegment.RETIRED:
            reasoning.append("适合退休人群的低风险投资")
        
        # Risk-based reasoning
        reasoning.append(f"风险等级为{product['risk_level'].value}，与您的风险承受能力匹配")
        
        # Return-based reasoning
        if product.get("expected_return"):
            return_pct = product["expected_return"] * 100
            reasoning.append(f"预期年化收益率约{return_pct:.1f}%")
        
        # Feature-based reasoning
        features = product.get("features", {})
        if features.get("auto_transfer") or features.get("auto_invest"):
            reasoning.append("支持自动投资，帮助养成理财习惯")
        if features.get("principal_safety"):
            reasoning.append("保障本金安全")
        
        return reasoning
    
    def _generate_overall_explanation(
        self,
        profile,
        recommendations: List[RecommendationItem],
        needs: List[str]
    ) -> str:
        """Generate overall explanation for the recommendation set"""
        
        segments_map = {
            UserSegment.YOUNG_PROFESSIONAL: "年轻职场人士",
            UserSegment.MIDDLE_AGED_FAMILY: "中年家庭用户",
            UserSegment.RETIRED: "退休人群"
        }
        
        risk_map = {
            RiskLevel.CONSERVATIVE: "保守型",
            RiskLevel.MODERATE: "稳健型",
            RiskLevel.AGGRESSIVE: "进取型"
        }
        
        segment_name = segments_map.get(profile.segment, "投资者")
        risk_name = risk_map.get(profile.risk_level, "")
        
        explanation = (
            f"基于您作为{segment_name}的财务状况和{risk_name}风险偏好，"
            f"我们为您精选了{len(recommendations)}款理财产品。"
        )
        
        if "auto_savings_plan" in needs:
            explanation += "考虑到您需要建立储蓄习惯，我们推荐了自动储蓄产品。"
        
        if "education_insurance" in needs:
            explanation += "针对子女教育规划需求，推荐了教育金保险。"
        
        explanation += "这些产品能够帮助您实现财务目标，并有效分散投资风险。"
        
        return explanation
    
    def _calculate_confidence(
        self,
        profile,
        recommendations: List[RecommendationItem]
    ) -> float:
        """Calculate confidence score for recommendations"""
        
        # Base confidence from data completeness
        data_completeness = 0.7  # Placeholder
        
        # Boost from number of transactions
        if len(profile.transaction_history) > 10:
            data_completeness += 0.1
        
        # Boost from profile completeness
        if profile.credit_score:
            data_completeness += 0.1
        
        # Average score of recommendations
        avg_score = np.mean([r.score for r in recommendations]) if recommendations else 0
        score_confidence = min(avg_score / 10, 1.0)
        
        # Combined confidence
        confidence = (data_completeness + score_confidence) / 2
        
        return min(confidence, 1.0)
    
    def _get_strategy_for_segment(self, segment: UserSegment) -> str:
        """Get recommendation strategy name for segment"""
        strategy_map = {
            UserSegment.YOUNG_PROFESSIONAL: "young_professional_strategy",
            UserSegment.MIDDLE_AGED_FAMILY: "middle_aged_family_strategy",
            UserSegment.RETIRED: "retired_strategy"
        }
        return strategy_map.get(segment, "default_strategy")
    
    def _generate_default_recommendations(
        self,
        request: RecommendationRequest,
        top_k: int
    ) -> RecommendationResponse:
        """Generate default recommendations when profile is not available"""
        
        default_products = self.product_catalog[:top_k]
        recommendations = []
        
        for product in default_products:
            item = RecommendationItem(
                product_id=product["product_id"],
                product_name=product["product_name"],
                product_type=product["product_type"],
                score=5.0,
                reasoning=["基于热门产品推荐"],
                risk_level=product["risk_level"],
                expected_return=product.get("expected_return"),
                min_investment=product.get("min_investment"),
                features=product.get("features", {})
            )
            recommendations.append(item)
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            explanation="这些是我们的热门理财产品，建议您完善个人信息以获得更精准的推荐。",
            confidence_score=0.5,
            strategy_used="default"
        )
