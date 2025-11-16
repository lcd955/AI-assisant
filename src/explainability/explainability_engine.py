"""
Explainable AI module for transparent decision making
Provides detailed explanations for recommendations
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from models.schemas import RecommendationResponse, RecommendationItem, UserProfile
from utils.logger import setup_logger
from utils.config_loader import config

logger = setup_logger()


class ExplainabilityEngine:
    """
    Generates human-readable explanations for AI recommendations
    Makes decision-making transparent and trustworthy
    """
    
    def __init__(self):
        """Initialize explainability engine"""
        self.enabled = config.get("explainability.enabled", True)
        self.detail_level = config.get("explainability.detail_level", "high")
        self.include_confidence = config.get("explainability.include_confidence_scores", True)
        
        logger.info(f"ExplainabilityEngine initialized with detail_level={self.detail_level}")
    
    def explain_recommendation(
        self,
        recommendation: RecommendationItem,
        user_profile: Optional[UserProfile] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation for a single recommendation
        
        Args:
            recommendation: RecommendationItem to explain
            user_profile: User profile for context
            context: Additional context information
            
        Returns:
            Dictionary with explanation details
        """
        if not self.enabled:
            return {"explanation": "Explainability is disabled"}
        
        explanation = {
            "product_id": recommendation.product_id,
            "product_name": recommendation.product_name,
            "recommendation_score": recommendation.score,
            "reasoning": recommendation.reasoning,
            "decision_factors": self._extract_decision_factors(recommendation, user_profile),
            "risk_assessment": self._explain_risk(recommendation),
            "expected_outcomes": self._explain_expected_outcomes(recommendation),
            "alternatives": self._suggest_alternatives(recommendation),
        }
        
        if self.include_confidence:
            explanation["confidence_breakdown"] = self._explain_confidence(recommendation, user_profile)
        
        if self.detail_level == "high":
            explanation["detailed_analysis"] = self._generate_detailed_analysis(
                recommendation, user_profile, context
            )
        
        return explanation
    
    def explain_recommendation_set(
        self,
        response: RecommendationResponse,
        user_profile: Optional[UserProfile] = None
    ) -> Dict[str, Any]:
        """
        Generate explanation for the entire recommendation set
        
        Args:
            response: RecommendationResponse to explain
            user_profile: User profile for context
            
        Returns:
            Dictionary with comprehensive explanation
        """
        if not self.enabled:
            return {"explanation": "Explainability is disabled"}
        
        explanation = {
            "overview": response.explanation,
            "strategy": response.strategy_used,
            "overall_confidence": response.confidence_score,
            "timestamp": response.timestamp.isoformat(),
            "recommendation_count": len(response.recommendations),
            "individual_recommendations": [],
            "portfolio_rationale": self._explain_portfolio_composition(response),
            "risk_distribution": self._explain_risk_distribution(response),
        }
        
        # Explain each recommendation
        for rec in response.recommendations:
            rec_explanation = self.explain_recommendation(rec, user_profile)
            explanation["individual_recommendations"].append(rec_explanation)
        
        # Generate summary insights
        explanation["key_insights"] = self._generate_key_insights(response, user_profile)
        
        return explanation
    
    def _extract_decision_factors(
        self,
        recommendation: RecommendationItem,
        user_profile: Optional[UserProfile]
    ) -> List[Dict[str, Any]]:
        """Extract and explain key decision factors"""
        factors = []
        
        # Factor 1: Risk alignment
        if user_profile:
            factors.append({
                "factor": "风险匹配度",
                "value": recommendation.risk_level.value,
                "user_preference": user_profile.risk_level.value,
                "weight": 0.3,
                "explanation": f"产品风险等级{recommendation.risk_level.value}与您的风险偏好{user_profile.risk_level.value}相匹配"
            })
        
        # Factor 2: Expected return
        if recommendation.expected_return:
            factors.append({
                "factor": "预期收益",
                "value": f"{recommendation.expected_return * 100:.1f}%",
                "weight": 0.25,
                "explanation": f"预期年化收益率为{recommendation.expected_return * 100:.1f}%"
            })
        
        # Factor 3: Minimum investment
        if recommendation.min_investment:
            factors.append({
                "factor": "投资门槛",
                "value": f"¥{recommendation.min_investment:,.0f}",
                "weight": 0.15,
                "explanation": f"最低投资金额为{recommendation.min_investment:,.0f}元，符合您的资金状况"
            })
        
        # Factor 4: Product features
        if recommendation.features:
            feature_count = len(recommendation.features)
            factors.append({
                "factor": "产品特性",
                "value": f"{feature_count}项特色功能",
                "weight": 0.2,
                "explanation": f"产品具有{feature_count}项特色功能，满足您的多样化需求"
            })
        
        # Factor 5: Recommendation score
        factors.append({
            "factor": "综合评分",
            "value": f"{recommendation.score:.2f}/10",
            "weight": 0.1,
            "explanation": f"基于多维度分析，该产品获得{recommendation.score:.2f}分（满分10分）"
        })
        
        return factors
    
    def _explain_risk(self, recommendation: RecommendationItem) -> Dict[str, Any]:
        """Explain risk characteristics"""
        risk_explanations = {
            "conservative": {
                "level": "低风险",
                "description": "本金相对安全，收益稳定，适合风险承受能力较低的投资者",
                "typical_volatility": "年波动率通常小于5%",
                "potential_loss": "本金损失可能性极低"
            },
            "moderate": {
                "level": "中等风险",
                "description": "收益与风险相对平衡，适合稳健型投资者",
                "typical_volatility": "年波动率通常在5-15%之间",
                "potential_loss": "可能出现短期浮亏，但长期持有风险可控"
            },
            "aggressive": {
                "level": "高风险",
                "description": "追求较高收益，但波动较大，适合风险承受能力强的投资者",
                "typical_volatility": "年波动率可能超过15%",
                "potential_loss": "可能出现较大幅度的本金损失"
            }
        }
        
        risk_level = recommendation.risk_level.value
        base_explanation = risk_explanations.get(risk_level, risk_explanations["moderate"])
        
        return {
            **base_explanation,
            "mitigation_strategies": self._suggest_risk_mitigation(recommendation)
        }
    
    def _explain_expected_outcomes(
        self,
        recommendation: RecommendationItem
    ) -> Dict[str, Any]:
        """Explain expected outcomes of the investment"""
        outcomes = {
            "best_case": None,
            "expected_case": None,
            "worst_case": None,
            "time_horizon": "建议持有期限"
        }
        
        if recommendation.expected_return:
            base_return = recommendation.expected_return
            
            # Simulate scenarios based on risk level
            risk_multipliers = {
                "conservative": {"best": 1.2, "worst": 0.9},
                "moderate": {"best": 1.5, "worst": 0.8},
                "aggressive": {"best": 2.0, "worst": 0.5}
            }
            
            multiplier = risk_multipliers.get(
                recommendation.risk_level.value,
                {"best": 1.3, "worst": 0.85}
            )
            
            outcomes["best_case"] = {
                "return": f"{base_return * multiplier['best'] * 100:.1f}%",
                "description": "市场表现良好的情况下"
            }
            outcomes["expected_case"] = {
                "return": f"{base_return * 100:.1f}%",
                "description": "正常市场条件下"
            }
            outcomes["worst_case"] = {
                "return": f"{base_return * multiplier['worst'] * 100:.1f}%",
                "description": "市场表现不佳的情况下"
            }
        
        # Recommend holding period based on product type
        holding_periods = {
            "savings": "3-6个月",
            "fund": "1-3年",
            "stock": "3-5年",
            "bond": "6个月-2年",
            "insurance": "长期持有（5年以上）"
        }
        outcomes["time_horizon"] = holding_periods.get(
            recommendation.product_type,
            "根据个人需求调整"
        )
        
        return outcomes
    
    def _suggest_alternatives(
        self,
        recommendation: RecommendationItem
    ) -> List[str]:
        """Suggest alternative products"""
        alternatives = [
            f"同类型其他{recommendation.product_type}产品",
            "风险等级相近的其他产品",
            "收益率相似但不同类型的产品"
        ]
        return alternatives
    
    def _explain_confidence(
        self,
        recommendation: RecommendationItem,
        user_profile: Optional[UserProfile]
    ) -> Dict[str, Any]:
        """Break down confidence score"""
        breakdown = {
            "overall_confidence": recommendation.score / 10,
            "components": []
        }
        
        if user_profile:
            # Data quality confidence
            data_quality = 0.7 if len(user_profile.transaction_history) > 10 else 0.5
            breakdown["components"].append({
                "component": "数据完整性",
                "score": data_quality,
                "description": "基于用户历史数据的完整程度"
            })
            
            # Profile match confidence
            breakdown["components"].append({
                "component": "用户匹配度",
                "score": 0.8,
                "description": "产品与用户画像的匹配程度"
            })
        
        # Market confidence
        breakdown["components"].append({
            "component": "市场环境",
            "score": 0.75,
            "description": "当前市场环境对该产品的适合度"
        })
        
        return breakdown
    
    def _generate_detailed_analysis(
        self,
        recommendation: RecommendationItem,
        user_profile: Optional[UserProfile],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate detailed textual analysis"""
        analysis_parts = [
            f"## {recommendation.product_name} 详细分析\n",
            f"### 推荐理由",
        ]
        
        for reason in recommendation.reasoning:
            analysis_parts.append(f"- {reason}")
        
        analysis_parts.append("\n### 产品特点")
        for key, value in recommendation.features.items():
            analysis_parts.append(f"- {key}: {value}")
        
        if user_profile:
            analysis_parts.append(f"\n### 个性化分析")
            analysis_parts.append(
                f"根据您{user_profile.age}岁的年龄、{user_profile.segment.value}的用户类型，"
                f"以及{user_profile.risk_level.value}的风险偏好，该产品能够很好地满足您的需求。"
            )
        
        return "\n".join(analysis_parts)
    
    def _explain_portfolio_composition(
        self,
        response: RecommendationResponse
    ) -> Dict[str, Any]:
        """Explain the composition of recommended portfolio"""
        composition = {
            "by_type": {},
            "by_risk": {},
            "diversification_score": 0.0
        }
        
        # Analyze by product type
        for rec in response.recommendations:
            prod_type = rec.product_type
            composition["by_type"][prod_type] = composition["by_type"].get(prod_type, 0) + 1
        
        # Analyze by risk level
        for rec in response.recommendations:
            risk = rec.risk_level.value
            composition["by_risk"][risk] = composition["by_risk"].get(risk, 0) + 1
        
        # Calculate diversification score
        type_diversity = len(composition["by_type"]) / len(response.recommendations)
        risk_diversity = len(composition["by_risk"]) / len(response.recommendations)
        composition["diversification_score"] = (type_diversity + risk_diversity) / 2
        
        return composition
    
    def _explain_risk_distribution(
        self,
        response: RecommendationResponse
    ) -> Dict[str, Any]:
        """Explain risk distribution across recommendations"""
        risk_counts = {}
        for rec in response.recommendations:
            risk = rec.risk_level.value
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        total = len(response.recommendations)
        risk_percentages = {
            risk: (count / total * 100) for risk, count in risk_counts.items()
        }
        
        return {
            "distribution": risk_percentages,
            "summary": self._generate_risk_summary(risk_percentages)
        }
    
    def _generate_risk_summary(self, risk_percentages: Dict[str, float]) -> str:
        """Generate textual summary of risk distribution"""
        conservative_pct = risk_percentages.get("conservative", 0)
        moderate_pct = risk_percentages.get("moderate", 0)
        aggressive_pct = risk_percentages.get("aggressive", 0)
        
        if conservative_pct > 60:
            return "推荐组合以低风险产品为主，注重本金安全"
        elif aggressive_pct > 60:
            return "推荐组合以高风险产品为主，追求更高收益"
        else:
            return "推荐组合风险分布均衡，兼顾收益与安全"
    
    def _suggest_risk_mitigation(
        self,
        recommendation: RecommendationItem
    ) -> List[str]:
        """Suggest risk mitigation strategies"""
        strategies = []
        
        if recommendation.risk_level.value == "aggressive":
            strategies.extend([
                "建议仅投入总资产的20-30%",
                "设置止损点，控制最大亏损",
                "定期审查投资表现，及时调整"
            ])
        elif recommendation.risk_level.value == "moderate":
            strategies.extend([
                "可配置总资产的30-50%",
                "采用定投策略分散时间风险",
                "保持适当的资产流动性"
            ])
        else:  # conservative
            strategies.extend([
                "适合作为核心资产配置",
                "可投入较大比例资金",
                "长期持有以获得稳定收益"
            ])
        
        return strategies
    
    def _generate_key_insights(
        self,
        response: RecommendationResponse,
        user_profile: Optional[UserProfile]
    ) -> List[str]:
        """Generate key insights about the recommendations"""
        insights = []
        
        # Insight about strategy
        insights.append(f"采用{response.strategy_used}策略为您推荐产品")
        
        # Insight about diversification
        product_types = set(rec.product_type for rec in response.recommendations)
        if len(product_types) > 2:
            insights.append(f"推荐组合涵盖{len(product_types)}种不同类型的产品，有效分散风险")
        
        # Insight about returns
        avg_return = np.mean([
            rec.expected_return for rec in response.recommendations
            if rec.expected_return
        ])
        if avg_return > 0:
            insights.append(f"平均预期年化收益率约{avg_return * 100:.1f}%")
        
        # Insight about accessibility
        min_investments = [rec.min_investment for rec in response.recommendations if rec.min_investment]
        if min_investments:
            min_threshold = min(min_investments)
            insights.append(f"最低投资门槛为¥{min_threshold:,.0f}，适合不同资金规模的投资者")
        
        return insights
    
    def generate_explanation_report(
        self,
        response: RecommendationResponse,
        user_profile: Optional[UserProfile] = None,
        format: str = "json"
    ) -> str:
        """
        Generate a comprehensive explanation report
        
        Args:
            response: RecommendationResponse to explain
            user_profile: User profile for context
            format: Output format ('json' or 'text')
            
        Returns:
            Formatted explanation report
        """
        explanation = self.explain_recommendation_set(response, user_profile)
        
        if format == "json":
            return json.dumps(explanation, ensure_ascii=False, indent=2)
        else:
            # Generate text report
            return self._format_text_report(explanation)
    
    def _format_text_report(self, explanation: Dict[str, Any]) -> str:
        """Format explanation as text report"""
        lines = [
            "=" * 60,
            "AI 理财推荐解释报告",
            "=" * 60,
            f"\n时间：{explanation['timestamp']}",
            f"策略：{explanation['strategy']}",
            f"置信度：{explanation['overall_confidence']:.2f}",
            f"\n总体说明：\n{explanation['overview']}",
            "\n" + "=" * 60,
            "关键洞察：",
            "=" * 60,
        ]
        
        for insight in explanation.get("key_insights", []):
            lines.append(f"• {insight}")
        
        lines.extend([
            "\n" + "=" * 60,
            "个别产品分析：",
            "=" * 60,
        ])
        
        for rec_exp in explanation.get("individual_recommendations", []):
            lines.append(f"\n产品：{rec_exp['product_name']}")
            lines.append(f"评分：{rec_exp['recommendation_score']:.2f}/10")
            lines.append("推荐理由：")
            for reason in rec_exp.get("reasoning", []):
                lines.append(f"  - {reason}")
        
        return "\n".join(lines)


# Import numpy for calculations
import numpy as np
