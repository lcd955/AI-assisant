"""
Example usage of the AI Financial Recommendation System
Demonstrates key features and workflows
"""
from datetime import datetime, timedelta
from models.schemas import (
    UserSegment,
    RiskLevel,
    Transaction,
    RecommendationRequest,
    QueryRequest
)
from src.dialogue_engine import FinancialDialogueEngine
from src.user_profile import UserProfileManager
from src.recommendation import PersonalizedRecommendationEngine
from src.explainability import ExplainabilityEngine
from src.simulation import FinancialSimulator, SimulationScenario
from utils.logger import setup_logger

logger = setup_logger()


def example_1_dialogue_interaction():
    """Example 1: Natural language dialogue interaction"""
    print("\n" + "="*60)
    print("Example 1: Natural Language Dialogue")
    print("="*60)
    
    # Initialize dialogue engine
    dialogue_engine = FinancialDialogueEngine()
    
    # User queries
    queries = [
        "帮我看看上个月餐饮花了多少？",
        "如果我想三年内买房，每月该存多少？",
        "那投资呢？"  # Context-aware follow-up
    ]
    
    user_id = "user_001"
    session_id = None
    
    for query in queries:
        print(f"\n用户: {query}")
        
        # Process query
        result = dialogue_engine.process_query(
            user_id=user_id,
            query=query,
            session_id=session_id
        )
        
        session_id = result["session_id"]
        
        print(f"助手: {result['response']}")
        print(f"意图: {result['intent']}")
        if result.get("suggestions"):
            print(f"建议: {', '.join(result['suggestions'][:3])}")


def example_2_financial_simulation():
    """Example 2: Financial simulation sandbox"""
    print("\n" + "="*60)
    print("Example 2: Financial Simulation Sandbox")
    print("="*60)
    
    simulator = FinancialSimulator()
    
    # Scenario: Saving for house down payment
    print("\n场景: 三年内存够房子首付")
    
    goal_result = simulator.simulate_goal_planning(
        goal_amount=500000,  # 50万首付
        time_horizon_months=36,  # 3年
        current_savings=100000,  # 当前有10万
        investment_return_rate=0.05
    )
    
    print(f"  目标金额: ¥{goal_result['goal_amount']:,.0f}")
    print(f"  当前储蓄: ¥100,000")
    print(f"  时间期限: {goal_result['time_horizon_months']}个月")
    print(f"  所需月存: ¥{goal_result['required_monthly_savings']:,.0f}")
    
    simulation = goal_result['simulation']
    print(f"\n模拟结果:")
    print(f"  最终余额: ¥{simulation['final_balance']:,.0f}")
    print(f"  累计存入: ¥{simulation['total_contributed']:,.0f}")
    print(f"  投资收益: ¥{simulation['investment_gains']:,.0f}")
    print(f"  目标达成: {'是' if simulation['goal_achieved'] else '否'}")
    
    print(f"\n建议:")
    for rec in goal_result['recommendations'][:3]:
        print(f"  - {rec}")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("AI Financial Recommendation System - Examples")
    print("="*60)
    
    try:
        # Run examples
        example_1_dialogue_interaction()
        example_2_financial_simulation()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
