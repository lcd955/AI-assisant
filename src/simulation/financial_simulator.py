"""
Financial Simulation Sandbox
Allows users to simulate financial scenarios and see projected outcomes
"""
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger()


@dataclass
class SimulationScenario:
    """Financial simulation scenario"""
    name: str
    monthly_income: float
    monthly_expenses: float
    current_savings: float
    monthly_savings: float
    investment_return_rate: float  # Annual rate
    time_horizon_months: int
    goal_amount: Optional[float] = None
    inflation_rate: float = 0.03  # Annual inflation rate


class FinancialSimulator:
    """
    Financial simulation sandbox for scenario planning
    Helps users understand long-term financial outcomes
    """
    
    def __init__(self):
        """Initialize financial simulator"""
        logger.info("FinancialSimulator initialized")
    
    def simulate_savings_plan(
        self,
        scenario: SimulationScenario
    ) -> Dict[str, Any]:
        """
        Simulate a savings plan over time
        
        Args:
            scenario: SimulationScenario with parameters
            
        Returns:
            Simulation results with projections
        """
        months = scenario.time_horizon_months
        monthly_return_rate = scenario.investment_return_rate / 12
        monthly_inflation_rate = scenario.inflation_rate / 12
        
        # Initialize arrays
        timeline = []
        balance = []
        total_saved = []
        real_value = []
        
        current_balance = scenario.current_savings
        cumulative_savings = 0
        
        for month in range(months + 1):
            # Record current state
            timeline.append(month)
            balance.append(current_balance)
            total_saved.append(cumulative_savings)
            
            # Calculate real value (adjusted for inflation)
            real_balance = current_balance / ((1 + monthly_inflation_rate) ** month)
            real_value.append(real_balance)
            
            # Update for next month
            if month < months:
                # Add monthly savings
                current_balance += scenario.monthly_savings
                cumulative_savings += scenario.monthly_savings
                
                # Apply investment returns
                current_balance *= (1 + monthly_return_rate)
        
        # Calculate goal achievement
        goal_achieved = False
        months_to_goal = None
        
        if scenario.goal_amount:
            goal_achieved = current_balance >= scenario.goal_amount
            
            # Find when goal is reached
            for i, bal in enumerate(balance):
                if bal >= scenario.goal_amount:
                    months_to_goal = i
                    break
        
        # Generate insights
        insights = self._generate_savings_insights(
            scenario, current_balance, cumulative_savings, goal_achieved, months_to_goal
        )
        
        return {
            "scenario": scenario.name,
            "timeline": timeline,
            "balance": balance,
            "total_saved": total_saved,
            "real_value": real_value,
            "final_balance": current_balance,
            "final_real_value": real_value[-1],
            "total_contributed": cumulative_savings,
            "investment_gains": current_balance - scenario.current_savings - cumulative_savings,
            "goal_achieved": goal_achieved,
            "months_to_goal": months_to_goal,
            "insights": insights
        }
    
    def simulate_goal_planning(
        self,
        goal_amount: float,
        time_horizon_months: int,
        current_savings: float,
        investment_return_rate: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate required monthly savings to reach a goal
        
        Args:
            goal_amount: Target amount to reach
            time_horizon_months: Time period in months
            current_savings: Starting amount
            investment_return_rate: Annual return rate
            
        Returns:
            Required savings plan
        """
        monthly_return_rate = investment_return_rate / 12
        
        # Calculate required monthly savings
        future_value_of_current = current_savings * ((1 + monthly_return_rate) ** time_horizon_months)
        remaining_amount = goal_amount - future_value_of_current
        
        if remaining_amount <= 0:
            required_monthly_savings = 0
        else:
            if monthly_return_rate > 0:
                required_monthly_savings = remaining_amount * monthly_return_rate / (
                    ((1 + monthly_return_rate) ** time_horizon_months) - 1
                )
            else:
                required_monthly_savings = remaining_amount / time_horizon_months
        
        # Create scenario and simulate
        scenario = SimulationScenario(
            name="Goal Planning",
            monthly_income=0,
            monthly_expenses=0,
            current_savings=current_savings,
            monthly_savings=required_monthly_savings,
            investment_return_rate=investment_return_rate,
            time_horizon_months=time_horizon_months,
            goal_amount=goal_amount
        )
        
        simulation_result = self.simulate_savings_plan(scenario)
        
        return {
            "required_monthly_savings": required_monthly_savings,
            "goal_amount": goal_amount,
            "time_horizon_months": time_horizon_months,
            "simulation": simulation_result,
            "is_achievable": True,
            "recommendations": self._generate_goal_recommendations(
                required_monthly_savings, goal_amount, time_horizon_months
            )
        }
    
    def _generate_savings_insights(
        self,
        scenario: SimulationScenario,
        final_balance: float,
        cumulative_savings: float,
        goal_achieved: bool,
        months_to_goal: Optional[int]
    ) -> List[str]:
        """Generate insights from savings simulation"""
        insights = []
        
        # Growth insight
        growth_rate = (final_balance - scenario.current_savings) / scenario.current_savings if scenario.current_savings > 0 else 0
        insights.append(f"您的资产将增长{growth_rate * 100:.1f}%，从{scenario.current_savings:,.0f}元增长到{final_balance:,.0f}元")
        
        # Savings vs investment gains
        investment_gains = final_balance - scenario.current_savings - cumulative_savings
        if investment_gains > 0:
            insights.append(f"通过投资获得{investment_gains:,.0f}元收益，占总增长的{investment_gains / (final_balance - scenario.current_savings) * 100:.1f}%")
        
        # Goal achievement
        if scenario.goal_amount:
            if goal_achieved:
                if months_to_goal:
                    years = months_to_goal // 12
                    months = months_to_goal % 12
                    insights.append(f"您将在{years}年{months}个月后达成目标")
            else:
                shortfall = scenario.goal_amount - final_balance
                insights.append(f"未能达成目标，还差{shortfall:,.0f}元")
        
        return insights
    
    def _generate_goal_recommendations(
        self,
        required_monthly_savings: float,
        goal_amount: float,
        time_horizon_months: int
    ) -> List[str]:
        """Generate recommendations for goal planning"""
        recommendations = []
        
        recommendations.append(f"建议每月储蓄{required_monthly_savings:,.0f}元以达成目标")
        
        if required_monthly_savings > 0:
            daily_savings = required_monthly_savings / 30
            recommendations.append(f"相当于每天储蓄约{daily_savings:,.0f}元")
        
        return recommendations
