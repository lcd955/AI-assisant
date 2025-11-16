"""
FastAPI REST API for AI Financial Recommendation System
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import uuid
from datetime import datetime

from models.schemas import (
    QueryRequest,
    QueryResponse,
    RecommendationRequest,
    RecommendationResponse,
    UserProfile,
    Transaction,
    UserSegment,
    RiskLevel
)
from src.dialogue_engine import FinancialDialogueEngine, VoiceInteractionHandler
from src.user_profile import UserProfileManager
from src.recommendation import PersonalizedRecommendationEngine
from src.explainability import ExplainabilityEngine
from src.multimodal import BillRecognitionEngine
from src.simulation import FinancialSimulator, SimulationScenario
from utils.config_loader import config
from utils.logger import setup_logger

logger = setup_logger()

# Initialize FastAPI app
app = FastAPI(
    title="AI Financial Recommendation System",
    description="Intelligent financial recommendation system with natural dialogue, personalized profiling, and explainable AI",
    version="1.0.0"
)

# Configure CORS
cors_origins = config.get("api.cors_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system components
dialogue_engine = None
user_profile_manager = None
recommendation_engine = None
explainability_engine = None
voice_handler = None
bill_recognition_engine = None
financial_simulator = None


@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global dialogue_engine, user_profile_manager, recommendation_engine, explainability_engine, voice_handler, bill_recognition_engine, financial_simulator
    
    logger.info("Initializing AI Financial Recommendation System...")
    
    # Initialize components
    dialogue_engine = FinancialDialogueEngine()
    user_profile_manager = UserProfileManager()
    recommendation_engine = PersonalizedRecommendationEngine(user_profile_manager)
    explainability_engine = ExplainabilityEngine()
    voice_handler = VoiceInteractionHandler(dialogue_engine)
    bill_recognition_engine = BillRecognitionEngine()
    financial_simulator = FinancialSimulator()
    
    logger.info("System initialization completed")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Financial Recommendation System API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "dialogue_engine": dialogue_engine is not None,
            "user_profile_manager": user_profile_manager is not None,
            "recommendation_engine": recommendation_engine is not None,
            "explainability_engine": explainability_engine is not None
        }
    }


# ========== Dialogue API Endpoints ==========

@app.post("/api/v1/dialogue/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a natural language query
    
    - **user_id**: User identifier
    - **query**: Natural language query in Chinese
    - **session_id**: Optional session ID for continuing conversation
    - **use_voice**: Whether this is a voice query
    """
    try:
        if request.use_voice and voice_handler:
            # Handle voice query
            result = voice_handler.voice_query(
                user_id=request.user_id,
                session_id=request.session_id,
                use_microphone=False,  # In API, expect audio file
                speak_response=False
            )
            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error"))
            
            return QueryResponse(**result)
        else:
            # Handle text query
            result = dialogue_engine.process_query(
                user_id=request.user_id,
                query=request.query,
                session_id=request.session_id
            )
            
            return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/dialogue/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a dialogue session"""
    try:
        dialogue_engine.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== User Profile API Endpoints ==========

@app.post("/api/v1/profile/create", response_model=UserProfile)
async def create_user_profile(
    user_id: str,
    age: int,
    monthly_income: float,
    monthly_expenses: float,
    savings: float,
    risk_level: RiskLevel = RiskLevel.MODERATE
):
    """
    Create a new user profile
    
    - **user_id**: Unique user identifier
    - **age**: User age
    - **monthly_income**: Monthly income in CNY
    - **monthly_expenses**: Monthly expenses in CNY
    - **savings**: Current savings in CNY
    - **risk_level**: Risk tolerance level (conservative, moderate, aggressive)
    """
    try:
        profile = user_profile_manager.create_profile(
            user_id=user_id,
            age=age,
            monthly_income=monthly_income,
            monthly_expenses=monthly_expenses,
            savings=savings,
            risk_level=risk_level
        )
        return profile
    except Exception as e:
        logger.error(f"Error creating profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/profile/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str):
    """Get user profile by ID"""
    try:
        profile = user_profile_manager.get_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        return profile
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/profile/{user_id}/transaction")
async def add_transaction(user_id: str, transaction: Transaction):
    """Add a transaction to user profile"""
    try:
        success = user_profile_manager.add_transaction(user_id, transaction)
        if not success:
            raise HTTPException(status_code=404, detail="Profile not found")
        return {"message": "Transaction added successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/profile/{user_id}/spending")
async def analyze_spending(user_id: str, period_days: int = 30):
    """Analyze user spending patterns"""
    try:
        analysis = user_profile_manager.analyze_spending_patterns(user_id, period_days)
        if not analysis:
            raise HTTPException(status_code=404, detail="Profile not found")
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing spending: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/profile/{user_id}/needs")
async def identify_needs(user_id: str):
    """Identify user financial needs"""
    try:
        needs = user_profile_manager.identify_needs(user_id)
        return {"user_id": user_id, "needs": needs}
    except Exception as e:
        logger.error(f"Error identifying needs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Recommendation API Endpoints ==========

@app.post("/api/v1/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest, top_k: int = 5):
    """
    Get personalized recommendations
    
    - **user_id**: User identifier
    - **context**: Optional context information
    - **preferences**: Optional user preferences
    - **top_k**: Number of recommendations to return (default: 5)
    """
    try:
        response = recommendation_engine.generate_recommendations(request, top_k)
        return response
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Explainability API Endpoints ==========

@app.post("/api/v1/explain/recommendation")
async def explain_recommendation(
    response: RecommendationResponse,
    user_id: Optional[str] = None
):
    """
    Get detailed explanation for recommendations
    
    - **response**: RecommendationResponse to explain
    - **user_id**: Optional user ID for profile-based explanation
    """
    try:
        user_profile = None
        if user_id:
            user_profile = user_profile_manager.get_profile(user_id)
        
        explanation = explainability_engine.explain_recommendation_set(
            response, user_profile
        )
        return explanation
    except Exception as e:
        logger.error(f"Error explaining recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/explain/report")
async def generate_explanation_report(
    response: RecommendationResponse,
    user_id: Optional[str] = None,
    format: str = "json"
):
    """
    Generate comprehensive explanation report
    
    - **response**: RecommendationResponse to explain
    - **user_id**: Optional user ID for profile-based explanation
    - **format**: Output format ('json' or 'text')
    """
    try:
        user_profile = None
        if user_id:
            user_profile = user_profile_manager.get_profile(user_id)
        
        report = explainability_engine.generate_explanation_report(
            response, user_profile, format
        )
        
        if format == "json":
            return {"report": report}
        else:
            return {"report": report, "content_type": "text/plain"}
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== System Statistics Endpoints ==========

@app.get("/api/v1/stats/graph")
async def get_graph_statistics():
    """Get statistics about the user-product-scene graph"""
    try:
        stats = user_profile_manager.get_graph_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting graph statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Multi-modal Endpoints ==========

@app.post("/api/v1/multimodal/recognize-bill")
async def recognize_bill(
    user_id: str,
    file: UploadFile = File(...)
):
    """
    Recognize bill information from uploaded image
    
    - **user_id**: User identifier
    - **file**: Image file (JPG, PNG, etc.)
    """
    try:
        # Read image data
        image_data = await file.read()
        
        # Recognize bill
        result = bill_recognition_engine.recognize_bill_from_image(
            image_data, user_id
        )
        
        # If successful and transaction created, add to profile
        if result.get("success") and result.get("transaction"):
            user_profile_manager.add_transaction(
                user_id, result["transaction"]
            )
        
        return result
    except Exception as e:
        logger.error(f"Error recognizing bill: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/multimodal/batch-recognize-bills")
async def batch_recognize_bills(
    user_id: str,
    files: List[UploadFile] = File(...)
):
    """
    Batch recognize bills from multiple images
    
    - **user_id**: User identifier
    - **files**: List of image files
    """
    try:
        image_list = []
        for file in files:
            image_data = await file.read()
            image_list.append(image_data)
        
        # Batch recognize
        results = bill_recognition_engine.batch_recognize(image_list, user_id)
        
        # Add successful transactions to profile
        for result in results:
            if result.get("success") and result.get("transaction"):
                user_profile_manager.add_transaction(
                    user_id, result["transaction"]
                )
        
        # Get statistics
        stats = bill_recognition_engine.get_recognition_statistics(results)
        
        return {
            "results": results,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error in batch recognition: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Financial Simulation Endpoints ==========

@app.post("/api/v1/simulation/savings-plan")
async def simulate_savings_plan(scenario: SimulationScenario):
    """
    Simulate a savings plan
    
    - **scenario**: SimulationScenario with all parameters
    """
    try:
        result = financial_simulator.simulate_savings_plan(scenario)
        return result
    except Exception as e:
        logger.error(f"Error simulating savings plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/simulation/goal-planning")
async def simulate_goal_planning(
    goal_amount: float,
    time_horizon_months: int,
    current_savings: float,
    investment_return_rate: float = 0.05
):
    """
    Calculate required monthly savings to reach a financial goal
    
    - **goal_amount**: Target amount (e.g., 500000 for house down payment)
    - **time_horizon_months**: Time period in months (e.g., 36 for 3 years)
    - **current_savings**: Current savings amount
    - **investment_return_rate**: Expected annual return rate (default: 0.05)
    """
    try:
        result = financial_simulator.simulate_goal_planning(
            goal_amount=goal_amount,
            time_horizon_months=time_horizon_months,
            current_savings=current_savings,
            investment_return_rate=investment_return_rate
        )
        return result
    except Exception as e:
        logger.error(f"Error simulating goal planning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/simulation/what-if")
async def what_if_scenario(
    user_id: str,
    scenario_name: str,
    monthly_savings_increase: float,
    time_horizon_years: int = 5
):
    """
    What-if scenario: "What if I save X more per month for Y years?"
    
    - **user_id**: User identifier
    - **scenario_name**: Name for the scenario
    - **monthly_savings_increase**: Additional monthly savings amount
    - **time_horizon_years**: Simulation period in years
    """
    try:
        # Get user profile
        profile = user_profile_manager.get_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Create scenario
        current_monthly_savings = profile.monthly_income - profile.monthly_expenses
        new_monthly_savings = current_monthly_savings + monthly_savings_increase
        
        scenario = SimulationScenario(
            name=scenario_name,
            monthly_income=profile.monthly_income,
            monthly_expenses=profile.monthly_expenses,
            current_savings=profile.savings,
            monthly_savings=new_monthly_savings,
            investment_return_rate=0.06,  # Moderate return
            time_horizon_months=time_horizon_years * 12
        )
        
        result = financial_simulator.simulate_savings_plan(scenario)
        
        return {
            "scenario_name": scenario_name,
            "current_monthly_savings": current_monthly_savings,
            "new_monthly_savings": new_monthly_savings,
            "monthly_increase": monthly_savings_increase,
            "simulation": result,
            "difference": result["final_balance"] - profile.savings
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in what-if scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
