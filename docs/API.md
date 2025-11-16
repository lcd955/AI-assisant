# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, implement API keys or OAuth2.

## Interactive Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check

#### GET /health

Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-16T10:30:00",
  "components": {
    "dialogue_engine": true,
    "user_profile_manager": true,
    "recommendation_engine": true,
    "explainability_engine": true
  }
}
```

---

### Dialogue API

#### POST /api/v1/dialogue/query

Process a natural language query.

**Request Body:**
```json
{
  "user_id": "user_001",
  "query": "帮我看看上个月餐饮花了多少？",
  "session_id": "optional_session_id",
  "use_voice": false
}
```

**Response:**
```json
{
  "response": "根据您的账单记录，上个月您在餐饮方面的支出为1,234元...",
  "intent": "query_expense",
  "entities": {
    "time_period": ["上个月"],
    "category": ["餐饮"]
  },
  "session_id": "generated_session_id",
  "suggestions": [
    "查看本月总支出",
    "对比上个月支出变化",
    "设置消费预算"
  ]
}
```

#### DELETE /api/v1/dialogue/session/{session_id}

Clear a dialogue session.

---

### User Profile API

#### POST /api/v1/profile/create

Create a new user profile.

**Parameters:**
- `user_id` (string, required): User identifier
- `age` (integer, required): User age
- `monthly_income` (float, required): Monthly income in CNY
- `monthly_expenses` (float, required): Monthly expenses in CNY
- `savings` (float, required): Current savings in CNY
- `risk_level` (string, optional): "conservative", "moderate", or "aggressive"

**Response:**
```json
{
  "user_id": "user_001",
  "segment": "young_professional",
  "risk_level": "moderate",
  "age": 28,
  "monthly_income": 12000.0,
  "monthly_expenses": 8000.0,
  "savings": 50000.0,
  "created_at": "2025-01-16T10:30:00"
}
```

#### GET /api/v1/profile/{user_id}

Get user profile by ID.

#### POST /api/v1/profile/{user_id}/transaction

Add a transaction to user profile.

**Request Body:**
```json
{
  "id": "t_001",
  "user_id": "user_001",
  "amount": 150.0,
  "category": "餐饮",
  "description": "午餐",
  "timestamp": "2025-01-16T12:30:00",
  "merchant": "美团外卖"
}
```

#### GET /api/v1/profile/{user_id}/spending

Analyze user spending patterns.

**Parameters:**
- `period_days` (integer, optional, default=30): Analysis period

**Response:**
```json
{
  "total_spending": 8500.0,
  "categories": {
    "餐饮": 2000.0,
    "交通": 1000.0,
    "住房": 3000.0,
    "其他": 2500.0
  },
  "category_percentages": {
    "餐饮": 23.5,
    "交通": 11.8,
    "住房": 35.3,
    "其他": 29.4
  },
  "is_overspending": false,
  "savings_rate": 0.29,
  "period_days": 30
}
```

#### GET /api/v1/profile/{user_id}/needs

Identify user financial needs.

---

### Recommendation API

#### POST /api/v1/recommendations

Get personalized product recommendations.

**Request Body:**
```json
{
  "user_id": "user_001",
  "context": "optional context",
  "preferences": {}
}
```

**Parameters:**
- `top_k` (integer, optional, default=5): Number of recommendations

**Response:**
```json
{
  "user_id": "user_001",
  "recommendations": [
    {
      "product_id": "FUND_001",
      "product_name": "稳健型混合基金",
      "product_type": "fund",
      "score": 8.5,
      "reasoning": [
        "适合28岁的年轻职场人士",
        "风险等级为moderate，与您的风险承受能力匹配",
        "预期年化收益率约6.0%"
      ],
      "risk_level": "moderate",
      "expected_return": 0.06,
      "min_investment": 1000.0,
      "features": {
        "asset_allocation": "60% bonds, 30% stocks, 10% cash",
        "management_fee": 0.012
      }
    }
  ],
  "explanation": "基于您作为年轻职场人士的财务状况...",
  "confidence_score": 0.85,
  "timestamp": "2025-01-16T10:30:00",
  "strategy_used": "young_professional_strategy"
}
```

---

### Explainability API

#### POST /api/v1/explain/recommendation

Get detailed explanation for recommendations.

**Request Body:**
```json
{
  "response": {...},  // RecommendationResponse object
  "user_id": "user_001"
}
```

#### POST /api/v1/explain/report

Generate comprehensive explanation report.

**Parameters:**
- `format` (string, optional): "json" or "text"

---

### Multi-modal API

#### POST /api/v1/multimodal/recognize-bill

Recognize bill information from uploaded image.

**Parameters:**
- `user_id` (string, required)

**Form Data:**
- `file` (file, required): Image file

**Response:**
```json
{
  "success": true,
  "recognized_text": "星巴克咖啡\n2025-01-15 14:30\n拿铁咖啡 x2 60.00元",
  "bill_info": {
    "merchant": "星巴克咖啡",
    "amount": 60.0,
    "date": "2025-01-15T14:30:00",
    "category": "餐饮"
  },
  "transaction": {
    "id": "bill_user_001_1705392000",
    "user_id": "user_001",
    "amount": 60.0,
    "category": "餐饮",
    "description": "星巴克咖啡",
    "merchant": "星巴克咖啡"
  }
}
```

#### POST /api/v1/multimodal/batch-recognize-bills

Batch recognize bills from multiple images.

**Form Data:**
- `files` (list of files): Multiple image files

---

### Simulation API

#### POST /api/v1/simulation/savings-plan

Simulate a savings plan.

**Request Body:**
```json
{
  "name": "三年购房储蓄计划",
  "monthly_income": 15000.0,
  "monthly_expenses": 10000.0,
  "current_savings": 100000.0,
  "monthly_savings": 5000.0,
  "investment_return_rate": 0.05,
  "time_horizon_months": 36,
  "goal_amount": 500000.0,
  "inflation_rate": 0.03
}
```

**Response:**
```json
{
  "scenario": "三年购房储蓄计划",
  "timeline": [0, 1, 2, ...36],
  "balance": [100000, 105250, ...],
  "final_balance": 285432.18,
  "total_contributed": 180000.0,
  "investment_gains": 5432.18,
  "goal_achieved": false,
  "months_to_goal": null,
  "insights": [
    "您的资产将增长185.4%，从100,000元增长到285,432元",
    "通过投资获得5,432元收益，占总增长的2.9%",
    "未能达成目标，还差214,568元"
  ]
}
```

#### POST /api/v1/simulation/goal-planning

Calculate required monthly savings to reach a goal.

**Parameters:**
- `goal_amount` (float, required)
- `time_horizon_months` (integer, required)
- `current_savings` (float, required)
- `investment_return_rate` (float, optional, default=0.05)

**Response:**
```json
{
  "required_monthly_savings": 10842.33,
  "goal_amount": 500000.0,
  "time_horizon_months": 36,
  "is_achievable": true,
  "recommendations": [
    "建议每月储蓄10,842元以达成目标",
    "相当于每天储蓄约361元"
  ],
  "simulation": {...}
}
```

#### POST /api/v1/simulation/what-if

What-if scenario simulation.

**Parameters:**
- `user_id` (string, required)
- `scenario_name` (string, required)
- `monthly_savings_increase` (float, required)
- `time_horizon_years` (integer, optional, default=5)

---

### Statistics API

#### GET /api/v1/stats/graph

Get statistics about the user-product-scene graph.

**Response:**
```json
{
  "num_nodes": 1523,
  "num_edges": 4567,
  "num_users": 500,
  "num_products": 1000,
  "num_scenes": 23
}
```

---

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message description"
}
```

**HTTP Status Codes:**
- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

---

## Rate Limiting

Currently no rate limiting. In production, implement rate limiting based on API keys.

---

## Python Client Example

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# Create user profile
response = requests.post(
    f"{BASE_URL}/api/v1/profile/create",
    params={
        "user_id": "user_001",
        "age": 28,
        "monthly_income": 12000,
        "monthly_expenses": 8000,
        "savings": 50000,
        "risk_level": "moderate"
    }
)
profile = response.json()

# Get recommendations
response = requests.post(
    f"{BASE_URL}/api/v1/recommendations",
    json={"user_id": "user_001"},
    params={"top_k": 5}
)
recommendations = response.json()

# Simulate savings goal
response = requests.post(
    f"{BASE_URL}/api/v1/simulation/goal-planning",
    params={
        "goal_amount": 500000,
        "time_horizon_months": 36,
        "current_savings": 100000
    }
)
plan = response.json()
print(f"Required monthly savings: ¥{plan['required_monthly_savings']:,.0f}")
```

---

## WebSocket Support (Future)

Real-time features like streaming dialogue responses will be supported via WebSocket in future versions.

---

Last Updated: 2025-01-16
