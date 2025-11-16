"""
Financial Dialogue Engine with LLM integration
Supports natural language interaction for financial queries
"""
from typing import Optional, Dict, Any
import uuid
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils.config_loader import config
from utils.logger import setup_logger
from .dialogue_state_tracker import DialogueStateTracker
from .nlu_engine import NLUEngine

logger = setup_logger()


class FinancialDialogueEngine:
    """
    Main dialogue engine for financial conversations.
    Integrates LLM, NLU, and dialogue state tracking.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize dialogue engine
        
        Args:
            model_name: Name of the LLM model to use (e.g., Qwen/Qwen-7B)
        """
        self.model_name = model_name or config.get("dialogue_engine.model_name", "gpt2")
        self.temperature = config.get("dialogue_engine.temperature", 0.7)
        self.max_tokens = config.get("dialogue_engine.max_tokens", 512)
        self.max_history = config.get("dialogue_engine.max_history_length", 10)
        
        # Initialize components
        self.dst = DialogueStateTracker(max_history=self.max_history)
        self.nlu = NLUEngine()
        
        # Initialize LLM (lazy loading for efficiency)
        self.tokenizer = None
        self.model = None
        self._model_loaded = False
        
        logger.info(f"Initialized FinancialDialogueEngine with model: {self.model_name}")
    
    def _load_model(self):
        """Lazy load the LLM model"""
        if self._model_loaded:
            return
        
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self._model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load model {self.model_name}: {e}")
            logger.info("Using mock responses for demonstration")
    
    def process_query(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user query with context awareness
        
        Args:
            user_id: User identifier
            query: User's natural language query
            session_id: Optional session ID for continuing conversation
            
        Returns:
            Response dictionary with answer, intent, entities, and session info
        """
        # Create or retrieve session
        if not session_id:
            session_id = str(uuid.uuid4())
            self.dst.create_session(user_id, session_id)
        
        session = self.dst.get_session(session_id)
        if not session:
            session = self.dst.create_session(user_id, session_id)
        
        # Parse query with NLU
        intent, entities = self.nlu.parse_query(query)
        
        # Resolve contextual references
        context = self.dst.resolve_reference(session_id, query)
        
        # Generate response
        response = self._generate_response(query, intent, entities, context)
        
        # Update dialogue state
        self.dst.update_session(
            session_id=session_id,
            user_message=query,
            assistant_message=response,
            intent=intent,
            entities=entities
        )
        
        # Generate suggestions for next queries
        suggestions = self._generate_suggestions(intent, entities)
        
        logger.info(f"Processed query for user {user_id}, intent: {intent}")
        
        return {
            "response": response,
            "intent": intent,
            "entities": entities,
            "session_id": session_id,
            "suggestions": suggestions
        }
    
    def _generate_response(
        self,
        query: str,
        intent: str,
        entities: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate response using LLM or rule-based fallback
        
        Args:
            query: User query
            intent: Recognized intent
            entities: Extracted entities
            context: Dialogue context
            
        Returns:
            Generated response
        """
        # Try to use LLM if available
        if self._model_loaded and self.model and self.tokenizer:
            try:
                prompt = self.nlu.generate_context_aware_prompt(
                    query, intent, entities, context
                )
                return self._generate_with_llm(prompt)
            except Exception as e:
                logger.error(f"Error generating with LLM: {e}")
        
        # Fallback to rule-based responses
        return self._generate_rule_based_response(query, intent, entities, context)
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate response using LLM"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the new generation
        response = response[len(prompt):].strip()
        
        return response
    
    def _generate_rule_based_response(
        self,
        query: str,
        intent: str,
        entities: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate rule-based response based on intent"""
        
        response_templates = {
            "query_expense": "根据您的账单记录，{time_period}您在{category}方面的支出为{amount}元。这个支出{comparison}您的平均水平。建议您可以通过制定预算来更好地控制开支。",
            "savings_plan": "基于您的财务状况和{goal}目标，建议您每月储蓄{amount}元。采用自动储蓄计划可以帮助您更好地实现目标，我可以为您推荐一些自动储蓄产品。",
            "investment_advice": "根据您的风险承受能力和投资目标，建议您考虑{product_type}。具体来说，我推荐一个包含{allocation}的投资组合，预期年化收益率约为{return_rate}。",
            "financial_goal": "实现{goal}需要合理的财务规划。基于您目前的收入和支出情况，建议您：1) 制定储蓄计划 2) 优化支出结构 3) 适当投资增值。我可以为您生成详细的规划方案。",
            "risk_assessment": "根据您的年龄、收入和投资经验，您的风险承受能力评估为{risk_level}。这意味着您适合投资{suitable_products}类产品。",
            "portfolio_review": "您当前的投资组合包括：{portfolio}。整体风险水平为{risk}，近期收益率为{return}。建议{suggestion}。",
            "general_query": "我理解您的问题。作为您的AI理财助手，我可以帮您：1) 分析消费情况 2) 制定储蓄计划 3) 提供投资建议 4) 规划财务目标。您想了解哪方面的信息呢？"
        }
        
        template = response_templates.get(intent, response_templates["general_query"])
        
        # Simple placeholder filling (in production, use actual data)
        response = template.format(
            time_period=entities.get("time_period", ["这个月"])[0] if entities.get("time_period") else "这个月",
            category=entities.get("category", ["总"])[0] if entities.get("category") else "各项",
            amount=entities.get("amount", ["XXXX"])[0] if entities.get("amount") else "XXXX",
            comparison="接近",
            goal="目标",
            product_type="稳健型理财产品",
            allocation="60%债券基金 + 30%股票基金 + 10%货币基金",
            return_rate="5-7%",
            risk_level="中等",
            suitable_products="平衡型",
            portfolio="基金30%，股票20%，债券30%，现金20%",
            risk="中等",
            return_value="+5.2%",
            suggestion="可以适当增加股票配置以提升收益"
        )
        
        return response
    
    def _generate_suggestions(
        self,
        intent: str,
        entities: Dict[str, Any]
    ) -> list:
        """Generate follow-up suggestions based on current intent"""
        
        suggestions_map = {
            "query_expense": [
                "查看本月总支出",
                "对比上个月支出变化",
                "设置消费预算"
            ],
            "savings_plan": [
                "查看推荐的储蓄产品",
                "设置自动储蓄",
                "调整储蓄目标"
            ],
            "investment_advice": [
                "了解推荐产品详情",
                "评估投资风险",
                "查看历史收益"
            ],
            "general_query": [
                "查看我的账单",
                "制定储蓄计划",
                "获取投资建议"
            ]
        }
        
        return suggestions_map.get(intent, suggestions_map["general_query"])
    
    def clear_session(self, session_id: str):
        """Clear a dialogue session"""
        self.dst.clear_session(session_id)
