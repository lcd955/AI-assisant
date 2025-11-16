"""
Natural Language Understanding module for intent recognition and entity extraction
"""
from typing import Dict, List, Tuple, Any
import re
from utils.logger import setup_logger

logger = setup_logger()


class NLUEngine:
    """
    Natural Language Understanding for financial queries.
    Handles intent recognition and entity extraction for Chinese queries.
    """
    
    def __init__(self):
        # Intent patterns (Chinese)
        self.intent_patterns = {
            "query_expense": [
                r"(上个月|这个月|去年|今年).*(花了|支出|消费).*多少",
                r"(餐饮|交通|购物|娱乐).*(花费|支出|消费)",
                r"看看.*(账单|花费|支出)"
            ],
            "savings_plan": [
                r"(存钱|储蓄|攒钱).*计划",
                r"每月.*存.*多少",
                r"如果.*买房.*存.*多少"
            ],
            "investment_advice": [
                r"(投资|理财).*建议",
                r"如何.*投资",
                r"推荐.*(基金|股票|理财产品)"
            ],
            "financial_goal": [
                r"(买房|购车|子女教育|退休).*目标",
                r"三年内.*买房",
                r"财务.*目标"
            ],
            "risk_assessment": [
                r"风险.*评估",
                r"投资.*风险",
                r"能承受.*风险"
            ],
            "portfolio_review": [
                r"(查看|看看).*投资组合",
                r"资产.*配置",
                r"持仓.*情况"
            ]
        }
        
        # Entity patterns
        self.entity_patterns = {
            "time_period": r"(上个月|这个月|去年|今年|本月|上月|本年|去年)",
            "category": r"(餐饮|交通|购物|娱乐|医疗|教育|住房|其他)",
            "amount": r"(\d+(?:\.\d+)?)[元万千百十]?",
            "duration": r"(\d+)[年个月天]",
            "product_type": r"(基金|股票|债券|理财产品|保险|国债)"
        }
    
    def recognize_intent(self, query: str) -> str:
        """
        Recognize user intent from query
        
        Args:
            query: User query in Chinese
            
        Returns:
            Intent label
        """
        query = query.lower().strip()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    logger.debug(f"Recognized intent: {intent} for query: {query}")
                    return intent
        
        logger.debug(f"No specific intent recognized for query: {query}, using default")
        return "general_query"
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract entities from query
        
        Args:
            query: User query in Chinese
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query)
            if matches:
                entities[entity_type] = matches if isinstance(matches, list) else [matches]
        
        logger.debug(f"Extracted entities: {entities} from query: {query}")
        return entities
    
    def parse_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse query to extract intent and entities
        
        Args:
            query: User query
            
        Returns:
            Tuple of (intent, entities)
        """
        intent = self.recognize_intent(query)
        entities = self.extract_entities(query)
        
        return intent, entities
    
    def generate_context_aware_prompt(
        self,
        query: str,
        intent: str,
        entities: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate context-aware prompt for LLM
        
        Args:
            query: Original user query
            intent: Recognized intent
            entities: Extracted entities
            context: Previous context information
            
        Returns:
            Enhanced prompt for LLM
        """
        prompt_parts = [
            "你是一位专业的金融顾问助手。请根据以下信息回答用户问题：\n",
            f"用户问题：{query}\n",
            f"问题类型：{intent}\n"
        ]
        
        if entities:
            prompt_parts.append(f"提取的信息：{entities}\n")
        
        if context.get("previous_intent"):
            prompt_parts.append(f"上一个问题的主题：{context['previous_intent']}\n")
        
        if context.get("entities"):
            prompt_parts.append(f"对话中的关键信息：{context['entities']}\n")
        
        prompt_parts.append("\n请提供专业、准确、易懂的回答。")
        
        return "".join(prompt_parts)
