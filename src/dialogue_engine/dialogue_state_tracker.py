"""
Dialogue State Tracker (DST) for maintaining conversation context
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from models.schemas import DialogueContext
from utils.logger import setup_logger

logger = setup_logger()


class DialogueStateTracker:
    """
    Tracks dialogue state and maintains context across conversation turns.
    Implements context tracking for natural conversation flow.
    """
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.sessions: Dict[str, DialogueContext] = {}
    
    def create_session(self, user_id: str, session_id: str) -> DialogueContext:
        """Create a new dialogue session"""
        context = DialogueContext(
            user_id=user_id,
            session_id=session_id,
            messages=[],
            context_variables={}
        )
        self.sessions[session_id] = context
        logger.info(f"Created new session {session_id} for user {user_id}")
        return context
    
    def get_session(self, session_id: str) -> Optional[DialogueContext]:
        """Get existing dialogue session"""
        return self.sessions.get(session_id)
    
    def update_session(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None
    ) -> DialogueContext:
        """Update dialogue session with new turn"""
        context = self.sessions.get(session_id)
        if not context:
            raise ValueError(f"Session {session_id} not found")
        
        # Add user message
        context.messages.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add assistant message
        context.messages.append({
            "role": "assistant",
            "content": assistant_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(context.messages) > self.max_history * 2:
            context.messages = context.messages[-(self.max_history * 2):]
        
        # Update intent and entities
        if intent:
            context.current_intent = intent
        
        if entities:
            context.entities.update(entities)
        
        context.timestamp = datetime.now()
        
        logger.debug(f"Updated session {session_id} with intent: {intent}")
        return context
    
    def resolve_reference(
        self,
        session_id: str,
        current_query: str
    ) -> Dict[str, Any]:
        """
        Resolve contextual references in the current query.
        Example: "那投资呢?" -> resolve "投资" with previous context
        """
        context = self.sessions.get(session_id)
        if not context:
            return {}
        
        resolved_context = {
            "previous_intent": context.current_intent,
            "entities": context.entities.copy(),
            "context_variables": context.context_variables.copy()
        }
        
        # Simple reference resolution - can be enhanced with NLP
        if current_query and len(context.messages) > 0:
            # Get last few messages for context
            recent_context = context.messages[-4:] if len(context.messages) >= 4 else context.messages
            resolved_context["recent_messages"] = recent_context
        
        return resolved_context
    
    def set_context_variable(self, session_id: str, key: str, value: Any):
        """Set a context variable for the session"""
        context = self.sessions.get(session_id)
        if context:
            context.context_variables[key] = value
    
    def get_context_variable(self, session_id: str, key: str) -> Optional[Any]:
        """Get a context variable from the session"""
        context = self.sessions.get(session_id)
        if context:
            return context.context_variables.get(key)
        return None
    
    def clear_session(self, session_id: str):
        """Clear a dialogue session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session {session_id}")
