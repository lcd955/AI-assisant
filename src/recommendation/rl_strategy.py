"""
Reinforcement Learning based Recommendation Strategy
Uses PPO algorithm to dynamically optimize recommendations based on user feedback
"""
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Tuple
from utils.logger import setup_logger
from utils.config_loader import config

logger = setup_logger()


class RecommendationEnvironment(gym.Env):
    """
    Custom Gym environment for financial product recommendations.
    The agent learns to recommend products based on user feedback.
    """
    
    def __init__(
        self,
        num_products: int = 100,
        state_dim: int = 50
    ):
        """
        Initialize recommendation environment
        
        Args:
            num_products: Number of available financial products
            state_dim: Dimension of state representation
        """
        super(RecommendationEnvironment, self).__init__()
        
        self.num_products = num_products
        self.state_dim = state_dim
        
        # Action space: select a product to recommend
        self.action_space = spaces.Discrete(num_products)
        
        # Observation space: user profile + context features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # State variables
        self.current_user_state = None
        self.recommended_products = []
        self.step_count = 0
        self.max_steps = 10
        
        logger.info(f"Initialized RecommendationEnvironment with {num_products} products")
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Generate random user state (in production, use actual user profile)
        self.current_user_state = np.random.randn(self.state_dim).astype(np.float32)
        self.recommended_products = []
        self.step_count = 0
        
        return self.current_user_state, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Product ID to recommend
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.step_count += 1
        
        # Simulate user feedback (in production, use actual feedback)
        # Feedback types: click (0.3), adopt (0.7), ignore (-0.1)
        feedback_type = np.random.choice(
            ['click', 'adopt', 'ignore'],
            p=[0.3, 0.3, 0.4]
        )
        
        # Calculate reward based on feedback
        if feedback_type == 'adopt':
            reward = 1.0
        elif feedback_type == 'click':
            reward = 0.3
        else:  # ignore
            reward = -0.1
        
        # Penalize duplicate recommendations
        if action in self.recommended_products:
            reward -= 0.5
        
        self.recommended_products.append(action)
        
        # Update state (simulate state evolution)
        noise = np.random.randn(self.state_dim) * 0.1
        self.current_user_state = self.current_user_state + noise
        self.current_user_state = self.current_user_state.astype(np.float32)
        
        # Check if episode is done
        terminated = self.step_count >= self.max_steps
        truncated = False
        
        info = {
            'feedback': feedback_type,
            'product_id': action,
            'step': self.step_count
        }
        
        return self.current_user_state, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (for debugging)"""
        pass


class RLRecommendationEngine:
    """
    Reinforcement Learning based recommendation engine using PPO
    """
    
    def __init__(
        self,
        num_products: int = 100,
        state_dim: int = 50,
        model_path: Optional[str] = None
    ):
        """
        Initialize RL recommendation engine
        
        Args:
            num_products: Number of products in catalog
            state_dim: Dimension of state representation
            model_path: Path to pre-trained model (optional)
        """
        self.num_products = num_products
        self.state_dim = state_dim
        
        # Create environment
        self.env = DummyVecEnv([lambda: RecommendationEnvironment(num_products, state_dim)])
        
        # Initialize PPO agent
        learning_rate = config.get("recommendation.rl.learning_rate", 0.0003)
        gamma = config.get("recommendation.rl.gamma", 0.99)
        
        if model_path:
            try:
                self.model = PPO.load(model_path, env=self.env)
                logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}, creating new model")
                self._create_new_model(learning_rate, gamma)
        else:
            self._create_new_model(learning_rate, gamma)
    
    def _create_new_model(self, learning_rate: float, gamma: float):
        """Create a new PPO model"""
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            gamma=gamma,
            verbose=1
        )
        logger.info("Created new PPO model")
    
    def train(self, total_timesteps: int = 10000):
        """
        Train the RL agent
        
        Args:
            total_timesteps: Number of timesteps to train
        """
        logger.info(f"Starting training for {total_timesteps} timesteps")
        self.model.learn(total_timesteps=total_timesteps)
        logger.info("Training completed")
    
    def recommend(
        self,
        user_state: np.ndarray,
        top_k: int = 5
    ) -> List[int]:
        """
        Generate top-k product recommendations
        
        Args:
            user_state: User state vector
            top_k: Number of recommendations to generate
            
        Returns:
            List of recommended product IDs
        """
        recommendations = []
        
        # Get predictions for multiple steps
        obs = user_state.reshape(1, -1)
        
        for _ in range(top_k):
            action, _ = self.model.predict(obs, deterministic=True)
            product_id = int(action[0])
            
            if product_id not in recommendations:
                recommendations.append(product_id)
            
            # Simulate state update (in production, update based on actual context)
            obs = obs + np.random.randn(1, self.state_dim) * 0.1
        
        return recommendations
    
    def update_from_feedback(
        self,
        user_state: np.ndarray,
        action: int,
        feedback: str
    ):
        """
        Update model based on user feedback
        
        Args:
            user_state: User state when recommendation was made
            action: Recommended product ID
            feedback: User feedback ('click', 'adopt', 'ignore')
        """
        # Convert feedback to reward
        reward_map = {
            'adopt': 1.0,
            'click': 0.3,
            'ignore': -0.1
        }
        reward = reward_map.get(feedback, 0.0)
        
        # In a full implementation, this would update the model
        # For now, we log the feedback for future training
        logger.debug(f"Received feedback: {feedback} for action {action}, reward: {reward}")
    
    def save_model(self, path: str):
        """Save the trained model"""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = PPO.load(path, env=self.env)
        logger.info(f"Model loaded from {path}")


# Add Optional import
from typing import Optional
