"""
Graph Neural Network for User-Product-Scene relationship modeling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data
import networkx as nx
from typing import Dict, List, Tuple, Any
from utils.logger import setup_logger
from utils.config_loader import config

logger = setup_logger()


class UserProductSceneGNN(nn.Module):
    """
    Graph Neural Network for modeling relationships between users, products, and scenes.
    Helps identify potential user needs and preferences.
    """
    
    def __init__(
        self,
        num_users: int,
        num_products: int,
        num_scenes: int,
        embedding_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        """
        Initialize GNN model
        
        Args:
            num_users: Number of unique users
            num_products: Number of financial products
            num_scenes: Number of financial scenes/contexts
            embedding_dim: Dimension of node embeddings
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(UserProductSceneGNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Node embeddings for different entity types
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)
        self.scene_embedding = nn.Embedding(num_scenes, embedding_dim)
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(embedding_dim, embedding_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(embedding_dim, embedding_dim))
        
        self.dropout = dropout
        
        # Output layer for link prediction
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
        logger.info(f"Initialized GNN with {num_layers} layers, embedding_dim={embedding_dim}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through GNN
        
        Args:
            x: Node features [num_nodes, embedding_dim]
            edge_index: Edge connections [2, num_edges]
            edge_type: Type of each edge [num_edges]
            
        Returns:
            Updated node embeddings
        """
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def predict_link(
        self,
        node_embeddings: torch.Tensor,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict link probability between source and destination nodes
        
        Args:
            node_embeddings: Node embeddings from forward pass
            src_nodes: Source node indices
            dst_nodes: Destination node indices
            
        Returns:
            Link probabilities
        """
        src_emb = node_embeddings[src_nodes]
        dst_emb = node_embeddings[dst_nodes]
        
        # Concatenate embeddings
        link_emb = torch.cat([src_emb, dst_emb], dim=-1)
        
        # Predict link probability
        return self.predictor(link_emb)


class UserGraphBuilder:
    """
    Builds and manages the user-product-scene relationship graph
    """
    
    def __init__(self):
        """Initialize graph builder"""
        self.graph = nx.MultiDiGraph()
        self.user_to_idx = {}
        self.product_to_idx = {}
        self.scene_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_product = {}
        self.idx_to_scene = {}
        
        logger.info("UserGraphBuilder initialized")
    
    def add_user(self, user_id: str, features: Dict[str, Any]):
        """Add user node to graph"""
        if user_id not in self.user_to_idx:
            idx = len(self.user_to_idx)
            self.user_to_idx[user_id] = idx
            self.idx_to_user[idx] = user_id
        
        self.graph.add_node(
            user_id,
            type='user',
            **features
        )
    
    def add_product(self, product_id: str, features: Dict[str, Any]):
        """Add product node to graph"""
        if product_id not in self.product_to_idx:
            idx = len(self.product_to_idx)
            self.product_to_idx[product_id] = idx
            self.idx_to_product[idx] = product_id
        
        self.graph.add_node(
            product_id,
            type='product',
            **features
        )
    
    def add_scene(self, scene_id: str, features: Dict[str, Any]):
        """Add scene node to graph"""
        if scene_id not in self.scene_to_idx:
            idx = len(self.scene_to_idx)
            self.scene_to_idx[scene_id] = idx
            self.idx_to_scene[idx] = scene_id
        
        self.graph.add_node(
            scene_id,
            type='scene',
            **features
        )
    
    def add_interaction(
        self,
        user_id: str,
        product_id: str,
        scene_id: str,
        interaction_type: str,
        weight: float = 1.0
    ):
        """
        Add interaction edges between user, product, and scene
        
        Args:
            user_id: User identifier
            product_id: Product identifier
            scene_id: Scene identifier
            interaction_type: Type of interaction (view, purchase, etc.)
            weight: Edge weight
        """
        # User -> Product
        self.graph.add_edge(
            user_id,
            product_id,
            type=f'user_product_{interaction_type}',
            weight=weight
        )
        
        # User -> Scene
        self.graph.add_edge(
            user_id,
            scene_id,
            type='user_scene',
            weight=weight
        )
        
        # Product -> Scene
        self.graph.add_edge(
            product_id,
            scene_id,
            type='product_scene',
            weight=weight
        )
    
    def to_pytorch_geometric(self) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data format
        
        Returns:
            PyTorch Geometric Data object
        """
        # Create node feature matrix (can be enhanced with actual features)
        num_nodes = self.graph.number_of_nodes()
        x = torch.randn(num_nodes, 128)  # Placeholder features
        
        # Create edge index
        edge_index = []
        edge_attr = []
        
        node_to_idx = {}
        for i, node in enumerate(self.graph.nodes()):
            node_to_idx[node] = i
        
        for u, v, data in self.graph.edges(data=True):
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_attr.append(data.get('weight', 1.0))
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def get_neighbors(self, node_id: str, hop: int = 1) -> List[str]:
        """
        Get k-hop neighbors of a node
        
        Args:
            node_id: Node identifier
            hop: Number of hops
            
        Returns:
            List of neighbor node IDs
        """
        if node_id not in self.graph:
            return []
        
        neighbors = set()
        current_level = {node_id}
        
        for _ in range(hop):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.successors(node))
            neighbors.update(next_level)
            current_level = next_level
        
        return list(neighbors - {node_id})
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_users": len(self.user_to_idx),
            "num_products": len(self.product_to_idx),
            "num_scenes": len(self.scene_to_idx)
        }
