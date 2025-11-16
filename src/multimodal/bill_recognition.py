"""
Multi-modal Bill Recognition Module
Supports image upload and automatic bill classification
"""
from typing import Dict, List, Any, Optional, Tuple
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from datetime import datetime
from models.schemas import Transaction
from utils.logger import setup_logger

logger = setup_logger()


class BillRecognitionEngine:
    """
    Bill recognition engine using OCR and classification
    Supports automatic bill information extraction from images
    """
    
    def __init__(self):
        """Initialize bill recognition engine"""
        self.ocr_available = False
        self.classifier_available = False
        
        # Try to import OCR libraries
        try:
            import pytesseract
            self.ocr_engine = pytesseract
            self.ocr_available = True
            logger.info("OCR engine initialized successfully")
        except ImportError:
            logger.warning("pytesseract not available, OCR features disabled")
        
        # Initialize category classifier (simplified version)
        self.category_keywords = {
            "餐饮": ["餐厅", "饭店", "外卖", "美团", "饿了么", "麦当劳", "肯德基", "星巴克"],
            "交通": ["地铁", "公交", "出租车", "滴滴", "加油", "停车", "高速"],
            "购物": ["超市", "商场", "淘宝", "京东", "拼多多", "服装", "电器"],
            "娱乐": ["电影", "KTV", "游戏", "健身", "旅游", "酒店"],
            "医疗": ["医院", "药店", "体检", "挂号"],
            "教育": ["学校", "培训", "书店", "网课"],
            "住房": ["房租", "物业", "水电", "燃气", "宽带"],
            "其他": []
        }
        
        logger.info("BillRecognitionEngine initialized")
    
    def recognize_bill_from_image(
        self,
        image_data: bytes,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Recognize bill information from image
        
        Args:
            image_data: Image data in bytes
            user_id: User identifier
            
        Returns:
            Dictionary with recognized information
        """
        try:
            # Load image
            image = Image.open(BytesIO(image_data))
            
            # Perform OCR
            text = self._extract_text_from_image(image)
            
            if not text:
                return {
                    "success": False,
                    "error": "No text found in image"
                }
            
            # Extract structured information
            bill_info = self._parse_bill_text(text)
            
            # Classify category
            category = self._classify_category(text)
            bill_info["category"] = category
            
            # Create transaction if enough info is available
            transaction = None
            if bill_info.get("amount") and bill_info.get("merchant"):
                transaction = Transaction(
                    id=f"bill_{user_id}_{datetime.now().timestamp()}",
                    user_id=user_id,
                    amount=bill_info["amount"],
                    category=category,
                    description=bill_info.get("merchant", ""),
                    timestamp=bill_info.get("date", datetime.now()),
                    merchant=bill_info.get("merchant")
                )
            
            return {
                "success": True,
                "recognized_text": text,
                "bill_info": bill_info,
                "transaction": transaction
            }
            
        except Exception as e:
            logger.error(f"Error recognizing bill: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using OCR"""
        if not self.ocr_available:
            logger.warning("OCR not available, using mock recognition")
            return self._mock_ocr_result()
        
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Perform OCR (configure for Chinese)
            text = self.ocr_engine.image_to_string(
                image,
                lang='chi_sim+eng'
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return ""
    
    def _mock_ocr_result(self) -> str:
        """Mock OCR result for demonstration"""
        return """
        星巴克咖啡
        2025-01-15 14:30
        拿铁咖啡 x2  60.00元
        小计: 60.00元
        """
    
    def _parse_bill_text(self, text: str) -> Dict[str, Any]:
        """
        Parse structured information from bill text
        
        Args:
            text: OCR recognized text
            
        Returns:
            Dictionary with parsed information
        """
        import re
        
        bill_info = {
            "merchant": None,
            "amount": None,
            "date": None,
            "items": []
        }
        
        # Extract amount (Chinese Yuan patterns)
        amount_patterns = [
            r'(\d+(?:\.\d{1,2})?)\s*元',
            r'¥\s*(\d+(?:\.\d{1,2})?)',
            r'RMB\s*(\d+(?:\.\d{1,2})?)',
            r'小计[:：]\s*(\d+(?:\.\d{1,2})?)',
            r'合计[:：]\s*(\d+(?:\.\d{1,2})?)'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    bill_info["amount"] = float(match.group(1))
                    break
                except ValueError:
                    continue
        
        # Extract date
        date_patterns = [
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{4}年\d{1,2}月\d{1,2}日)'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(1)
                try:
                    # Try to parse date
                    if '-' in date_str or '/' in date_str:
                        bill_info["date"] = datetime.strptime(
                            date_str.replace('/', '-'),
                            '%Y-%m-%d'
                        )
                    break
                except ValueError:
                    continue
        
        # Extract merchant (first line usually)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            bill_info["merchant"] = lines[0]
        
        return bill_info
    
    def _classify_category(self, text: str) -> str:
        """
        Classify bill category based on text content
        
        Args:
            text: Bill text
            
        Returns:
            Category label
        """
        text_lower = text.lower()
        
        # Check keywords for each category
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            return best_category[0]
        
        return "其他"
    
    def batch_recognize(
        self,
        image_list: List[bytes],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """
        Batch recognize multiple bills
        
        Args:
            image_list: List of image data in bytes
            user_id: User identifier
            
        Returns:
            List of recognition results
        """
        results = []
        
        for image_data in image_list:
            result = self.recognize_bill_from_image(image_data, user_id)
            results.append(result)
        
        return results
    
    def get_recognition_statistics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get statistics from batch recognition results
        
        Args:
            results: List of recognition results
            
        Returns:
            Statistics dictionary
        """
        stats = {
            "total": len(results),
            "successful": sum(1 for r in results if r.get("success")),
            "failed": sum(1 for r in results if not r.get("success")),
            "total_amount": 0.0,
            "categories": {}
        }
        
        for result in results:
            if result.get("success") and result.get("bill_info"):
                bill_info = result["bill_info"]
                
                # Add to total amount
                if bill_info.get("amount"):
                    stats["total_amount"] += bill_info["amount"]
                
                # Count categories
                category = bill_info.get("category", "其他")
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
        
        return stats
