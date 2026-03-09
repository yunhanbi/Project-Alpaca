"""
Matching Engine Simulator - Simulate realistic order execution outcomes
Part 2, Step 4: Matching Engine Simulator
"""

import random
import numpy as np
from typing import Dict, List, Tuple
from enum import Enum
from datetime import datetime
from orderbook import Order, OrderType, OrderSide


class ExecutionOutcome(Enum):
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"


class MatchingEngine:
    """Simulate realistic order execution outcomes"""
    
    def __init__(self, fill_probability: float = 0.85, partial_fill_probability: float = 0.1, rejection_probability: float = 0.05):
        self.fill_probability = fill_probability
        self.partial_fill_probability = partial_fill_probability
        self.rejection_probability = rejection_probability
        self.execution_history = []
        
        # Ensure probabilities sum correctly
        total_prob = fill_probability + partial_fill_probability + rejection_probability
        if abs(total_prob - 1.0) > 0.001:
            print(f"Warning: Probabilities sum to {total_prob}, adjusting...")
            # Normalize probabilities
            self.fill_probability = fill_probability / total_prob
            self.partial_fill_probability = partial_fill_probability / total_prob
            self.rejection_probability = rejection_probability / total_prob
    
    def simulate_execution(self, order: Order, market_data: Dict) -> Dict:
        """Simulate order execution with random outcomes"""
        
        # Determine execution outcome randomly
        outcome_rand = random.random()
        
        if outcome_rand < self.rejection_probability:
            # Rejection (check first as it's usually rare)
            outcome = ExecutionOutcome.REJECTED
            filled_quantity = 0.0
            execution_price = 0.0
            rejection_reason = self._get_random_rejection_reason()
            
        elif outcome_rand < self.rejection_probability + self.partial_fill_probability:
            # Partial fill
            outcome = ExecutionOutcome.PARTIALLY_FILLED
            # Random fill between 10% and 90% of order quantity
            filled_quantity = order.quantity * random.uniform(0.1, 0.9)
            execution_price = self._calculate_execution_price(order, market_data)
            rejection_reason = None
            
        else:
            # Full fill
            outcome = ExecutionOutcome.FILLED
            filled_quantity = order.quantity
            execution_price = self._calculate_execution_price(order, market_data)
            rejection_reason = None
        
        # Add realistic latency simulation
        execution_latency = self._simulate_latency()
        
        execution_result = {
            'order_id': order.order_id,
            'outcome': outcome,
            'filled_quantity': filled_quantity,
            'remaining_quantity': order.quantity - filled_quantity,
            'execution_price': execution_price,
            'timestamp': datetime.now(),
            'latency_ms': execution_latency,
            'market_data': market_data,
            'rejection_reason': rejection_reason
        }
        
        self.execution_history.append(execution_result)
        return execution_result
    
    def _calculate_execution_price(self, order: Order, market_data: Dict) -> float:
        """Calculate realistic execution price based on market conditions"""
        current_price = market_data.get('close', order.price)
        
        if order.order_type == OrderType.MARKET:
            # Market orders get filled at current price with some slippage
            slippage_factor = self._calculate_slippage(order, market_data)
            
            if order.side == OrderSide.BUY:
                # Buy orders face positive slippage (pay more)
                return current_price * (1 + slippage_factor)
            else:
                # Sell orders face negative slippage (receive less)
                return current_price * (1 - slippage_factor)
        
        elif order.order_type == OrderType.LIMIT:
            # Limit orders get filled at limit price or better
            if order.side == OrderSide.BUY:
                # For buy limits, execution price can't exceed limit price
                return min(order.price, current_price)
            else:  # SELL
                # For sell limits, execution price can't be below limit price
                return max(order.price, current_price)
        
        return order.price
    
    def _calculate_slippage(self, order: Order, market_data: Dict) -> float:
        """Calculate slippage based on order size and market conditions"""
        base_slippage = 0.001  # 0.1% base slippage
        
        # Volume-based slippage
        current_volume = market_data.get('volume', 1000000)
        order_value = order.quantity * market_data.get('close', order.price)
        
        # Higher order value relative to volume increases slippage
        volume_impact = (order_value / current_volume) * 0.01
        
        # Volatility-based slippage (simplified)
        volatility_factor = random.uniform(0.5, 2.0)  # In practice, use actual volatility
        
        total_slippage = base_slippage + volume_impact * volatility_factor
        
        # Cap slippage at reasonable levels
        return min(total_slippage, 0.02)  # Maximum 2% slippage
    
    def _simulate_latency(self) -> float:
        """Simulate execution latency in milliseconds"""
        # Realistic latency distribution (mostly fast, occasionally slow)
        if random.random() < 0.9:
            # Normal execution: 1-10ms
            return random.uniform(1.0, 10.0)
        else:
            # Slow execution: 50-200ms
            return random.uniform(50.0, 200.0)
    
    def _get_random_rejection_reason(self) -> str:
        """Get a random but realistic rejection reason"""
        rejection_reasons = [
            "Insufficient liquidity",
            "Price outside allowed range",
            "Market closed",
            "Order size too large",
            "Technical error",
            "Risk limits exceeded",
            "Invalid order parameters"
        ]
        return random.choice(rejection_reasons)
    
    def batch_execute_orders(self, orders: List[Order], market_data: Dict) -> List[Dict]:
        """Execute multiple orders in batch"""
        results = []
        
        for order in orders:
            result = self.simulate_execution(order, market_data)
            results.append(result)
            
        return results
    
    def get_execution_statistics(self) -> Dict:
        """Get execution statistics"""
        if not self.execution_history:
            return {'message': 'No execution history available'}
        
        total_orders = len(self.execution_history)
        filled_orders = sum(1 for ex in self.execution_history if ex['outcome'] == ExecutionOutcome.FILLED)
        partial_orders = sum(1 for ex in self.execution_history if ex['outcome'] == ExecutionOutcome.PARTIALLY_FILLED)
        rejected_orders = sum(1 for ex in self.execution_history if ex['outcome'] == ExecutionOutcome.REJECTED)
        
        # Calculate average fill ratio for successful orders
        successful_orders = [ex for ex in self.execution_history if ex['filled_quantity'] > 0]
        if successful_orders:
            fill_ratios = [ex['filled_quantity'] / ex['filled_quantity'] + ex['remaining_quantity'] 
                          for ex in successful_orders]
            avg_fill_ratio = np.mean(fill_ratios)
        else:
            avg_fill_ratio = 0.0
        
        # Calculate average latency
        latencies = [ex['latency_ms'] for ex in self.execution_history]
        avg_latency = np.mean(latencies)
        
        return {
            'total_orders': total_orders,
            'fill_rate': filled_orders / total_orders,
            'partial_fill_rate': partial_orders / total_orders,
            'rejection_rate': rejected_orders / total_orders,
            'average_fill_ratio': avg_fill_ratio,
            'average_latency_ms': avg_latency,
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies)
        }
    
    def get_execution_history(self, order_id: str = None) -> List[Dict]:
        """Get execution history, optionally filtered by order ID"""
        if order_id:
            return [ex for ex in self.execution_history if ex['order_id'] == order_id]
        return self.execution_history.copy()
    
    def reset_statistics(self):
        """Reset execution statistics"""
        self.execution_history = []
    
    def update_execution_probabilities(self, fill_prob: float, partial_prob: float, reject_prob: float):
        """Update execution probabilities"""
        total = fill_prob + partial_prob + reject_prob
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Probabilities must sum to 1.0, got {total}")
            
        self.fill_probability = fill_prob
        self.partial_fill_probability = partial_prob
        self.rejection_probability = reject_prob
    
    def simulate_market_impact(self, order: Order, market_data: Dict) -> Dict:
        """Simulate market impact of large orders"""
        order_value = order.quantity * market_data.get('close', order.price)
        market_volume = market_data.get('volume', 1000000)
        
        # Calculate market impact as percentage of daily volume
        volume_percentage = order_value / market_volume
        
        # Larger orders have more market impact
        if volume_percentage > 0.1:  # Order is >10% of daily volume
            impact_factor = volume_percentage * 0.05  # 5% impact per 100% of volume
        else:
            impact_factor = 0.0
            
        return {
            'volume_percentage': volume_percentage,
            'price_impact_factor': impact_factor,
            'estimated_price_impact': market_data.get('close', order.price) * impact_factor
        }