"""
Order Manager - Validate and record orders before execution
Part 2, Step 3: Order Manager & Gateway (Order Manager component)
"""

from typing import Dict, List
from datetime import datetime, timedelta
from orderbook import Order, OrderSide


class OrderManager:
    """Validate and record orders before execution"""
    
    def __init__(self, initial_capital: float, risk_limits: Dict):
        self.capital = initial_capital
        self.available_capital = initial_capital
        self.risk_limits = risk_limits
        self.order_history = []
        self.position = 0.0  # Current position (positive = long, negative = short)
        self.orders_per_minute = []
        
    def validate_order(self, order: Order, current_price: float) -> Dict:
        """Validate order against capital and risk limits"""
        validation_result = {
            'valid': True,
            'errors': []
        }
        
        # Capital sufficiency check
        if order.side == OrderSide.BUY:
            required_capital = order.quantity * order.price
            if required_capital > self.available_capital:
                validation_result['valid'] = False
                validation_result['errors'].append(
                    f"Insufficient capital: Required {required_capital:.2f}, Available {self.available_capital:.2f}"
                )
        
        # Orders per minute limit
        current_time = datetime.now()
        recent_orders = [t for t in self.orders_per_minute if current_time - t < timedelta(minutes=1)]
        max_orders_per_minute = self.risk_limits.get('max_orders_per_minute', 10)
        
        if len(recent_orders) >= max_orders_per_minute:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Orders per minute limit exceeded: {len(recent_orders)}/{max_orders_per_minute}"
            )
        
        # Position limits
        new_position = self.position
        if order.side == OrderSide.BUY:
            new_position += order.quantity
        else:
            new_position -= order.quantity
            
        max_position = self.risk_limits.get('max_position', 1000)
        if abs(new_position) > max_position:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Position limit exceeded: New position {new_position}, Limit {max_position}"
            )
        
        # Maximum order size check
        max_order_size = self.risk_limits.get('max_order_size', self.capital * 0.1)
        order_value = order.quantity * order.price
        if order_value > max_order_size:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Order size too large: {order_value:.2f}, Maximum allowed: {max_order_size:.2f}"
            )
            
        # Price deviation check (prevent fat finger errors)
        max_price_deviation = self.risk_limits.get('max_price_deviation', 0.05)  # 5% default
        price_deviation = abs(order.price - current_price) / current_price
        if price_deviation > max_price_deviation:
            validation_result['valid'] = False
            validation_result['errors'].append(
                f"Price deviation too large: {price_deviation:.2%}, Maximum allowed: {max_price_deviation:.2%}"
            )
        
        return validation_result
    
    def record_order(self, order: Order, validation_result: Dict):
        """Record order in history"""
        order_record = {
            'timestamp': datetime.now(),
            'order': order,
            'validation': validation_result
        }
        self.order_history.append(order_record)
        
        if validation_result['valid']:
            self.orders_per_minute.append(datetime.now())
            # Reserve capital for buy orders
            if order.side == OrderSide.BUY:
                self.available_capital -= order.quantity * order.price
                
        return order_record
    
    def update_position(self, trade: Dict):
        """Update position and available capital after trade execution"""
        # Check if this was our order
        our_order_ids = [record['order'].order_id for record in self.order_history 
                        if record['validation']['valid']]
        
        if trade['buy_order_id'] in our_order_ids:
            # Our buy order was filled
            self.position += trade['quantity']
            # Capital was already reserved, now it's spent
            
        elif trade['sell_order_id'] in our_order_ids:
            # Our sell order was filled
            self.position -= trade['quantity']
            self.available_capital += trade['quantity'] * trade['price']
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        total_orders = len(self.order_history)
        valid_orders = sum(1 for record in self.order_history if record['validation']['valid'])
        
        return {
            'total_capital': self.capital,
            'available_capital': self.available_capital,
            'used_capital': self.capital - self.available_capital,
            'current_position': self.position,
            'total_orders': total_orders,
            'valid_orders': valid_orders,
            'rejected_orders': total_orders - valid_orders,
            'capital_utilization': (self.capital - self.available_capital) / self.capital
        }
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk exposure metrics"""
        current_time = datetime.now()
        
        # Orders in last minute
        recent_orders = [t for t in self.orders_per_minute if current_time - t < timedelta(minutes=1)]
        
        # Maximum position reached
        max_position_reached = max([abs(r.get('position_after', 0)) for r in self.order_history] + [0])
        
        # Capital at risk
        capital_at_risk = self.capital - self.available_capital
        
        return {
            'orders_last_minute': len(recent_orders),
            'max_orders_per_minute': self.risk_limits.get('max_orders_per_minute', 10),
            'current_position': abs(self.position),
            'max_position_limit': self.risk_limits.get('max_position', 1000),
            'max_position_reached': max_position_reached,
            'capital_at_risk': capital_at_risk,
            'capital_at_risk_pct': capital_at_risk / self.capital,
            'position_utilization': abs(self.position) / self.risk_limits.get('max_position', 1000)
        }
    
    def reset_position(self):
        """Reset position (for testing/simulation purposes)"""
        self.position = 0.0
        self.available_capital = self.capital
        
    def update_capital(self, new_capital: float):
        """Update available capital"""
        self.capital = new_capital
        self.available_capital = new_capital