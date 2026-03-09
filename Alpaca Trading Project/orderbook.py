"""
Order Book Implementation - Manage and match bid/ask orders using efficient data structures
Part 2, Step 2: Order Book Implementation
"""

import heapq
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass
class Order:
    order_id: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float
    timestamp: datetime
    filled_quantity: float = 0.0
    
    def __lt__(self, other):
        # For buy orders: higher price has priority, then earlier time
        # For sell orders: lower price has priority, then earlier time
        if self.side == OrderSide.BUY:
            if self.price != other.price:
                return self.price > other.price  # Higher price first
            return self.timestamp < other.timestamp  # Earlier time first
        else:  # SELL
            if self.price != other.price:
                return self.price < other.price  # Lower price first
            return self.timestamp < other.timestamp  # Earlier time first


class OrderBook:
    """Manage and match bid/ask orders using efficient data structures"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.buy_orders = []  # Max heap for buy orders
        self.sell_orders = []  # Min heap for sell orders
        self.orders_by_id = {}
        self.trade_history = []
        
    def add_order(self, order: Order) -> List[Dict]:
        """Add order to the book and attempt matching"""
        trades = []
        
        # Store order
        self.orders_by_id[order.order_id] = order
        
        if order.side == OrderSide.BUY:
            trades = self._match_buy_order(order)
            if order.quantity > order.filled_quantity:
                heapq.heappush(self.buy_orders, order)
        else:  # SELL
            trades = self._match_sell_order(order)
            if order.quantity > order.filled_quantity:
                heapq.heappush(self.sell_orders, order)
                
        return trades
    
    def _match_buy_order(self, buy_order: Order) -> List[Dict]:
        """Match buy order against sell orders"""
        trades = []
        
        while (self.sell_orders and 
               buy_order.quantity > buy_order.filled_quantity and
               (buy_order.order_type == OrderType.MARKET or 
                buy_order.price >= self.sell_orders[0].price)):
            
            sell_order = heapq.heappop(self.sell_orders)
            trade = self._execute_trade(buy_order, sell_order)
            trades.append(trade)
            
            # Put back sell order if not fully filled
            if sell_order.quantity > sell_order.filled_quantity:
                heapq.heappush(self.sell_orders, sell_order)
                
        return trades
    
    def _match_sell_order(self, sell_order: Order) -> List[Dict]:
        """Match sell order against buy orders"""
        trades = []
        
        while (self.buy_orders and 
               sell_order.quantity > sell_order.filled_quantity and
               (sell_order.order_type == OrderType.MARKET or 
                sell_order.price <= self.buy_orders[0].price)):
            
            buy_order = heapq.heappop(self.buy_orders)
            trade = self._execute_trade(buy_order, sell_order)
            trades.append(trade)
            
            # Put back buy order if not fully filled
            if buy_order.quantity > buy_order.filled_quantity:
                heapq.heappush(self.buy_orders, buy_order)
                
        return trades
    
    def _execute_trade(self, buy_order: Order, sell_order: Order) -> Dict:
        """Execute trade between two orders"""
        trade_quantity = min(
            buy_order.quantity - buy_order.filled_quantity,
            sell_order.quantity - sell_order.filled_quantity
        )
        
        trade_price = sell_order.price  # Use sell order price (first come, first served)
        
        # Update filled quantities
        buy_order.filled_quantity += trade_quantity
        sell_order.filled_quantity += trade_quantity
        
        trade = {
            'timestamp': datetime.now(),
            'buy_order_id': buy_order.order_id,
            'sell_order_id': sell_order.order_id,
            'quantity': trade_quantity,
            'price': trade_price,
            'symbol': self.symbol
        }
        
        self.trade_history.append(trade)
        return trade
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id in self.orders_by_id:
            order = self.orders_by_id[order_id]
            
            # Remove from appropriate heap (simplified - in practice, you'd mark as canceled)
            if order.side == OrderSide.BUY:
                self.buy_orders = [o for o in self.buy_orders if o.order_id != order_id]
                heapq.heapify(self.buy_orders)
            else:
                self.sell_orders = [o for o in self.sell_orders if o.order_id != order_id]
                heapq.heapify(self.sell_orders)
                
            del self.orders_by_id[order_id]
            return True
        
        return False
    
    def modify_order(self, order_id: str, new_quantity: float, new_price: float) -> bool:
        """Modify an existing order"""
        if order_id in self.orders_by_id:
            order = self.orders_by_id[order_id]
            
            # Cancel old order
            self.cancel_order(order_id)
            
            # Create new order with updated parameters
            modified_order = Order(
                order_id=order_id + "_mod",
                side=order.side,
                order_type=order.order_type,
                quantity=new_quantity,
                price=new_price,
                timestamp=datetime.now(),
                filled_quantity=order.filled_quantity
            )
            
            # Add modified order
            self.add_order(modified_order)
            return True
        
        return False
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price"""
        return self.buy_orders[0].price if self.buy_orders else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price"""
        return self.sell_orders[0].price if self.sell_orders else None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask - best_bid
        return None
    
    def get_market_depth(self, levels: int = 5) -> Dict:
        """Get market depth (top N levels of bids and asks)"""
        bids = sorted(self.buy_orders, key=lambda x: (-x.price, x.timestamp))[:levels]
        asks = sorted(self.sell_orders, key=lambda x: (x.price, x.timestamp))[:levels]
        
        return {
            'bids': [(order.price, order.quantity - order.filled_quantity) for order in bids],
            'asks': [(order.price, order.quantity - order.filled_quantity) for order in asks]
        }
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of an order"""
        if order_id in self.orders_by_id:
            order = self.orders_by_id[order_id]
            return {
                'order_id': order.order_id,
                'side': order.side.value,
                'type': order.order_type.value,
                'quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'remaining_quantity': order.quantity - order.filled_quantity,
                'price': order.price,
                'timestamp': order.timestamp,
                'status': 'FILLED' if order.filled_quantity >= order.quantity else 'PARTIAL' if order.filled_quantity > 0 else 'OPEN'
            }
        return None