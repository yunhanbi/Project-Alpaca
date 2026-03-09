"""
Order Gateway - Write all orders to file for audit and analysis
Part 2, Step 3: Order Manager & Gateway (Gateway component)
"""

import json
from typing import Dict, List
from datetime import datetime
from orderbook import Order


class OrderGateway:
    """Write all orders to file for audit and analysis"""
    
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.order_log = []
        
        # Initialize log file with header
        self._initialize_log_file()
        
    def _initialize_log_file(self):
        """Initialize log file with header information"""
        header = {
            'log_type': 'ORDER_GATEWAY_LOG',
            'created_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        try:
            with open(self.log_file, 'w') as f:
                f.write(json.dumps(header) + '\n')
        except Exception as e:
            print(f"Error initializing log file: {e}")
    
    def log_order_sent(self, order: Order):
        """Log when order is sent"""
        log_entry = {
            'action': 'SENT',
            'timestamp': datetime.now().isoformat(),
            'order_id': order.order_id,
            'side': order.side.value,
            'type': order.order_type.value,
            'quantity': order.quantity,
            'price': order.price,
            'original_timestamp': order.timestamp.isoformat()
        }
        
        self._write_log(log_entry)
    
    def log_order_modified(self, order_id: str, new_quantity: float, new_price: float, old_quantity: float = None, old_price: float = None):
        """Log when order is modified"""
        log_entry = {
            'action': 'MODIFIED',
            'timestamp': datetime.now().isoformat(),
            'order_id': order_id,
            'new_quantity': new_quantity,
            'new_price': new_price
        }
        
        # Include old values if provided
        if old_quantity is not None:
            log_entry['old_quantity'] = old_quantity
        if old_price is not None:
            log_entry['old_price'] = old_price
            
        self._write_log(log_entry)
    
    def log_order_cancelled(self, order_id: str, reason: str = None):
        """Log when order is cancelled"""
        log_entry = {
            'action': 'CANCELLED',
            'timestamp': datetime.now().isoformat(),
            'order_id': order_id
        }
        
        if reason:
            log_entry['reason'] = reason
            
        self._write_log(log_entry)
    
    def log_order_filled(self, trade: Dict):
        """Log when order is filled"""
        log_entry = {
            'action': 'FILLED',
            'timestamp': trade['timestamp'].isoformat() if isinstance(trade['timestamp'], datetime) else trade['timestamp'],
            'buy_order_id': trade['buy_order_id'],
            'sell_order_id': trade['sell_order_id'],
            'quantity': trade['quantity'],
            'price': trade['price'],
            'symbol': trade.get('symbol', 'UNKNOWN')
        }
        
        self._write_log(log_entry)
    
    def log_order_partially_filled(self, order_id: str, filled_quantity: float, remaining_quantity: float, fill_price: float):
        """Log when order is partially filled"""
        log_entry = {
            'action': 'PARTIALLY_FILLED',
            'timestamp': datetime.now().isoformat(),
            'order_id': order_id,
            'filled_quantity': filled_quantity,
            'remaining_quantity': remaining_quantity,
            'fill_price': fill_price
        }
        
        self._write_log(log_entry)
    
    def log_order_rejected(self, order: Order, rejection_reason: str):
        """Log when order is rejected"""
        log_entry = {
            'action': 'REJECTED',
            'timestamp': datetime.now().isoformat(),
            'order_id': order.order_id,
            'side': order.side.value,
            'type': order.order_type.value,
            'quantity': order.quantity,
            'price': order.price,
            'rejection_reason': rejection_reason
        }
        
        self._write_log(log_entry)
    
    def log_order_expired(self, order_id: str):
        """Log when order expires"""
        log_entry = {
            'action': 'EXPIRED',
            'timestamp': datetime.now().isoformat(),
            'order_id': order_id
        }
        
        self._write_log(log_entry)
    
    def _write_log(self, log_entry: Dict):
        """Write log entry to file"""
        self.order_log.append(log_entry)
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Error writing to log file: {e}")
    
    def get_log_summary(self) -> Dict:
        """Get summary statistics of logged activities"""
        if not self.order_log:
            return {'total_entries': 0}
            
        actions = {}
        for entry in self.order_log:
            action = entry.get('action', 'UNKNOWN')
            actions[action] = actions.get(action, 0) + 1
            
        return {
            'total_entries': len(self.order_log),
            'actions_breakdown': actions,
            'log_file': self.log_file
        }
    
    def get_order_timeline(self, order_id: str) -> List[Dict]:
        """Get timeline of all activities for a specific order"""
        return [entry for entry in self.order_log if entry.get('order_id') == order_id]
    
    def export_logs(self, export_file: str = None) -> str:
        """Export all logs to a new file or return as JSON string"""
        if export_file:
            try:
                with open(export_file, 'w') as f:
                    for entry in self.order_log:
                        f.write(json.dumps(entry, indent=2) + '\n')
                return f"Logs exported to {export_file}"
            except Exception as e:
                return f"Error exporting logs: {e}"
        else:
            return json.dumps(self.order_log, indent=2)
    
    def clear_logs(self):
        """Clear all logs (use with caution)"""
        self.order_log = []
        self._initialize_log_file()
        
    def search_logs(self, filter_criteria: Dict) -> List[Dict]:
        """Search logs based on filter criteria"""
        results = []
        
        for entry in self.order_log:
            match = True
            for key, value in filter_criteria.items():
                if key in entry and entry[key] != value:
                    match = False
                    break
            if match:
                results.append(entry)
                
        return results