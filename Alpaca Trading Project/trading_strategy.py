"""
BABA Algorithmic Trading Strategy
Implements momentum-based, ML signal generation, and sentiment analysis strategies
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class BABAAlgoTradingStrategy:
    def __init__(self, portfolio_value=100000, base_risk_per_trade=0.03):
        self.portfolio_value = portfolio_value
        self.base_risk_per_trade = base_risk_per_trade
        self.ml_model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.positions = []
    
    def sanitize_market_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure OHLCV columns are numeric and index is valid."""
        df = data.copy()
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with invalid index or missing market values
        df = df[df.index.notna()]
        df = df.dropna(subset=[c for c in numeric_cols if c in df.columns])
        
        return df
        
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical features for ML model"""
        df = self.sanitize_market_data(data)
        
        # Momentum features
        df['momentum_5'] = df['Close'].pct_change(5)
        df['momentum_10'] = df['Close'].pct_change(10)
        df['momentum_20'] = df['Close'].pct_change(20)
        
        # Technical indicators
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['macd'] = df['Close'].ewm(12).mean() - df['Close'].ewm(26).mean()
        df['bb_position'] = (df['Close'] - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).std() * 2)
        
        # Volume indicators
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Price action features
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        return df.dropna()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def momentum_signal(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum-based signals"""
        df = self.sanitize_market_data(data)
        
        # Short-term momentum (5-period)
        short_momentum = df['Close'].pct_change(5)
        # Medium-term momentum (20-period) 
        medium_momentum = df['Close'].pct_change(20)
        
        # Momentum strength
        momentum_strength = (short_momentum + medium_momentum) / 2
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        signals[momentum_strength > 0.015] = 1    # Buy signal (>1.5%)
        signals[momentum_strength < -0.015] = -1  # Sell signal (<-1.5%)
        
        return signals
    
    def train_ml_model(self, data: pd.DataFrame):
        """Train ML signal generation model"""
        df = self.prepare_features(data)
        
        # Create target variable (future returns)
        df['future_return'] = df['Close'].shift(-5).pct_change(5)
        df['target'] = np.where(df['future_return'] > 0.005, 1, 
                               np.where(df['future_return'] < -0.005, -1, 0))
        
        # Features for ML model
        feature_cols = ['momentum_5', 'momentum_10', 'momentum_20', 'rsi', 
                       'macd', 'bb_position', 'volume_ratio', 'high_low_ratio', 'close_position']
        
        # Prepare training data
        X = df[feature_cols].dropna()
        y = df.loc[X.index, 'target']
        
        # Remove neutral signals for training
        mask = y != 0
        X_train = X[mask]
        y_train = y[mask]
        
        if len(X_train) == 0:
            print("No training data available")
            return
        
        # Scale features and train
        X_scaled = self.scaler.fit_transform(X_train)
        self.ml_model.fit(X_scaled, y_train)
        self.is_trained = True
        
        print(f"ML Model trained on {len(X_train)} samples")
    
    def ml_signal(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Generate ML-based signals with confidence"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_ml_model first.")
        
        df = self.prepare_features(data)
        feature_cols = ['momentum_5', 'momentum_10', 'momentum_20', 'rsi', 
                       'macd', 'bb_position', 'volume_ratio', 'high_low_ratio', 'close_position']
        
        X = df[feature_cols]
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions = self.ml_model.predict(X_scaled)
        probabilities = self.ml_model.predict_proba(X_scaled)
        
        # Calculate confidence (max probability)
        confidence = np.max(probabilities, axis=1)
        
        # Apply confidence threshold (70%)
        signals = pd.Series(predictions, index=df.index)
        signals[confidence < 0.7] = 0
        
        confidence_series = pd.Series(confidence, index=df.index)
        
        return signals, confidence_series
    
    def sentiment_signal(self, data: pd.DataFrame, news_text: str = None) -> pd.Series:
        """Generate sentiment-based signals (simplified)"""
        # Simplified sentiment - in practice, you'd use news APIs
        signals = pd.Series(0, index=data.index)
        
        if news_text:
            blob = TextBlob(news_text)
            sentiment_score = blob.sentiment.polarity
            
            if sentiment_score > 0.2:
                signals[:] = 1
            elif sentiment_score < -0.2:
                signals[:] = -1
        else:
            # Default neutral for demo
            signals[:] = 0
        
        return signals
    
    def generate_signals(self, data: pd.DataFrame, news_text: str = None) -> Dict:
        """Generate combined trading signals"""
        clean_data = self.sanitize_market_data(data)
        
        # Get individual signals
        momentum_sig = self.momentum_signal(clean_data)
        ml_sig, ml_conf = self.ml_signal(clean_data)
        sentiment_sig = self.sentiment_signal(clean_data, news_text)
        
        # Align all signals to the same index
        common_index = clean_data.index
        momentum_sig = momentum_sig.reindex(common_index, fill_value=0)
        ml_sig = ml_sig.reindex(common_index, fill_value=0)
        ml_conf = ml_conf.reindex(common_index, fill_value=0.0)
        sentiment_sig = sentiment_sig.reindex(common_index, fill_value=0)
        
        # Combine signals (confluence approach)
        combined_signals = pd.Series(0, index=common_index)
        signal_strength = pd.Series(0, index=common_index)
        
        for i in common_index:
            signals = [momentum_sig.loc[i], ml_sig.loc[i], sentiment_sig.loc[i]]
            
            # Count agreeing signals
            buy_votes = sum(1 for s in signals if s > 0)
            sell_votes = sum(1 for s in signals if s < 0)
            
            # Signal generation logic (2/3 confluence)
            if buy_votes >= 2:
                combined_signals.loc[i] = 1
                signal_strength.loc[i] = buy_votes
            elif sell_votes >= 2:
                combined_signals.loc[i] = -1
                signal_strength.loc[i] = sell_votes
        
        return {
            'momentum': momentum_sig,
            'ml': ml_sig,
            'ml_confidence': ml_conf,
            'sentiment': sentiment_sig,
            'combined': combined_signals,
            'strength': signal_strength
        }
    
    def calculate_position_size(self, signal_strength: int, current_volatility: float, avg_volatility: float) -> float:
        """Calculate dynamic position size"""
        # Base position size based on signal strength
        if signal_strength == 3:
            base_size = 0.04  # 4% for strong signals
        elif signal_strength == 2:
            base_size = 0.025  # 2.5% for medium signals
        else:
            base_size = 0.01   # 1% for weak signals
        
        # Volatility adjustment
        if current_volatility > 0:
            vol_adjustment = avg_volatility / current_volatility
        else:
            vol_adjustment = 1.0
            
        adjusted_size = base_size * vol_adjustment
        
        # Cap at maximum risk
        max_size = 0.05  # 5% maximum
        final_size = min(adjusted_size, max_size)
        
        return final_size * self.portfolio_value
    
    def execute_strategy(self, data: pd.DataFrame, news_text: str = None) -> pd.DataFrame:
        """Execute complete trading strategy"""
        clean_data = self.sanitize_market_data(data)
        
        # Generate signals
        signals = self.generate_signals(clean_data, news_text)
        
        # Calculate volatility metrics
        returns = clean_data['Close'].pct_change()
        current_vol = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else returns.std()
        avg_vol = returns.std()
        
        # Create trading log
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        
        for timestamp, row in clean_data.iterrows():
            signal = signals['combined'].loc[timestamp]
            strength = signals['strength'].loc[timestamp]
            
            current_price = row['Close']
            
            # Entry logic
            if signal != 0 and position == 0:
                position_size = self.calculate_position_size(strength, current_vol, avg_vol)
                position = signal
                entry_price = current_price
                entry_time = timestamp
                
                trades.append({
                    'timestamp': timestamp,
                    'action': 'ENTRY',
                    'signal': 'BUY' if signal > 0 else 'SELL',
                    'price': current_price,
                    'position_size': position_size,
                    'strength': strength
                })
            
            # Exit logic
            elif position != 0:
                # Calculate P&L
                if position > 0:
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                # Profit target (4%)
                if pnl_pct >= 0.04:
                    should_exit = True
                    exit_reason = "PROFIT_TARGET"
                
                # Stop loss (2.5%)
                elif pnl_pct <= -0.025:
                    should_exit = True
                    exit_reason = "STOP_LOSS"
                
                # Signal reversal
                elif (position > 0 and signal < 0) or (position < 0 and signal > 0):
                    should_exit = True
                    exit_reason = "SIGNAL_REVERSAL"
                
                # Time-based exit (5 hours = 300 minutes)
                elif (timestamp - entry_time).total_seconds() / 60 > 300:
                    should_exit = True
                    exit_reason = "TIME_EXIT"
                
                if should_exit:
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'EXIT',
                        'signal': exit_reason,
                        'price': current_price,
                        'pnl_pct': pnl_pct,
                        'hold_time_min': (timestamp - entry_time).total_seconds() / 60
                    })
                    position = 0
                    entry_price = 0
                    entry_time = None
        
        return pd.DataFrame(trades)