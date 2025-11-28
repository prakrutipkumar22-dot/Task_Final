"""
models.py - Machine Learning Models for Game Prediction

Trains models to predict wins/losses and analyze game outcomes.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

class WinPredictor:
    """Predicts win/loss based on game statistics."""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for model training.
        
        Features used:
        - Shooting: FG%, 3PT%, FT%
        - Ball control: Turnovers, Assists
        - Rebounding: Total rebounds
        - Defense: Opponent FG%
        - Relative metrics: Shooting differential, turnover differential
        """
        self.feature_names = [
            'fg_pct', 'three_pt_pct', 'ft_pct',
            'rebounds', 'assists', 'turnovers', 'steals',
            'opp_fg_pct', 'opp_three_pt_pct',
            'shooting_diff', 'turnover_diff', 'rebound_diff'
        ]
        
        # Create derived features
        df = df.copy()
        df['shooting_diff'] = df['fg_pct'] - df['opp_fg_pct']
        df['turnover_diff'] = df['opp_turnovers'] - df['turnovers']  # Positive is good
        df['rebound_diff'] = df['rebounds'] - df['opp_rebounds']
        
        X = df[self.feature_names].values
        y = df['win'].values
        
        return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train the win prediction model.
        
        Args:
            df: DataFrame with game data
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare data
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=random_state,
                class_weight='balanced'
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state
            )
        else:  # logistic regression (baseline)
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                class_weight='balanced'
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='accuracy'
        )
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        
        results = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True),
            'feature_importance': self.feature_importance
        }
        
        return results
    
    def predict(self, game_stats: Dict) -> Dict:
        """
        Predict win/loss for a game.
        
        Args:
            game_stats: Dictionary with game statistics
            
        Returns:
            Prediction and probability
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        features = []
        for feat in self.feature_names:
            if feat in ['shooting_diff', 'turnover_diff', 'rebound_diff']:
                # Calculate derived features
                if feat == 'shooting_diff':
                    features.append(game_stats['fg_pct'] - game_stats['opp_fg_pct'])
                elif feat == 'turnover_diff':
                    features.append(game_stats['opp_turnovers'] - game_stats['turnovers'])
                elif feat == 'rebound_diff':
                    features.append(game_stats['rebounds'] - game_stats['opp_rebounds'])
            else:
                features.append(game_stats[feat])
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return {
            'prediction': 'Win' if prediction == 1 else 'Loss',
            'win_probability': probability[1],
            'loss_probability': probability[0]
        }
    
    def get_top_features(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        if self.feature_importance is None:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_features[:n]
    
    def save(self, filepath: str = 'models/win_predictor.pkl'):
        """Save trained model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str = 'models/win_predictor.pkl'):
        """Load trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filepath}")


class GameQualityClassifier:
    """Classifies games as good/average/poor performance."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.thresholds = None
        
    def classify_games(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify each game as good/average/poor based on performance.
        
        Uses multiple factors:
        - Point differential
        - Shooting efficiency
        - Turnover control
        - Win/loss result
        """
        # Calculate performance score
        scores = pd.DataFrame()
        
        # Shooting score (0-3)
        scores['shooting'] = (
            (df['fg_pct'] >= 0.45).astype(int) +
            (df['three_pt_pct'] >= 0.35).astype(int) +
            (df['ft_pct'] >= 0.75).astype(int)
        )
        
        # Turnover score (0-2)
        scores['turnovers'] = (
            (df['turnovers'] <= 12).astype(int) +
            (df['turnovers'] <= 10).astype(int)
        )
        
        # Result score (0-3)
        scores['result'] = (
            (df['win'] == 1).astype(int) * 2 +
            (df['point_diff'] >= 0).astype(int)
        )
        
        # Defense score (0-2)
        scores['defense'] = (
            (df['opp_fg_pct'] <= 0.45).astype(int) +
            (df['opp_fg_pct'] <= 0.40).astype(int)
        )
        
        # Total score (0-10)
        total_score = scores.sum(axis=1)
        
        # Classify
        classifications = pd.cut(
            total_score,
            bins=[0, 4, 7, 10],
            labels=['Poor', 'Average', 'Good'],
            include_lowest=True
        )
        
        return classifications
    
    def get_performance_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary of team performance quality."""
        classifications = self.classify_games(df)
        
        return {
            'good_games': (classifications == 'Good').sum(),
            'average_games': (classifications == 'Average').sum(),
            'poor_games': (classifications == 'Poor').sum(),
            'good_pct': (classifications == 'Good').mean(),
            'avg_stats_by_quality': df.groupby(classifications).mean().to_dict()
        }


# Example usage
if __name__ == "__main__":
    from data_loader import GameDataLoader
    
    # Load data
    loader = GameDataLoader()
    games = loader.load_game_data()
    
    print("="*60)
    print("TRAINING WIN PREDICTION MODEL")
    print("="*60)
    
    # Train model
    predictor = WinPredictor(model_type='random_forest')
    results = predictor.train(games)
    
    print(f"\nTrain Accuracy: {results['train_accuracy']:.3f}")
    print(f"Test Accuracy: {results['test_accuracy']:.3f}")
    print(f"Cross-Validation: {results['cv_mean']:.3f} (+/- {results['cv_std']:.3f})")
    
    print("\nTop 5 Most Important Features:")
    for feat, importance in predictor.get_top_features(5):
        print(f"  {feat}: {importance:.3f}")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Save model
    predictor.save()
    
    # Test prediction on Tennessee game
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION: Tennessee Game")
    print("="*60)
    
    tennessee_game = games[games['opponent'] == 'Tennessee'].iloc[0]
    prediction = predictor.predict(tennessee_game.to_dict())
    
    print(f"Predicted: {prediction['prediction']}")
    print(f"Win Probability: {prediction['win_probability']:.1%}")
    print(f"Actual Result: {'Win' if tennessee_game['win'] == 1 else 'Loss'}")
    
    # Game quality classification
    print("\n" + "="*60)
    print("GAME QUALITY ANALYSIS")
    print("="*60)
    
    classifier = GameQualityClassifier()
    performance = classifier.get_performance_summary(games)
    
    print(f"\nGood Games: {performance['good_games']}")
    print(f"Average Games: {performance['average_games']}")
    print(f"Poor Games: {performance['poor_games']}")
    print(f"Good Game Rate: {performance['good_pct']:.1%}")
