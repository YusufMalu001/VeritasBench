from typing import Dict, List, Tuple, Any
import math

class EloSystem:
    def __init__(self, k_factor: float = 32.0, initial_rating: float = 1200.0):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
        self.matches: Dict[str, int] = {}
        self.wins: Dict[str, int] = {}
        
    def add_model(self, model_name: str) -> None:
        if model_name not in self.ratings:
            self.ratings[model_name] = self.initial_rating
            self.matches[model_name] = 0
            self.wins[model_name] = 0
            
    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
        
    def record_match(self, model_a: str, model_b: str, result: str) -> None:
        """
        result: 'A', 'B', or 'Tie'
        """
        self.add_model(model_a)
        self.add_model(model_b)
        
        rating_a = self.ratings[model_a]
        rating_b = self.ratings[model_b]
        
        expected_a = self._expected_score(rating_a, rating_b)
        expected_b = self._expected_score(rating_b, rating_a)
        
        if result == 'A':
            score_a, score_b = 1.0, 0.0
            self.wins[model_a] += 1
        elif result == 'B':
            score_a, score_b = 0.0, 1.0
            self.wins[model_b] += 1
        else: # Tie
            score_a, score_b = 0.5, 0.5
            
        self.ratings[model_a] = rating_a + self.k_factor * (score_a - expected_a)
        self.ratings[model_b] = rating_b + self.k_factor * (score_b - expected_b)
        
        self.matches[model_a] += 1
        self.matches[model_b] += 1
        
    def get_leaderboard(self) -> List[Dict[str, Any]]:
        board = []
        for model, rating in sorted(self.ratings.items(), key=lambda x: x[1], reverse=True):
            matches = self.matches[model]
            win_rate = (self.wins[model] / matches * 100) if matches > 0 else 0.0
            board.append({
                "model": model,
                "rating": round(rating, 2),
                "matches": matches,
                "win_rate": round(win_rate, 2)
            })
        return board
