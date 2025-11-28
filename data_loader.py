import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class GameDataLoader:
    """Loads and preprocesses Syracuse basketball game data."""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.games_df = None
        self.players_df = None
        
    def load_game_data(self, filename: str = 'syracuse_games_2024_25.csv') -> pd.DataFrame:
        """
        Load game-by-game data for Syracuse 2024-25 season.
        
        Args:
            filename: CSV file with game data
            
        Returns:
            DataFrame with game-level statistics
        """
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found. Creating sample data...")
            self.games_df = self._create_sample_data()
        else:
            self.games_df = pd.read_csv(filepath)
        
        # Clean and validate
        self.games_df = self._clean_game_data(self.games_df)
        
        return self.games_df
    
    def load_player_data(self, filename: str = 'syracuse_player_stats.csv') -> pd.DataFrame:
        """Load player-level statistics."""
        filepath = self.data_dir / filename
        
        if filepath.exists():
            self.players_df = pd.read_csv(filepath)
            return self.players_df
        else:
            print(f"Warning: {filepath} not found.")
            return pd.DataFrame()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create sample game data for Syracuse 2024-25 season.
        
        Based on actual season results: 14-19 record, 7-13 in ACC.
        """
        games = [
            # Non-conference games (early season)
            {
                'date': '2024-11-06', 'opponent': 'Binghamton', 'location': 'Home',
                'result': 'W', 'score': 86, 'opp_score': 77,
                'fg_pct': 0.458, 'three_pt_pct': 0.400, 'ft_pct': 0.750,
                'rebounds': 38, 'assists': 18, 'turnovers': 11, 'steals': 8,
                'opp_fg_pct': 0.421, 'opp_three_pt_pct': 0.318, 'opp_rebounds': 32,
                'opp_assists': 12, 'opp_turnovers': 14, 'opp_steals': 6
            },
            {
                'date': '2024-11-09', 'opponent': 'Delaware State', 'location': 'Home',
                'result': 'W', 'score': 83, 'opp_score': 43,
                'fg_pct': 0.520, 'three_pt_pct': 0.444, 'ft_pct': 0.789,
                'rebounds': 42, 'assists': 22, 'turnovers': 9, 'steals': 12,
                'opp_fg_pct': 0.299, 'opp_three_pt_pct': 0.222, 'opp_rebounds': 25,
                'opp_assists': 8, 'opp_turnovers': 18, 'opp_steals': 5
            },
            {
                'date': '2024-11-16', 'opponent': 'Drexel', 'location': 'Away',
                'result': 'W', 'score': 80, 'opp_score': 50,
                'fg_pct': 0.479, 'three_pt_pct': 0.385, 'ft_pct': 0.714,
                'rebounds': 40, 'assists': 19, 'turnovers': 10, 'steals': 9,
                'opp_fg_pct': 0.333, 'opp_three_pt_pct': 0.278, 'opp_rebounds': 28,
                'opp_assists': 10, 'opp_turnovers': 16, 'opp_steals': 7
            },
            {
                'date': '2024-11-19', 'opponent': 'Monmouth', 'location': 'Home',
                'result': 'W', 'score': 78, 'opp_score': 73,
                'fg_pct': 0.436, 'three_pt_pct': 0.357, 'ft_pct': 0.800,
                'rebounds': 35, 'assists': 15, 'turnovers': 13, 'steals': 7,
                'opp_fg_pct': 0.457, 'opp_three_pt_pct': 0.409, 'opp_rebounds': 33,
                'opp_assists': 16, 'opp_turnovers': 10, 'opp_steals': 8
            },
            # Tough tournament games
            {
                'date': '2024-11-25', 'opponent': 'Houston', 'location': 'Neutral',
                'result': 'L', 'score': 74, 'opp_score': 78,
                'fg_pct': 0.408, 'three_pt_pct': 0.294, 'ft_pct': 0.778,
                'rebounds': 34, 'assists': 13, 'turnovers': 15, 'steals': 6,
                'opp_fg_pct': 0.442, 'opp_three_pt_pct': 0.375, 'opp_rebounds': 38,
                'opp_assists': 15, 'opp_turnovers': 10, 'opp_steals': 9
            },
            {
                'date': '2024-11-26', 'opponent': 'Kansas', 'location': 'Neutral',
                'result': 'L', 'score': 60, 'opp_score': 71,
                'fg_pct': 0.362, 'three_pt_pct': 0.263, 'ft_pct': 0.714,
                'rebounds': 31, 'assists': 10, 'turnovers': 16, 'steals': 5,
                'opp_fg_pct': 0.456, 'opp_three_pt_pct': 0.333, 'opp_rebounds': 36,
                'opp_assists': 14, 'opp_turnovers': 8, 'opp_steals': 10
            },
            {
                'date': '2024-11-27', 'opponent': 'Iowa State', 'location': 'Neutral',
                'result': 'L', 'score': 64, 'opp_score': 95,
                'fg_pct': 0.373, 'three_pt_pct': 0.300, 'ft_pct': 0.750,
                'rebounds': 28, 'assists': 11, 'turnovers': 17, 'steals': 4,
                'opp_fg_pct': 0.528, 'opp_three_pt_pct': 0.476, 'opp_rebounds': 40,
                'opp_assists': 22, 'opp_turnovers': 8, 'opp_steals': 11
            },
            # Big loss to Tennessee
            {
                'date': '2024-12-03', 'opponent': 'Tennessee', 'location': 'Home',
                'result': 'L', 'score': 70, 'opp_score': 96,
                'fg_pct': 0.382, 'three_pt_pct': 0.333, 'ft_pct': 0.769,
                'rebounds': 30, 'assists': 12, 'turnovers': 18, 'steals': 3,
                'opp_fg_pct': 0.522, 'opp_three_pt_pct': 0.520, 'opp_rebounds': 38,
                'opp_assists': 19, 'opp_turnovers': 10, 'opp_steals': 10
            },
            # ACC conference play
            {
                'date': '2024-12-07', 'opponent': 'Youngstown State', 'location': 'Home',
                'result': 'W', 'score': 104, 'opp_score': 95,
                'fg_pct': 0.500, 'three_pt_pct': 0.429, 'ft_pct': 0.826,
                'rebounds': 41, 'assists': 24, 'turnovers': 14, 'steals': 10,
                'opp_fg_pct': 0.492, 'opp_three_pt_pct': 0.444, 'opp_rebounds': 35,
                'opp_assists': 20, 'opp_turnovers': 12, 'opp_steals': 8
            },
            {
                'date': '2024-12-31', 'opponent': 'Notre Dame', 'location': 'Home',
                'result': 'W', 'score': 69, 'opp_score': 64,
                'fg_pct': 0.429, 'three_pt_pct': 0.368, 'ft_pct': 0.778,
                'rebounds': 36, 'assists': 14, 'turnovers': 11, 'steals': 7,
                'opp_fg_pct': 0.400, 'opp_three_pt_pct': 0.286, 'opp_rebounds': 33,
                'opp_assists': 11, 'opp_turnovers': 13, 'opp_steals': 6
            },
            # More ACC losses
            {
                'date': '2025-01-04', 'opponent': 'Wake Forest', 'location': 'Away',
                'result': 'L', 'score': 71, 'opp_score': 81,
                'fg_pct': 0.404, 'three_pt_pct': 0.318, 'ft_pct': 0.786,
                'rebounds': 32, 'assists': 12, 'turnovers': 14, 'steals': 5,
                'opp_fg_pct': 0.481, 'opp_three_pt_pct': 0.391, 'opp_rebounds': 36,
                'opp_assists': 16, 'opp_turnovers': 9, 'opp_steals': 8
            },
            {
                'date': '2025-01-11', 'opponent': 'Pittsburgh', 'location': 'Home',
                'result': 'L', 'score': 73, 'opp_score': 77,
                'fg_pct': 0.418, 'three_pt_pct': 0.333, 'ft_pct': 0.800,
                'rebounds': 34, 'assists': 13, 'turnovers': 12, 'steals': 6,
                'opp_fg_pct': 0.444, 'opp_three_pt_pct': 0.385, 'opp_rebounds': 35,
                'opp_assists': 15, 'opp_turnovers': 11, 'opp_steals': 7
            },
            {
                'date': '2025-02-15', 'opponent': 'Duke', 'location': 'Away',
                'result': 'L', 'score': 54, 'opp_score': 83,
                'fg_pct': 0.324, 'three_pt_pct': 0.217, 'ft_pct': 0.706,
                'rebounds': 27, 'assists': 8, 'turnovers': 15, 'steals': 4,
                'opp_fg_pct': 0.500, 'opp_three_pt_pct': 0.421, 'opp_rebounds': 39,
                'opp_assists': 18, 'opp_turnovers': 7, 'opp_steals': 11
            },
        ]
        
        df = pd.DataFrame(games)
        return df
    
    def _clean_game_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate game data."""
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Create binary result column
        df['win'] = (df['result'] == 'W').astype(int)
        
        # Calculate point differential
        df['point_diff'] = df['score'] - df['opp_score']
        
        # Validate percentages
        pct_cols = [col for col in df.columns if 'pct' in col]
        for col in pct_cols:
            df[col] = df[col].clip(0, 1)
        
        return df
    
    def get_wins_losses(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into wins and losses."""
        if self.games_df is None:
            self.load_game_data()
        
        wins = self.games_df[self.games_df['win'] == 1]
        losses = self.games_df[self.games_df['win'] == 0]
        
        return wins, losses
    
    def get_game_by_opponent(self, opponent: str) -> pd.DataFrame:
        """Get specific game(s) against an opponent."""
        if self.games_df is None:
            self.load_game_data()
        
        return self.games_df[self.games_df['opponent'].str.contains(opponent, case=False)]
    
    def get_season_stats(self) -> Dict:
        """Calculate season-wide statistics."""
        if self.games_df is None:
            self.load_game_data()
        
        return {
            'record': f"{self.games_df['win'].sum()}-{len(self.games_df) - self.games_df['win'].sum()}",
            'avg_points': self.games_df['score'].mean(),
            'avg_opp_points': self.games_df['opp_score'].mean(),
            'avg_fg_pct': self.games_df['fg_pct'].mean(),
            'avg_three_pt_pct': self.games_df['three_pt_pct'].mean(),
            'avg_ft_pct': self.games_df['ft_pct'].mean(),
            'avg_rebounds': self.games_df['rebounds'].mean(),
            'avg_assists': self.games_df['assists'].mean(),
            'avg_turnovers': self.games_df['turnovers'].mean(),
            'avg_steals': self.games_df['steals'].mean()
        }
    
    def export_to_csv(self, filename: str = 'syracuse_games_2024_25.csv'):
        """Export current dataframe to CSV."""
        if self.games_df is not None:
            output_path = self.data_dir / filename
            self.games_df.to_csv(output_path, index=False)
            print(f"Data exported to {output_path}")
        else:
            print("No data to export. Load data first.")


# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = GameDataLoader()
    
    # Load data (creates sample if not exists)
    games = loader.load_game_data()
    
    print(f"Loaded {len(games)} games")
    print(f"\nSeason stats:")
    stats = loader.get_season_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Get wins and losses
    wins, losses = loader.get_wins_losses()
    print(f"\nWins: {len(wins)}, Losses: {len(losses)}")
    
    # Find specific game
    tennessee_game = loader.get_game_by_opponent('Tennessee')
    if not tennessee_game.empty:
        print(f"\nTennessee game:")
        print(f"  Score: {tennessee_game.iloc[0]['score']} - {tennessee_game.iloc[0]['opp_score']}")
        print(f"  FG%: {tennessee_game.iloc[0]['fg_pct']:.1%}")
    
    # Export to CSV
    loader.export_to_csv()
