"""
analyzer.py - AI Strategic Analyst Engine

Core module that answers:
- Why did we lose?
- What could we have done differently?
- How can we improve?
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import pickle

class StrategicAnalyst:
    """
    AI-powered strategic analyst for basketball games.
    
    Provides analysis comparable to human basketball analysts.
    """
    
    def __init__(self):
        self.models = {}
        self.season_averages = None
        
    def load_models(self, model_dir: str = 'models'):
        """Load trained ML models."""
        model_path = Path(model_dir)
        
        try:
            with open(model_path / 'win_predictor.pkl', 'rb') as f:
                self.models['predictor'] = pickle.load(f)
        except FileNotFoundError:
            print("Win predictor model not found. Train first.")
            
    def set_season_context(self, games_df: pd.DataFrame):
        """Set season-wide averages for comparison."""
        self.season_averages = {
            'fg_pct': games_df['fg_pct'].mean(),
            'three_pt_pct': games_df['three_pt_pct'].mean(),
            'ft_pct': games_df['ft_pct'].mean(),
            'rebounds': games_df['rebounds'].mean(),
            'assists': games_df['assists'].mean(),
            'turnovers': games_df['turnovers'].mean(),
            'steals': games_df['steals'].mean(),
            'opp_fg_pct': games_df['opp_fg_pct'].mean(),
            'point_diff': games_df['point_diff'].mean()
        }
        
    def analyze_game(self, game: pd.Series, games_df: pd.DataFrame = None) -> Dict:
        """
        Complete strategic analysis of a single game.
        
        Args:
            game: Single game data (pd.Series)
            games_df: Full season data for context
            
        Returns:
            Dictionary with complete analysis
        """
        if games_df is not None:
            self.set_season_context(games_df)
        
        analysis = {
            'game_info': self._extract_game_info(game),
            'why_we_lost': self._analyze_why_lost(game) if game['result'] == 'L' else "N/A (Won game)",
            'what_to_change': self._counterfactual_analysis(game),
            'how_to_improve': self._generate_recommendations(game),
            'key_metrics': self._identify_key_metrics(game)
        }
        
        return analysis
    
    def _extract_game_info(self, game: pd.Series) -> Dict:
        """Extract basic game information."""
        return {
            'date': game['date'],
            'opponent': game['opponent'],
            'location': game['location'],
            'result': game['result'],
            'score': f"{int(game['score'])}-{int(game['opp_score'])}",
            'point_diff': int(game['point_diff'])
        }
    
    def _analyze_why_lost(self, game: pd.Series) -> Dict:
        """
        Answer: "Why did we lose this game?"
        
        Identifies top 3 factors contributing to loss.
        """
        if self.season_averages is None:
            return {"error": "Season context not set"}
        
        factors = []
        
        # Calculate deviations from season averages
        fg_diff = game['fg_pct'] - self.season_averages['fg_pct']
        three_diff = game['three_pt_pct'] - self.season_averages['three_pt_pct']
        ft_diff = game['ft_pct'] - self.season_averages['ft_pct']
        to_diff = game['turnovers'] - self.season_averages['turnovers']
        reb_diff = game['rebounds'] - self.season_averages['rebounds']
        def_diff = game['opp_fg_pct'] - self.season_averages['opp_fg_pct']
        
        # Shooting performance
        if fg_diff < -0.05:
            severity = "CRITICAL" if fg_diff < -0.10 else "HIGH IMPACT"
            factors.append({
                'factor': 'Field Goal %',
                'value': f"{game['fg_pct']:.1%}",
                'season_avg': f"{self.season_averages['fg_pct']:.1%}",
                'diff': f"{fg_diff:.1%}",
                'severity': severity,
                'explanation': f"Shot {abs(fg_diff)*100:.1f}% below season average"
            })
        
        # Three-point shooting
        if three_diff < -0.05:
            severity = "HIGH IMPACT" if three_diff < -0.08 else "MEDIUM"
            factors.append({
                'factor': '3-Point %',
                'value': f"{game['three_pt_pct']:.1%}",
                'season_avg': f"{self.season_averages['three_pt_pct']:.1%}",
                'diff': f"{three_diff:.1%}",
                'severity': severity,
                'explanation': f"3PT shooting {abs(three_diff)*100:.1f}% below average"
            })
        
        # Turnovers
        if to_diff > 3:
            severity = "CRITICAL" if to_diff > 6 else "HIGH IMPACT"
            factors.append({
                'factor': 'Turnovers',
                'value': int(game['turnovers']),
                'season_avg': f"{self.season_averages['turnovers']:.1f}",
                'diff': f"+{int(to_diff)}",
                'severity': severity,
                'explanation': f"{int(to_diff)} more turnovers than season average"
            })
        
        # Defensive performance
        if def_diff > 0.05:
            severity = "CRITICAL" if def_diff > 0.10 else "HIGH IMPACT"
            factors.append({
                'factor': 'Opponent FG%',
                'value': f"{game['opp_fg_pct']:.1%}",
                'season_avg': f"{self.season_averages['opp_fg_pct']:.1%}",
                'diff': f"+{def_diff:.1%}",
                'severity': severity,
                'explanation': f"Opponent shot {def_diff*100:.1f}% above what we typically allow"
            })
        
        # Rebounding
        opp_reb_diff = game['opp_rebounds'] - game['rebounds']
        if opp_reb_diff > 5:
            severity = "HIGH IMPACT" if opp_reb_diff > 8 else "MEDIUM"
            factors.append({
                'factor': 'Rebounding Margin',
                'value': int(opp_reb_diff),
                'season_avg': "Even",
                'diff': f"-{int(opp_reb_diff)}",
                'severity': severity,
                'explanation': f"Out-rebounded by {int(opp_reb_diff)}"
            })
        
        # Sort by severity
        severity_order = {'CRITICAL': 0, 'HIGH IMPACT': 1, 'MEDIUM': 2, 'LOW': 3}
        factors.sort(key=lambda x: (severity_order.get(x['severity'], 4), abs(float(x['diff'].replace('%', '').replace('+', '').replace('-', '')))))
        
        # Return top 3 factors
        return {
            'primary_factors': factors[:3],
            'summary': self._generate_loss_summary(factors[:3], game)
        }
    
    def _generate_loss_summary(self, factors: List[Dict], game: pd.Series) -> str:
        """Generate natural language summary of why we lost."""
        if not factors:
            return "Loss occurred despite playing close to season averages. Opponent simply outperformed."
        
        summary = f"Lost to {game['opponent']} by {abs(int(game['point_diff']))} points. "
        
        if len(factors) >= 1:
            summary += f"Primary issue: {factors[0]['explanation']}. "
        
        if len(factors) >= 2:
            summary += f"Also struggled with {factors[1]['factor'].lower()}: {factors[1]['explanation']}. "
        
        if len(factors) >= 3:
            summary += f"Additionally, {factors[2]['explanation']}."
        
        return summary
    
    def _counterfactual_analysis(self, game: pd.Series) -> Dict:
        """
        Answer: "What could we have done differently?"
        
        Simulates alternative scenarios.
        """
        scenarios = []
        
        # Scenario 1: Improve shooting to season average
        if game['fg_pct'] < self.season_averages['fg_pct']:
            improved_fg = self.season_averages['fg_pct']
            # Rough estimate: each 1% FG improvement = ~2 points
            point_gain = (improved_fg - game['fg_pct']) * 100 * 2
            new_diff = game['point_diff'] + point_gain
            
            scenarios.append({
                'scenario': f"If FG% = {improved_fg:.1%} (season avg)",
                'improvement': f"+{point_gain:.1f} points",
                'new_result': 'Win' if new_diff > 0 else f'Loss by {abs(int(new_diff))}',
                'probability': self._estimate_win_prob(new_diff),
                'feasibility': 'High - season average performance'
            })
        
        # Scenario 2: Reduce turnovers to season average
        if game['turnovers'] > self.season_averages['turnovers']:
            to_reduction = game['turnovers'] - self.season_averages['turnovers']
            # Each turnover costs ~1.5 points
            point_gain = to_reduction * 1.5
            new_diff = game['point_diff'] + point_gain
            
            scenarios.append({
                'scenario': f"If turnovers = {self.season_averages['turnovers']:.1f} (season avg)",
                'improvement': f"+{point_gain:.1f} points",
                'new_result': 'Win' if new_diff > 0 else f'Loss by {abs(int(new_diff))}',
                'probability': self._estimate_win_prob(new_diff),
                'feasibility': 'Medium - requires better ball security'
            })
        
        # Scenario 3: Improve defense to season average
        if game['opp_fg_pct'] > self.season_averages['opp_fg_pct']:
            def_improvement = game['opp_fg_pct'] - self.season_averages['opp_fg_pct']
            # Each 1% opponent FG reduction = ~2 points saved
            point_gain = def_improvement * 100 * 2
            new_diff = game['point_diff'] + point_gain
            
            scenarios.append({
                'scenario': f"If opponent FG% = {self.season_averages['opp_fg_pct']:.1%} (held to avg)",
                'improvement': f"+{point_gain:.1f} points",
                'new_result': 'Win' if new_diff > 0 else f'Loss by {abs(int(new_diff))}',
                'probability': self._estimate_win_prob(new_diff),
                'feasibility': 'Medium - requires defensive execution'
            })
        
        # Scenario 4: Combined improvements
        total_gain = sum(float(s['improvement'].replace('+', '').replace(' points', '')) 
                        for s in scenarios[:3])
        if total_gain > 0:
            new_diff = game['point_diff'] + total_gain
            scenarios.append({
                'scenario': "If all key metrics = season average",
                'improvement': f"+{total_gain:.1f} points",
                'new_result': 'Win' if new_diff > 0 else f'Loss by {abs(int(new_diff))}',
                'probability': self._estimate_win_prob(new_diff),
                'feasibility': 'Low - multiple factors must align'
            })
        
        return {
            'scenarios': scenarios,
            'recommendation': self._select_best_scenario(scenarios)
        }
    
    def _estimate_win_prob(self, point_diff: float) -> str:
        """Estimate win probability based on point differential."""
        if point_diff > 10:
            return "95%+"
        elif point_diff > 5:
            return "80-95%"
        elif point_diff > 0:
            return "60-80%"
        elif point_diff > -5:
            return "40-60%"
        elif point_diff > -10:
            return "20-40%"
        else:
            return "<20%"
    
    def _select_best_scenario(self, scenarios: List[Dict]) -> str:
        """Select the most actionable scenario."""
        if not scenarios:
            return "Team performed near season averages. Focus on execution."
        
        # Prioritize high feasibility scenarios that flip result
        winnable = [s for s in scenarios if 'Win' in s['new_result'] and 'High' in s['feasibility']]
        
        if winnable:
            return f"Most actionable: {winnable[0]['scenario']}. This high-feasibility improvement would have changed the outcome."
        
        # Otherwise, return highest impact
        scenarios_sorted = sorted(scenarios, 
                                 key=lambda x: float(x['improvement'].replace('+', '').replace(' points', '')), 
                                 reverse=True)
        return f"Highest impact: {scenarios_sorted[0]['scenario']}. Focus practice on this area."
    
    def _generate_recommendations(self, game: pd.Series) -> Dict:
        """
        Answer: "How can we improve?"
        
        Generates actionable recommendations.
        """
        recommendations = []
        
        # Shooting recommendations
        if game['fg_pct'] < 0.40:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Shooting Efficiency',
                'issue': f"FG% of {game['fg_pct']:.1%} is below acceptable threshold",
                'recommendation': "Focus on shot selection and getting higher-percentage looks",
                'tactics': [
                    "Increase ball movement to create open shots",
                    "Attack the rim more for higher-percentage attempts",
                    "Review and correct poor shot selection in film session"
                ]
            })
        
        if game['three_pt_pct'] < 0.30:
            recommendations.append({
                'priority': 'MEDIUM',
                'area': '3-Point Shooting',
                'issue': f"3PT% of {game['three_pt_pct']:.1%} severely limited scoring",
                'recommendation': "Either improve 3PT shooting or reduce volume",
                'tactics': [
                    "Extra shooting drills for perimeter players",
                    "Take fewer contested threes",
                    "Generate better looks through screens and movement"
                ]
            })
        
        # Turnover recommendations
        if game['turnovers'] > 15:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Ball Security',
                'issue': f"{int(game['turnovers'])} turnovers directly led to opponent points",
                'recommendation': "Emphasize decision-making and ball protection",
                'tactics': [
                    "Practice against pressure defense",
                    "Simplify offense - make the easy pass",
                    "Reduce live-ball turnovers that lead to transition points"
                ]
            })
        
        # Defensive recommendations
        if game['opp_fg_pct'] > 0.48:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Perimeter Defense',
                'issue': f"Opponent shot {game['opp_fg_pct']:.1%}, too high to win consistently",
                'recommendation': "Tighten defensive rotations and closeouts",
                'tactics': [
                    "Drill closeout technique - contest without fouling",
                    "Improve help defense positioning",
                    "Communicate switches and screens better"
                ]
            })
        
        # Rebounding recommendations
        if game['rebounds'] < game['opp_rebounds'] - 5:
            recommendations.append({
                'priority': 'MEDIUM',
                'area': 'Rebounding',
                'issue': f"Out-rebounded by {int(game['opp_rebounds'] - game['rebounds'])}",
                'recommendation': "Emphasize boxing out and crashing boards",
                'tactics': [
                    "Boxing out drills in practice",
                    "Send more players to offensive glass",
                    "Don't concede second-chance points"
                ]
            })
        
        # Sort by priority
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return {
            'recommendations': recommendations[:4],  # Top 4
            'practice_focus': self._generate_practice_plan(recommendations[:2])
        }
    
    def _generate_practice_plan(self, top_recs: List[Dict]) -> List[str]:
        """Generate specific practice plan items."""
        plan = []
        
        for rec in top_recs:
            plan.append(f"[{rec['priority']}] {rec['area']}: {rec['recommendation']}")
            plan.extend([f"  • {tactic}" for tactic in rec['tactics'][:2]])
        
        return plan
    
    def _identify_key_metrics(self, game: pd.Series) -> Dict:
        """Identify which metrics mattered most in this game."""
        return {
            'shooting_impact': self._calculate_impact(game['fg_pct'], self.season_averages['fg_pct']),
            'defense_impact': self._calculate_impact(game['opp_fg_pct'], self.season_averages['opp_fg_pct'], inverse=True),
            'turnover_impact': self._calculate_impact(game['turnovers'], self.season_averages['turnovers'], inverse=True),
            'rebounding_impact': self._calculate_impact(game['rebounds'], self.season_averages['rebounds'])
        }
    
    def _calculate_impact(self, game_value: float, season_avg: float, inverse: bool = False) -> str:
        """Calculate impact level of a metric."""
        diff = game_value - season_avg
        if inverse:
            diff = -diff
        
        if abs(diff) > 0.1 * season_avg:
            return "High Impact" if diff > 0 else "Significant Negative Impact"
        elif abs(diff) > 0.05 * season_avg:
            return "Moderate Impact" if diff > 0 else "Moderate Negative Impact"
        else:
            return "Minimal Impact"
    
    def generate_report(self, game: pd.Series, games_df: pd.DataFrame = None) -> str:
        """Generate formatted text report."""
        analysis = self.analyze_game(game, games_df)
        
        report = []
        report.append("="*60)
        report.append("AI STRATEGIC ANALYSIS")
        report.append("="*60)
        
        # Game info
        info = analysis['game_info']
        report.append(f"\nGame: vs {info['opponent']} ({info['location']})")
        report.append(f"Date: {info['date']}")
        report.append(f"Result: {info['result']} {info['score']}")
        
        # Why we lost
        if analysis['why_we_lost'] != "N/A (Won game)":
            report.append("\n" + "="*60)
            report.append("WHY WE LOST")
            report.append("="*60)
            
            for i, factor in enumerate(analysis['why_we_lost']['primary_factors'], 1):
                report.append(f"\n{i}. {factor['factor']}: {factor['value']} "
                            f"(Season Avg: {factor['season_avg']}) [{factor['severity']}]")
                report.append(f"   → {factor['explanation']}")
            
            report.append(f"\n{analysis['why_we_lost']['summary']}")
        
        # What to change
        report.append("\n" + "="*60)
        report.append("WHAT COULD WE HAVE DONE DIFFERENTLY")
        report.append("="*60)
        
        for scenario in analysis['what_to_change']['scenarios'][:3]:
            report.append(f"\n• {scenario['scenario']}")
            report.append(f"  Improvement: {scenario['improvement']}")
            report.append(f"  New result: {scenario['new_result']}")
            report.append(f"  Win probability: {scenario['probability']}")
        
        report.append(f"\n{analysis['what_to_change']['recommendation']}")
        
        # How to improve
        report.append("\n" + "="*60)
        report.append("HOW TO IMPROVE")
        report.append("="*60)
        
        for rec in analysis['how_to_improve']['recommendations']:
            report.append(f"\n[{rec['priority']}] {rec['area']}")
            report.append(f"Issue: {rec['issue']}")
            report.append(f"Recommendation: {rec['recommendation']}")
            report.append("Tactics:")
            for tactic in rec['tactics']:
                report.append(f"  • {tactic}")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    from data_loader import GameDataLoader
    
    # Load data
    loader = GameDataLoader()
    games = loader.load_game_data()
    
    # Initialize analyst
    analyst = StrategicAnalyst()
    
    # Find Tennessee game (big loss)
    tennessee_game = games[games['opponent'] == 'Tennessee'].iloc[0]
    
    # Generate full report
    report = analyst.generate_report(tennessee_game, games)
    print(report)
