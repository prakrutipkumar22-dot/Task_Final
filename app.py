"""
app.py - Streamlit Web Interface for AI Strategic Analyst

Launch with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import custom modules
import sys
sys.path.append('src')
from data_loader import GameDataLoader
from analyzer import StrategicAnalyst
from models import WinPredictor

# Page config
st.set_page_config(
    page_title="AI Basketball Analyst",
    page_icon="🏀",
    layout="wide"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analyst' not in st.session_state:
    st.session_state.analyst = None

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def load_data():
    """Load game data and initialize analyst."""
    loader = GameDataLoader()
    games = loader.load_game_data()
    
    analyst = StrategicAnalyst()
    analyst.set_season_context(games)
    
    return games, analyst

def main():
    # Header
    st.title("🏀 AI Strategic Basketball Analyst")
    st.markdown("**Can AI Replace a Human Analyst?** Explore game analysis powered by machine learning")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Analysis",
        ["Overview", "Game Analysis", "Model Performance", "Season Insights"]
    )
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            games, analyst = load_data()
            st.session_state.games = games
            st.session_state.analyst = analyst
            st.session_state.data_loaded = True
    
    games = st.session_state.games
    analyst = st.session_state.analyst
    
    # Page routing
    if page == "Overview":
        show_overview(games)
    elif page == "Game Analysis":
        show_game_analysis(games, analyst)
    elif page == "Model Performance":
        show_model_performance(games)
    elif page == "Season Insights":
        show_season_insights(games)

def show_overview(games: pd.DataFrame):
    """Display season overview."""
    st.header("Syracuse 2024-25 Season Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    wins = games['win'].sum()
    losses = len(games) - wins
    
    with col1:
        st.metric("Record", f"{wins}-{losses}")
    with col2:
        st.metric("Avg Points", f"{games['score'].mean():.1f}")
    with col3:
        st.metric("Avg FG%", f"{games['fg_pct'].mean():.1%}")
    with col4:
        st.metric("Avg Point Diff", f"{games['point_diff'].mean():.1f}")
    
    # Wins vs Losses comparison
    st.subheader("Performance: Wins vs Losses")
    
    col1, col2 = st.columns(2)
    
    wins_df = games[games['win'] == 1]
    losses_df = games[games['win'] == 0]
    
    with col1:
        st.markdown("**Wins**")
        st.metric("Games", len(wins_df))
        st.metric("Avg FG%", f"{wins_df['fg_pct'].mean():.1%}")
        st.metric("Avg Turnovers", f"{wins_df['turnovers'].mean():.1f}")
        st.metric("Avg Point Margin", f"+{wins_df['point_diff'].mean():.1f}")
    
    with col2:
        st.markdown("**Losses**")
        st.metric("Games", len(losses_df))
        st.metric("Avg FG%", f"{losses_df['fg_pct'].mean():.1%}")
        st.metric("Avg Turnovers", f"{losses_df['turnovers'].mean():.1f}")
        st.metric("Avg Point Margin", f"{losses_df['point_diff'].mean():.1f}")
    
    # Game results chart
    st.subheader("Game Results Timeline")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ['green' if w == 1 else 'red' for w in games['win']]
    ax.bar(range(len(games)), games['point_diff'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Game Number')
    ax.set_ylabel('Point Differential')
    ax.set_title('Point Differential by Game (Green=Win, Red=Loss)')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Recent games
    st.subheader("Recent Games")
    recent = games.tail(5)[['date', 'opponent', 'result', 'score', 'opp_score', 'point_diff']]
    st.dataframe(recent, use_container_width=True)

def show_game_analysis(games: pd.DataFrame, analyst: StrategicAnalyst):
    """Show detailed game analysis."""
    st.header("Game-by-Game Analysis")
    
    # Game selector
    game_list = [f"{row['date']} vs {row['opponent']} ({row['result']})" 
                 for _, row in games.iterrows()]
    
    selected_game = st.selectbox("Select a game to analyze:", game_list)
    game_idx = game_list.index(selected_game)
    game = games.iloc[game_idx]
    
    # Display game info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Opponent", game['opponent'])
    with col2:
        st.metric("Result", f"{game['result']} {int(game['score'])}-{int(game['opp_score'])}")
    with col3:
        st.metric("Point Diff", f"{int(game['point_diff'])}")
    
    # Generate analysis
    with st.spinner("Analyzing game..."):
        report_text = analyst.generate_report(game, games)
    
    # Display full report
    st.subheader("AI Strategic Analysis")
    st.text(report_text)
    
    # Key stats comparison
    st.subheader("Key Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Syracuse**")
        metrics = {
            'FG%': f"{game['fg_pct']:.1%}",
            '3PT%': f"{game['three_pt_pct']:.1%}",
            'FT%': f"{game['ft_pct']:.1%}",
            'Rebounds': int(game['rebounds']),
            'Assists': int(game['assists']),
            'Turnovers': int(game['turnovers']),
            'Steals': int(game['steals'])
        }
        for key, value in metrics.items():
            st.metric(key, value)
    
    with col2:
        st.markdown("**Opponent**")
        opp_metrics = {
            'FG%': f"{game['opp_fg_pct']:.1%}",
            '3PT%': f"{game['opp_three_pt_pct']:.1%}",
            'Rebounds': int(game['opp_rebounds']),
            'Assists': int(game['opp_assists']),
            'Turnovers': int(game['opp_turnovers']),
            'Steals': int(game['opp_steals'])
        }
        for key, value in opp_metrics.items():
            st.metric(key, value)

def show_model_performance(games: pd.DataFrame):
    """Show ML model performance."""
    st.header("Machine Learning Model Performance")
    
    st.markdown("""
    This page shows the performance of our AI models in predicting game outcomes
    and identifying key factors in wins/losses.
    """)
    
    # Train model (in real app, load pre-trained)
    with st.spinner("Training model..."):
        predictor = WinPredictor(model_type='random_forest')
        results = predictor.train(games)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test Accuracy", f"{results['test_accuracy']:.1%}")
    with col2:
        st.metric("CV Accuracy", f"{results['cv_mean']:.1%}")
    with col3:
        st.metric("Train Accuracy", f"{results['train_accuracy']:.1%}")
    
    # Feature importance
    st.subheader("Most Important Factors for Winning")
    
    if results['feature_importance']:
        feat_df = pd.DataFrame(
            list(results['feature_importance'].items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(feat_df['Feature'], feat_df['Importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance in Win Prediction')
        st.pyplot(fig)
    
    # Confusion matrix
    st.subheader("Model Predictions vs Actual Results")
    
    cm = results['confusion_matrix']
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Loss', 'Win'],
                yticklabels=['Loss', 'Win'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    
    # Classification report
    st.subheader("Detailed Classification Metrics")
    
    report = results['classification_report']
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3))

def show_season_insights(games: pd.DataFrame):
    """Show season-wide insights and trends."""
    st.header("Season Insights & Trends")
    
    # Performance over time
    st.subheader("Performance Trends")
    
    games['game_num'] = range(1, len(games) + 1)
    games['fg_pct_ma'] = games['fg_pct'].rolling(window=3).mean()
    games['point_diff_ma'] = games['point_diff'].rolling(window=3).mean()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # FG% trend
    ax1.plot(games['game_num'], games['fg_pct'], 'o-', alpha=0.5, label='FG%')
    ax1.plot(games['game_num'], games['fg_pct_ma'], '-', linewidth=2, label='3-game avg')
    ax1.axhline(y=games['fg_pct'].mean(), color='red', linestyle='--', label='Season avg')
    ax1.set_ylabel('Field Goal %')
    ax1.set_title('Shooting Performance Over Season')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Point differential trend
    ax2.plot(games['game_num'], games['point_diff'], 'o-', alpha=0.5, label='Point Diff')
    ax2.plot(games['game_num'], games['point_diff_ma'], '-', linewidth=2, label='3-game avg')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=games['point_diff'].mean(), color='red', linestyle='--', label='Season avg')
    ax2.set_xlabel('Game Number')
    ax2.set_ylabel('Point Differential')
    ax2.set_title('Scoring Margin Over Season')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Home vs Away
    st.subheader("Home vs Away Performance")
    
    home_games = games[games['location'] == 'Home']
    away_games = games[games['location'] == 'Away']
    neutral_games = games[games['location'] == 'Neutral']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Home Games**")
        st.metric("Record", f"{home_games['win'].sum()}-{len(home_games) - home_games['win'].sum()}")
        st.metric("Avg Point Diff", f"{home_games['point_diff'].mean():.1f}")
    
    with col2:
        st.markdown("**Away Games**")
        st.metric("Record", f"{away_games['win'].sum()}-{len(away_games) - away_games['win'].sum()}")
        st.metric("Avg Point Diff", f"{away_games['point_diff'].mean():.1f}")
    
    with col3:
        st.markdown("**Neutral Site**")
        st.metric("Record", f"{neutral_games['win'].sum()}-{len(neutral_games) - neutral_games['win'].sum()}")
        st.metric("Avg Point Diff", f"{neutral_games['point_diff'].mean():.1f}")
    
    # Key correlations
    st.subheader("What Stats Correlate with Winning?")
    
    correlations = games[['win', 'fg_pct', 'three_pt_pct', 'turnovers', 
                          'rebounds', 'opp_fg_pct']].corr()['win'].sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations[1:].plot(kind='barh', ax=ax)
    ax.set_xlabel('Correlation with Winning')
    ax.set_title('Statistical Factors Most Correlated with Wins')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    st.pyplot(fig)

# Run app
if __name__ == "__main__":
    main()
