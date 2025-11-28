"""
setup.py - Quick setup and test script

Run this after cloning the repository to:
1. Create directory structure
2. Generate sample data
3. Train initial models
4. Run test analysis

Usage: python setup.py
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create required directories."""
    print("Creating directory structure...")
    
    directories = [
        'data',
        'models',
        'outputs',
        'outputs/game_reports',
        'outputs/visualizations',
        'src',
        'tests',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}/")
    
    # Create __init__.py for src
    (Path('src') / '__init__.py').touch()
    print("  ✓ Created: src/__init__.py")

def generate_sample_data():
    """Generate sample game data."""
    print("\nGenerating sample data...")
    
    try:
        sys.path.insert(0, 'src')
        from data_loader import GameDataLoader
        
        loader = GameDataLoader()
        games = loader.load_game_data()
        loader.export_to_csv()
        
        print(f"  ✓ Generated {len(games)} games")
        print(f"  ✓ Saved to: data/syracuse_games_2024_25.csv")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def train_models():
    """Train ML models."""
    print("\nTraining machine learning models...")
    
    try:
        from src.data_loader import GameDataLoader
        from src.models import WinPredictor
        
        # Load data
        loader = GameDataLoader()
        games = loader.load_game_data()
        
        # Train predictor
        predictor = WinPredictor(model_type='random_forest')
        results = predictor.train(games)
        
        print(f"  ✓ Model trained")
        print(f"    - Train accuracy: {results['train_accuracy']:.1%}")
        print(f"    - Test accuracy: {results['test_accuracy']:.1%}")
        print(f"    - CV score: {results['cv_mean']:.1%} (+/- {results['cv_std']:.1%})")
        
        # Save model
        predictor.save()
        print(f"  ✓ Model saved to: models/win_predictor.pkl")
        
        # Show top features
        print("\n  Top 3 Most Important Features:")
        for feat, importance in predictor.get_top_features(3):
            print(f"    {feat}: {importance:.3f}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def run_sample_analysis():
    """Run sample game analysis."""
    print("\nRunning sample analysis...")
    
    try:
        from src.data_loader import GameDataLoader
        from src.analyzer import StrategicAnalyst
        
        # Load data
        loader = GameDataLoader()
        games = loader.load_game_data()
        
        # Find a loss to analyze
        losses = games[games['win'] == 0]
        if len(losses) == 0:
            print("  ! No losses found in dataset")
            return False
        
        sample_game = losses.iloc[0]
        
        # Analyze
        analyst = StrategicAnalyst()
        report = analyst.generate_report(sample_game, games)
        
        # Save report
        report_path = Path('outputs') / 'game_reports' / f"sample_analysis_{sample_game['opponent']}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"  ✓ Analysis complete")
        print(f"  ✓ Report saved to: {report_path}")
        print("\n" + "="*60)
        print("SAMPLE OUTPUT (first 500 chars):")
        print("="*60)
        print(report[:500] + "...")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main setup workflow."""
    print("="*60)
    print("AI STRATEGIC ANALYST - SETUP")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Setup incomplete: Install missing dependencies first")
        return
    
    print("\n✓ All dependencies installed\n")
    
    # Create directories
    create_directory_structure()
    
    # Generate data
    if not generate_sample_data():
        print("\n❌ Setup incomplete: Failed to generate data")
        return
    
    # Train models
    if not train_models():
        print("\n❌ Setup incomplete: Failed to train models")
        return
    
    # Run sample analysis
    if not run_sample_analysis():
        print("\n⚠️  Setup complete but sample analysis failed")
        print("You can still use the system manually")
    
    # Success message
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review sample analysis in outputs/game_reports/")
    print("  2. Run: streamlit run app.py")
    print("  3. Or: python src/analyzer.py")
    print("\nDocumentation:")
    print("  - README.md: Complete project documentation")
    print("  - PROJECT_SUMMARY.md: Quick overview")
    print("\n Happy analyzing! 🏀")

if __name__ == "__main__":
    main()
