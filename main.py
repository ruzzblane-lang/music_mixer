#!/usr/bin/env python3
"""
AI-Assisted Music Mixer - Main Entry Point

This module provides the command-line interface for the AI music mixer.
"""

import click
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from features.scanner import MusicScanner
from recommend.engine import RecommendationEngine
from mix.engine import MixingEngine
from ui.cli import CLIInterface


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """AI-Assisted Music Mixer - Intelligent track recommendations and mixing."""
    pass


@cli.command()
@click.argument('music_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--force', '-f', is_flag=True, help='Force re-scan of already analyzed tracks')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def scanmusic(music_path, force, verbose):
    """Scan and analyze a music library directory."""
    click.echo(f"🎵 Scanning music library: {music_path}")
    
    scanner = MusicScanner(verbose=verbose)
    try:
        results = scanner.scan_directory(music_path, force_rescan=force)
        click.echo(f"✅ Analysis complete! Processed {results['total_tracks']} tracks")
        click.echo(f"📊 Found {results['new_tracks']} new tracks, {results['updated_tracks']} updated")
    except Exception as e:
        click.echo(f"❌ Error during scanning: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('track_name')
@click.option('--count', '-c', default=5, help='Number of recommendations to show')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed compatibility scores')
def recommend(track_name, count, verbose):
    """Get track recommendations for a given track."""
    click.echo(f"🎯 Finding recommendations for: {track_name}")
    
    engine = RecommendationEngine()
    try:
        recommendations = engine.get_recommendations(track_name, count=count)
        
        if not recommendations:
            click.echo("❌ No recommendations found. Make sure the track exists in your library.")
            return
        
        click.echo(f"\n🎵 Top {len(recommendations)} recommendations:")
        for i, (track, score) in enumerate(recommendations, 1):
            if verbose:
                click.echo(f"{i}. {track['title']} - {track['artist']} (Score: {score:.2f})")
            else:
                click.echo(f"{i}. {track['title']} - {track['artist']}")
                
    except Exception as e:
        click.echo(f"❌ Error getting recommendations: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--interactive', '-i', is_flag=True, help='Start interactive mixing session')
def mix(interactive):
    """Start a mixing session."""
    if interactive:
        click.echo("🎧 Starting interactive mixing session...")
        cli_interface = CLIInterface()
        cli_interface.start_session()
    else:
        click.echo("🎧 Mixing mode - use --interactive for full session")


@cli.command()
def status():
    """Show system status and database statistics."""
    click.echo("📊 AI Music Mixer Status")
    click.echo("=" * 30)
    
    # TODO: Implement status checking
    click.echo("Database: Connected")
    click.echo("Tracks analyzed: 0")
    click.echo("Recommendations cached: 0")


if __name__ == '__main__':
    cli()
