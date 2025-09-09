"""
Command Line Interface

Provides an interactive CLI for the AI music mixer.
"""

import click
import sys
from typing import List, Dict, Optional
from colorama import init, Fore, Style
import time

from recommend.engine import RecommendationEngine
from mix.engine import MixingEngine
from features.database import MusicDatabase

# Initialize colorama for cross-platform colored output
init()


class CLIInterface:
    """Interactive command-line interface for the AI music mixer."""
    
    def __init__(self, db_path: str = "data/music_library.db"):
        """
        Initialize the CLI interface.
        
        Args:
            db_path: Path to the music database
        """
        self.db = MusicDatabase(db_path)
        self.recommendation_engine = RecommendationEngine(db_path)
        self.mixing_engine = MixingEngine()
        self.current_track = None
        self.mix_history = []
    
    def start_session(self):
        """Start an interactive mixing session."""
        self._print_welcome()
        
        # Check if library is scanned
        stats = self.db.get_library_stats()
        if stats['total_tracks'] == 0:
            self._print_error("No tracks found in library. Please run 'scanmusic' first.")
            return
        
        self._print_library_stats(stats)
        
        # Main session loop
        while True:
            try:
                self._print_main_menu()
                choice = input(f"{Fore.CYAN}Enter your choice: {Style.RESET_ALL}").strip().lower()
                
                if choice == '1':
                    self._search_and_select_track()
                elif choice == '2':
                    self._get_recommendations()
                elif choice == '3':
                    self._create_mix()
                elif choice == '4':
                    self._show_mix_history()
                elif choice == '5':
                    self._show_library_stats()
                elif choice == '6':
                    self._settings_menu()
                elif choice == 'q' or choice == 'quit':
                    self._print_goodbye()
                    break
                else:
                    self._print_error("Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                self._print_goodbye()
                break
            except Exception as e:
                self._print_error(f"An error occurred: {e}")
    
    def _print_welcome(self):
        """Print welcome message."""
        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🎵 AI-Assisted Music Mixer{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Welcome to your intelligent music mixing assistant!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Get AI-powered track recommendations and create seamless mixes.{Style.RESET_ALL}\n")
    
    def _print_main_menu(self):
        """Print the main menu."""
        print(f"\n{Fore.BLUE}Main Menu:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}1.{Style.RESET_ALL} Search and select track")
        print(f"{Fore.WHITE}2.{Style.RESET_ALL} Get recommendations for current track")
        print(f"{Fore.WHITE}3.{Style.RESET_ALL} Create mix")
        print(f"{Fore.WHITE}4.{Style.RESET_ALL} Show mix history")
        print(f"{Fore.WHITE}5.{Style.RESET_ALL} Library statistics")
        print(f"{Fore.WHITE}6.{Style.RESET_ALL} Settings")
        print(f"{Fore.WHITE}q.{Style.RESET_ALL} Quit")
    
    def _search_and_select_track(self):
        """Search for and select a track."""
        query = input(f"{Fore.CYAN}Enter track name or artist: {Style.RESET_ALL}").strip()
        
        if not query:
            self._print_error("Please enter a search query.")
            return
        
        tracks = self.db.search_tracks(query, limit=10)
        
        if not tracks:
            self._print_error(f"No tracks found matching '{query}'")
            return
        
        print(f"\n{Fore.GREEN}Found {len(tracks)} tracks:{Style.RESET_ALL}")
        for i, track in enumerate(tracks, 1):
            print(f"{Fore.WHITE}{i}.{Style.RESET_ALL} {track['title']} - {track['artist']}")
            print(f"    {Fore.YELLOW}Tempo: {track['tempo']:.1f} BPM, Key: {track['key']} {track['mode']}{Style.RESET_ALL}")
        
        try:
            choice = int(input(f"\n{Fore.CYAN}Select track (1-{len(tracks)}): {Style.RESET_ALL}")) - 1
            if 0 <= choice < len(tracks):
                self.current_track = tracks[choice]
                self._print_success(f"Selected: {self.current_track['title']} - {self.current_track['artist']}")
            else:
                self._print_error("Invalid selection.")
        except ValueError:
            self._print_error("Please enter a valid number.")
    
    def _get_recommendations(self):
        """Get recommendations for the current track."""
        if not self.current_track:
            self._print_error("No track selected. Please search and select a track first.")
            return
        
        print(f"\n{Fore.GREEN}Getting recommendations for:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{self.current_track['title']} - {self.current_track['artist']}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Tempo: {self.current_track['tempo']:.1f} BPM, Key: {self.current_track['key']} {self.current_track['mode']}{Style.RESET_ALL}")
        
        # Get recommendations
        recommendations = self.recommendation_engine.get_recommendations(
            self.current_track['title'], count=5
        )
        
        if not recommendations:
            self._print_error("No recommendations found.")
            return
        
        print(f"\n{Fore.GREEN}Top {len(recommendations)} recommendations:{Style.RESET_ALL}")
        for i, (track, score) in enumerate(recommendations, 1):
            print(f"{Fore.WHITE}{i}.{Style.RESET_ALL} {track['title']} - {track['artist']}")
            print(f"    {Fore.YELLOW}Tempo: {track['tempo']:.1f} BPM, Key: {track['key']} {track['mode']}{Style.RESET_ALL}")
            print(f"    {Fore.CYAN}Compatibility Score: {score:.2f}{Style.RESET_ALL}")
        
        # Ask for feedback
        self._get_user_feedback(recommendations)
    
    def _get_user_feedback(self, recommendations: List):
        """Get user feedback on recommendations."""
        print(f"\n{Fore.CYAN}Feedback Options:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}a.{Style.RESET_ALL} Accept a recommendation")
        print(f"{Fore.WHITE}r.{Style.RESET_ALL} Reject a recommendation")
        print(f"{Fore.WHITE}s.{Style.RESET_ALL} Skip feedback")
        
        choice = input(f"{Fore.CYAN}Enter your choice: {Style.RESET_ALL}").strip().lower()
        
        if choice == 'a':
            try:
                track_num = int(input(f"{Fore.CYAN}Enter track number to accept (1-{len(recommendations)}): {Style.RESET_ALL}")) - 1
                if 0 <= track_num < len(recommendations):
                    track, score = recommendations[track_num]
                    self.recommendation_engine.add_feedback(
                        self.current_track['title'],
                        track['title'],
                        True,
                        score
                    )
                    self._print_success(f"Accepted recommendation: {track['title']}")
                else:
                    self._print_error("Invalid track number.")
            except ValueError:
                self._print_error("Please enter a valid number.")
        
        elif choice == 'r':
            try:
                track_num = int(input(f"{Fore.CYAN}Enter track number to reject (1-{len(recommendations)}): {Style.RESET_ALL}")) - 1
                if 0 <= track_num < len(recommendations):
                    track, score = recommendations[track_num]
                    self.recommendation_engine.add_feedback(
                        self.current_track['title'],
                        track['title'],
                        False,
                        score
                    )
                    self._print_success(f"Rejected recommendation: {track['title']}")
                else:
                    self._print_error("Invalid track number.")
            except ValueError:
                self._print_error("Please enter a valid number.")
    
    def _create_mix(self):
        """Create a mix from recommendations."""
        if not self.current_track:
            self._print_error("No track selected. Please search and select a track first.")
            return
        
        # Get recommendations
        recommendations = self.recommendation_engine.get_recommendations(
            self.current_track['title'], count=3
        )
        
        if not recommendations:
            self._print_error("No recommendations available for mixing.")
            return
        
        print(f"\n{Fore.GREEN}Creating mix with:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}1.{Style.RESET_ALL} {self.current_track['title']} - {self.current_track['artist']}")
        
        track_paths = [self.current_track['file_path']]
        
        for i, (track, score) in enumerate(recommendations, 2):
            print(f"{Fore.WHITE}{i}.{Style.RESET_ALL} {track['title']} - {track['artist']}")
            track_paths.append(track['file_path'])
        
        # Create mix
        output_path = f"mix_{int(time.time())}.mp3"
        
        print(f"\n{Fore.YELLOW}Creating mix...{Style.RESET_ALL}")
        success = self.mixing_engine.create_playlist_mix(track_paths, output_path)
        
        if success:
            self._print_success(f"Mix created successfully: {output_path}")
            self.mix_history.append({
                'tracks': [self.current_track] + [track for track, _ in recommendations],
                'output_path': output_path,
                'timestamp': time.time()
            })
        else:
            self._print_error("Failed to create mix.")
    
    def _show_mix_history(self):
        """Show mix history."""
        if not self.mix_history:
            self._print_error("No mix history available.")
            return
        
        print(f"\n{Fore.GREEN}Mix History:{Style.RESET_ALL}")
        for i, mix in enumerate(self.mix_history, 1):
            print(f"{Fore.WHITE}{i}.{Style.RESET_ALL} {mix['output_path']}")
            print(f"    {Fore.YELLOW}Tracks: {len(mix['tracks'])}{Style.RESET_ALL}")
            print(f"    {Fore.CYAN}Created: {time.ctime(mix['timestamp'])}{Style.RESET_ALL}")
    
    def _show_library_stats(self):
        """Show library statistics."""
        stats = self.db.get_library_stats()
        self._print_library_stats(stats)
    
    def _print_library_stats(self, stats: Dict):
        """Print library statistics."""
        print(f"\n{Fore.GREEN}Library Statistics:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Total Tracks:{Style.RESET_ALL} {stats['total_tracks']}")
        print(f"{Fore.WHITE}Unique Artists:{Style.RESET_ALL} {stats['unique_artists']}")
        print(f"{Fore.WHITE}Unique Albums:{Style.RESET_ALL} {stats['unique_albums']}")
        print(f"{Fore.WHITE}Tempo Range:{Style.RESET_ALL} {stats['tempo_min']:.1f} - {stats['tempo_max']:.1f} BPM")
        print(f"{Fore.WHITE}Average Tempo:{Style.RESET_ALL} {stats['tempo_avg']:.1f} BPM")
        
        print(f"\n{Fore.GREEN}Key Distribution:{Style.RESET_ALL}")
        for key, count in sorted(stats['key_distribution'].items()):
            print(f"{Fore.WHITE}{key}:{Style.RESET_ALL} {count} tracks")
    
    def _settings_menu(self):
        """Show settings menu."""
        print(f"\n{Fore.GREEN}Settings:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}1.{Style.RESET_ALL} Retrain ML model")
        print(f"{Fore.WHITE}2.{Style.RESET_ALL} Clear mix history")
        print(f"{Fore.WHITE}b.{Style.RESET_ALL} Back to main menu")
        
        choice = input(f"{Fore.CYAN}Enter your choice: {Style.RESET_ALL}").strip().lower()
        
        if choice == '1':
            self._print_yellow("Retraining ML model...")
            self.recommendation_engine.retrain_model()
            self._print_success("ML model retrained successfully.")
        elif choice == '2':
            self.mix_history.clear()
            self._print_success("Mix history cleared.")
        elif choice == 'b':
            return
        else:
            self._print_error("Invalid choice.")
    
    def _print_success(self, message: str):
        """Print success message."""
        print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")
    
    def _print_error(self, message: str):
        """Print error message."""
        print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")
    
    def _print_yellow(self, message: str):
        """Print yellow message."""
        print(f"{Fore.YELLOW}{message}{Style.RESET_ALL}")
    
    def _print_goodbye(self):
        """Print goodbye message."""
        print(f"\n{Fore.GREEN}Thanks for using AI Music Mixer! 🎵{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Keep mixing and discovering great music!{Style.RESET_ALL}\n")
