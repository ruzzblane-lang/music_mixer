# Quick Start Guide - AI Music Mixer

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the installation:**
   ```bash
   python test_installation.py
   ```

## Basic Usage

### 1. Scan Your Music Library

First, scan your music library to analyze all tracks:

```bash
python main.py scanmusic /path/to/your/music/library
```

Example:
```bash
python main.py scanmusic ~/Music
```

This will:
- Find all audio files (MP3, WAV, FLAC, etc.)
- Extract audio features (BPM, key, energy, etc.)
- Store metadata in SQLite database
- Show progress with a progress bar

### 2. Get Track Recommendations

Get AI-powered recommendations for a specific track:

```bash
python main.py recommend "track_name"
```

Example:
```bash
python main.py recommend "Bohemian Rhapsody"
```

### 3. Interactive Mixing Session

Start an interactive mixing session:

```bash
python main.py mix --interactive
```

This opens a menu where you can:
- Search and select tracks
- Get recommendations
- Create seamless mixes
- Provide feedback to improve recommendations

## Command Reference

### `scanmusic` Command
```bash
python main.py scanmusic [OPTIONS] MUSIC_PATH

Options:
  -f, --force    Force re-scan of already analyzed tracks
  -v, --verbose  Enable verbose output
```

### `recommend` Command
```bash
python main.py recommend [OPTIONS] TRACK_NAME

Options:
  -c, --count INTEGER  Number of recommendations to show (default: 5)
  -v, --verbose        Show detailed compatibility scores
```

### `mix` Command
```bash
python main.py mix [OPTIONS]

Options:
  -i, --interactive  Start interactive mixing session
```

### `status` Command
```bash
python main.py status
```
Shows system status and database statistics.

## Example Workflow

1. **Scan your music library:**
   ```bash
   python main.py scanmusic ~/Music
   ```

2. **Check what was found:**
   ```bash
   python main.py status
   ```

3. **Get recommendations for a track:**
   ```bash
   python main.py recommend "Hotel California"
   ```

4. **Start interactive session:**
   ```bash
   python main.py mix --interactive
   ```

5. **In the interactive session:**
   - Search for "Hotel California"
   - Select the track
   - Get recommendations
   - Accept/reject recommendations to train the AI
   - Create a mix

## Features

### Audio Analysis
- **Tempo (BPM)**: Automatic tempo detection
- **Key Detection**: Musical key and mode identification
- **Energy Analysis**: RMS energy and loudness
- **Spectral Features**: Brightness, rolloff, bandwidth
- **Rhythm Analysis**: Onset strength and tempo stability

### Smart Recommendations
- **Rule-based**: BPM, key, and energy matching
- **ML-powered**: k-NN model with feature vectors
- **Learning**: Improves with user feedback
- **Compatibility Scoring**: 0-1 compatibility scores

### Mixing Engine
- **Crossfading**: Seamless track transitions
- **Beat Alignment**: Aligns transitions to beats
- **EQ Ducking**: Prevents bass conflicts
- **Transition Effects**: Reverb, echo, filter sweeps

### User Interface
- **CLI Interface**: Command-line with colored output
- **Interactive Mode**: Menu-driven mixing sessions
- **Progress Tracking**: Visual progress bars
- **Error Handling**: Graceful error handling

## Troubleshooting

### Common Issues

1. **"No tracks found in library"**
   - Make sure you've run `scanmusic` first
   - Check that your music directory contains supported audio formats

2. **"Module not found" errors**
   - Run `pip install -r requirements.txt`
   - Make sure you're using Python 3.8+

3. **Audio processing errors**
   - Ensure audio files are not corrupted
   - Check file permissions
   - Try with different audio formats

4. **Database errors**
   - Delete `data/music_library.db` and re-scan
   - Check disk space

### Supported Audio Formats
- MP3
- WAV
- FLAC
- M4A
- AAC
- OGG
- WMA

### System Requirements
- Python 3.8+
- 4GB+ RAM (for large music libraries)
- FFmpeg (for audio processing)

## Next Steps

1. **Customize Recommendations**: Provide feedback to train the AI
2. **Create Playlists**: Build themed playlists with AI assistance
3. **Export Mixes**: Save your AI-generated mixes
4. **Advanced Features**: Explore tempo matching and key transitions

## Getting Help

- Check the logs for detailed error messages
- Use `--verbose` flag for more detailed output
- Run `python test_installation.py` to verify setup

Happy mixing! 🎵
