import os
import sys

os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"

from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, 
                             QWidget, QPushButton, QLineEdit, QLabel, 
                             QScrollArea, QFrame, QMessageBox, QDialog,
                             QTextEdit, QDialogButtonBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont, QIcon, QCursor

# Import core modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from core.speech_to_text import SpeechRecognizer
from core.song_matcher import SongMatcher
from core.corpus_manager import CorpusManager

class LyricsDialog(QDialog):
    """Dialog to display song lyrics"""
    def __init__(self, song_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{song_data.get('title', 'Unknown')} - {song_data.get('artist', 'Unknown Artist')}")
        
        # Make the dialog larger and center it on screen
        self.setGeometry(150, 100, 900, 700)
        
        # Alternative: Set minimum size and make it resizable
        self.setMinimumSize(800, 600)
        self.resize(900, 700)
        
        layout = QVBoxLayout()
        
        # Song info header
        info_frame = QFrame()
        info_frame.setStyleSheet("background-color: #1DB954; border-radius: 10px; padding: 15px;")
        info_layout = QVBoxLayout()
        
        title_label = QLabel(song_data.get('title', 'Unknown Title'))
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        
        artist_label = QLabel(f"by {song_data.get('artist', 'Unknown Artist')}")
        artist_label.setFont(QFont("Arial", 14))
        artist_label.setStyleSheet("color: white;")
        
        if song_data.get('album'):
            album_label = QLabel(f"Album: {song_data['album']}")
            album_label.setFont(QFont("Arial", 12))
            album_label.setStyleSheet("color: white;")
            info_layout.addWidget(album_label)
        
        info_layout.addWidget(title_label)
        info_layout.addWidget(artist_label)
        info_frame.setLayout(info_layout)
        layout.addWidget(info_frame)
        
        # Lyrics display
        lyrics_label = QLabel("Lyrics:")
        lyrics_label.setFont(QFont("Arial", 14, QFont.Bold))
        lyrics_label.setStyleSheet("color: #333; margin-top: 15px; margin-bottom: 5px;")
        layout.addWidget(lyrics_label)
        
        self.lyrics_text = QTextEdit()
        self.lyrics_text.setReadOnly(True)
        self.lyrics_text.setFont(QFont("Arial", 20))  # Much larger font
        self.lyrics_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 2px solid #ddd;
                border-radius: 8px;
                padding: 25px;
                line-height: 2.0;
                font-size: 14px;
            }
        """)
        
        # Set lyrics content with formatting
        lyrics = song_data.get('lyric', 'Lyrics not available')
        if lyrics and lyrics.strip():
            formatted_lyrics = self.format_lyrics(lyrics)
            self.lyrics_text.setPlainText(formatted_lyrics)
        else:
            self.lyrics_text.setPlainText("Lyrics not available for this song.")
        
        layout.addWidget(self.lyrics_text)
        
        # Close button
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet("""
            QPushButton {
                background-color: #1DB954;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1ED760;
            }
        """)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        self.setStyleSheet("background-color: white;")
    
    def format_lyrics(self, lyrics):
        """Format lyrics with better spacing and structure"""
        if not lyrics:
            return "Lyrics not available"
        
        # Split lyrics into lines
        lines = lyrics.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines but preserve intentional spacing
            if not line:
                formatted_lines.append("")
                continue
            
            # Detect common song structure markers and add extra spacing
            line_lower = line.lower()
            
            # Check if this line is a section marker
            is_section_marker = any(marker in line_lower for marker in [
                'verse', 'chorus', 'bridge', 'pre-chorus', 'outro', 'intro',
                '[verse', '[chorus', '[bridge', '[pre-chorus', '[outro', '[intro',
                '(verse', '(chorus', '(bridge', '(pre-chorus', '(outro', '(intro'
            ])
            
            # Add extra spacing before section markers (except for the first line)
            if is_section_marker and formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")  # Add blank line before section
            
            formatted_lines.append(line)
            
            # Add spacing after section markers
            if is_section_marker:
                formatted_lines.append("")  # Add blank line after section marker
        
        # Join lines and add some general formatting improvements
        formatted_text = '\n'.join(formatted_lines)
        
        # Add extra spacing between what appears to be different sections
        # (when there are multiple consecutive empty lines, reduce to just two)
        while '\n\n\n' in formatted_text:
            formatted_text = formatted_text.replace('\n\n\n', '\n\n')
        
        return formatted_text

class AudioProcessingThread(QThread):
    """Thread for handling audio processing without blocking UI"""
    audio_processed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.speech_recognizer = SpeechRecognizer()
    
    def run(self):
        try:
            text = self.speech_recognizer.listen_and_convert()
            if text:
                self.audio_processed.emit(text)
            else:
                self.error_occurred.emit("Could not understand audio")
        except Exception as e:
            self.error_occurred.emit(f"Audio processing error: {str(e)}")

class SongResultWidget(QFrame):
    """Widget to display individual song results - now clickable"""
    def __init__(self, song_data, score, parent_window):
        super().__init__()
        self.song_data = song_data
        self.parent_window = parent_window
        
        self.setStyleSheet("""
            QFrame {
                background-color: #b5bda5;
                border-radius: 8px;
                margin: 5px;
                padding: 15px;
            }
            QFrame:hover {
                background-color: #a8b393;
            }
        """)
        
        # Make it clickable
        self.setCursor(QCursor(Qt.PointingHandCursor))
        
        layout = QVBoxLayout()
        
        # Song title
        title_label = QLabel(song_data.get('title', 'Unknown Title'))
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: #000000;")
        
        # Artist
        artist_label = QLabel(f"Artist: {song_data.get('artist', 'Unknown Artist')}")
        artist_label.setFont(QFont("Arial", 10))
        artist_label.setStyleSheet("color: #333333;")
        
        # Album (if available)
        if song_data.get('album'):
            album_label = QLabel(f"Album: {song_data['album']}")
            album_label.setFont(QFont("Arial", 10))
            album_label.setStyleSheet("color: #333333;")
            layout.addWidget(album_label)
        
        # Match score
        score_label = QLabel(f"Match Score: {score:.2%}")
        score_label.setFont(QFont("Arial", 10))
        score_label.setStyleSheet("color: #000000;")
        
        # Click to view lyrics hint
        hint_label = QLabel("Click to view lyrics")
        hint_label.setFont(QFont("Arial", 9))
        hint_label.setStyleSheet("color: #666666; font-style: italic;")
        
        layout.addWidget(title_label)
        layout.addWidget(artist_label)
        layout.addWidget(score_label)
        layout.addWidget(hint_label)
        
        self.setLayout(layout)
    
    def mousePressEvent(self, event):
        """Handle mouse click to show lyrics"""
        if event.button() == Qt.LeftButton:
            self.show_lyrics()
        super().mousePressEvent(event)
    
    def show_lyrics(self):
        """Show lyrics dialog"""
        dialog = LyricsDialog(self.song_data, self.parent_window)
        dialog.exec_()

class MainWindow(QMainWindow):
    """Main application window with Spotify-inspired design"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Song Finder System")
        self.setGeometry(100, 100, 1000, 700)
        
        # Initialize core components - but don't create song matcher yet!
        self.corpus_manager = CorpusManager()
        self.song_matcher = None  # Will be initialized after loading corpus
        self.audio_thread = None
        
        # Setup UI
        self.setup_ui()
        self.apply_styles()
        
        # Load corpus data FIRST, then initialize song matcher
        self.load_corpus_data()
    
    def setup_ui(self):
        """Setup the user interface components in a single row header"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Top row layout: logo + search + buttons
        top_layout = QHBoxLayout()
        top_layout.setSpacing(10)

        # Logo
        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'icon', 'logo.png')
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        top_layout.addWidget(logo_label)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type or speak lyrics to find songs...")
        self.search_input.setFont(QFont("Arial", 14))
        self.search_input.returnPressed.connect(self.search_songs)
        top_layout.addWidget(self.search_input, 4)

        # Search button
        self.search_button = QPushButton()
        search_icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'icon', 'search.png')
        if os.path.exists(search_icon_path):
            self.search_button.setIcon(QIcon(search_icon_path))
        self.search_button.clicked.connect(self.search_songs)
        top_layout.addWidget(self.search_button, 0)

        # Microphone button
        self.mic_button = QPushButton()
        mic_icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'icon', 'microphone.png')
        if os.path.exists(mic_icon_path):
            self.mic_button.setIcon(QIcon(mic_icon_path))
        self.mic_button.clicked.connect(self.start_voice_search)
        top_layout.addWidget(self.mic_button, 0)

        main_layout.addLayout(top_layout)

        # Results area
        self.results_area = self.create_results_section()
        main_layout.addWidget(self.results_area)

        # Status bar
        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet("color: #B3B3B3; padding: 10px;")
        main_layout.addWidget(self.status_label)
    
    def create_results_section(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("background-color: #b5bda5;")
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_widget.setLayout(self.results_layout)
        initial_label = QLabel("Loading song database...")
        initial_label.setAlignment(Qt.AlignCenter)
        initial_label.setFont(QFont("Arial", 14))
        initial_label.setStyleSheet("color: #000000; padding: 50px;")
        self.results_layout.addWidget(initial_label)
        scroll_area.setWidget(self.results_widget)
        return scroll_area
    
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QLineEdit {
                background-color: #2A2A2A;
                border: 2px solid #404040;
                border-radius: 25px;
                padding: 12px 20px;
                color: #FFFFFF;
                font-size: 14px;
            }
            QLineEdit:focus { border-color: #1DB954; }
            QPushButton {
                background-color: #1DB954;
                border: none;
                border-radius: 20px;
                padding: 12px;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 12px;
                min-width: 40px;
            }
            QPushButton:hover { background-color: #1ED760; }
            QPushButton:pressed { background-color: #169C46; }
            QScrollArea { border: none; }
        """)
    
    def load_corpus_data(self):
        """Load corpus data FIRST, then initialize song matcher"""
        try:
            self.status_label.setText("Loading song database...")
            data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dictionary')
            
            # Load the corpus first
            self.corpus_manager.load_corpus(data_path)
            songs_count = len(self.corpus_manager.get_all_songs())
            
            self.status_label.setText("Processing songs for matching...")
            
            # Now initialize the song matcher with the loaded corpus
            self.song_matcher = SongMatcher(self.corpus_manager)
            
            # Update UI to show ready state
            self.update_ready_state(songs_count)
            
        except Exception as e:
            self.status_label.setText(f"Error loading corpus: {str(e)}")
            QMessageBox.warning(self, "Warning", f"Could not load song database: {str(e)}")
    
    def update_ready_state(self, songs_count):
        """Update UI to ready state after successful loading"""
        # Clear loading message
        for i in reversed(range(self.results_layout.count())):
            child = self.results_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        # Add ready message
        ready_label = QLabel("Start searching for songs by typing lyrics or using the microphone!")
        ready_label.setAlignment(Qt.AlignCenter)
        ready_label.setFont(QFont("Arial", 14))
        ready_label.setStyleSheet("color: #000000; padding: 50px;")
        self.results_layout.addWidget(ready_label)
        
        # Update status
        processed_songs = len(self.song_matcher.preprocessed_songs) if self.song_matcher else 0
        self.status_label.setText(f"Ready! Loaded {songs_count} songs, {processed_songs} ready for search")
    
    def search_songs(self):
        """Search for songs using the query"""
        if not self.song_matcher:
            QMessageBox.warning(self, "Warning", "Song database is still loading. Please wait.")
            return
            
        query = self.search_input.text().strip()
        if not query:
            return
            
        self.status_label.setText("Searching...")
        self.search_button.setEnabled(False)
        
        try:
            # Specify max_results to show more matches (or remove limit entirely with a large number)
            results = self.song_matcher.find_matches(query, max_results=100)  # Show up to 100 matches
            self.display_results(results, query)
        except Exception as e:
            self.status_label.setText(f"Search error: {str(e)}")
            QMessageBox.critical(self, "Error", f"Search failed: {str(e)}")
        finally:
            self.search_button.setEnabled(True)
    
    def start_voice_search(self):
        """Start voice search"""
        if not self.song_matcher:
            QMessageBox.warning(self, "Warning", "Song database is still loading. Please wait.")
            return
            
        if self.audio_thread and self.audio_thread.isRunning():
            return
            
        self.mic_button.setEnabled(False)
        self.mic_button.setText("")
        self.status_label.setText("Listening... Speak now!")
        self.audio_thread = AudioProcessingThread()
        self.audio_thread.audio_processed.connect(self.on_audio_processed)
        self.audio_thread.error_occurred.connect(self.on_audio_error)
        self.audio_thread.start()
    
    def on_audio_processed(self, text):
        self.search_input.setText(text)
        self.search_songs()
        self.mic_button.setEnabled(True)
        self.mic_button.setText("")
    
    def on_audio_error(self, error_message):
        self.status_label.setText(f"Audio error: {error_message}")
        self.mic_button.setEnabled(True)
        self.mic_button.setText("")
        QMessageBox.warning(self, "Audio Error", error_message)
    
    def display_results(self, results, query):
        """Display search results"""
        for i in reversed(range(self.results_layout.count())):
            child = self.results_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
                
        if not results:
            no_results_label = QLabel("No matches found.")
            no_results_label.setAlignment(Qt.AlignCenter)
            no_results_label.setFont(QFont("Arial", 16))
            no_results_label.setStyleSheet("background-color: #b5bda5; color: #000000; padding: 50px;")
            self.results_layout.addWidget(no_results_label)

            suggestion_label = QLabel("Try different lyrics or check your spelling.")
            suggestion_label.setAlignment(Qt.AlignCenter)
            suggestion_label.setFont(QFont("Arial", 12))
            suggestion_label.setStyleSheet("background-color: #b5bda5; color: #000000; padding: 10px;")
            self.results_layout.addWidget(suggestion_label)

            self.status_label.setText("No matches found")
        else:
            results_header = QLabel(f"Found {len(results)} match(es) for: \"{query}\" - Click any song to view lyrics")
            results_header.setFont(QFont("Arial", 14, QFont.Bold))
            results_header.setStyleSheet("background-color: #b5bda5; color: #000000; padding: 10px; margin-bottom: 10px;")
            self.results_layout.addWidget(results_header)
            
            for song_data, score in results:
                # Create a container frame for each result
                container = QFrame()
                container.setStyleSheet("""
                    QFrame {
                        background-color: #e0e5d1;
                        border-radius: 12px;
                        border: 2px solid #a8b393;
                        margin: 8px 0;
                        padding: 10px;
                    }
                """)
                
                container_layout = QVBoxLayout(container)
                
                # Add the actual song widget inside the container
                result_widget = SongResultWidget(song_data, score, self)
                container_layout.addWidget(result_widget)
                
                self.results_layout.addWidget(container)
                
            self.results_layout.addStretch()
            self.status_label.setText(f"Found {len(results)} matches")

    def closeEvent(self, event):
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.terminate()
            self.audio_thread.wait()
        event.accept()