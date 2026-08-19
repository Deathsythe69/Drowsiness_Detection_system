"""Alerts Module.

Manages sound playback for drowsiness alerts. Synthesizes default audio if missing.
"""

import sys
import os
import wave
import struct
import math
from typing import Dict, Any

# Dynamic win32 winsound import
if sys.platform == "win32":
    import winsound
else:
    winsound = None

def generate_default_alarm(filepath: str):
    """Synthesizes a 1-second pulsing alert sound as a WAV file.
    
    Uses only standard libraries (wave, struct, math).
    
    Args:
        filepath: Path to save the output WAV file.
    """
    if os.path.exists(filepath):
        return
        
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    sample_rate = 44100
    duration = 1.0  # seconds
    frequency = 880.0  # A5 note
    num_samples = int(sample_rate * duration)
    
    with wave.open(filepath, 'w') as wav_file:
        # Channels: 1 (mono), Sample Width: 2 bytes (16-bit), Rate: 44100Hz, Frames, Compression: None
        wav_file.setparams((1, 2, sample_rate, num_samples, 'NONE', 'not compressed'))
        
        for i in range(num_samples):
            # Standard sine wave
            val = math.sin(2.0 * math.pi * frequency * (i / sample_rate))
            # Pulse shape (amplitude modulation at 5Hz to sound like an alarm)
            pulse = math.sin(2.0 * math.pi * 5.0 * (i / sample_rate))
            val = val * (0.5 + 0.5 * pulse)
            
            # Scale to 16-bit signed integer limits
            int_val = int(val * 32767)
            data = struct.pack('<h', int_val)
            wav_file.writeframesraw(data)

class AlertManager:
    """Class to control alert sound playing and stopping."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Alert Manager.
        
        Args:
            config: Config dictionary.
        """
        self.update_config(config)
        self.is_playing = False
        self.sound_enabled = True

        # Pre-generate default alarm file if it doesn't exist
        try:
            generate_default_alarm(self.sound_file)
        except Exception as e:
            print(f"Warning: Failed to generate default alarm file: {e}")

    def update_config(self, config: Dict[str, Any]):
        """Update configurations."""
        self.sound_file = config.get("alerts", {}).get("sound_file", "assets/alarm.wav")

    def play_alarm(self):
        """Play sound loop in the background."""
        if not self.sound_enabled or self.is_playing:
            return
            
        if sys.platform == "win32" and winsound is not None:
            try:
                abs_path = os.path.abspath(self.sound_file)
                if os.path.exists(abs_path):
                    # Play WAV on a background thread loop
                    winsound.PlaySound(
                        abs_path, 
                        winsound.SND_FILENAME | winsound.SND_LOOP | winsound.SND_ASYNC
                    )
                    self.is_playing = True
                else:
                    # Windows system beep fallback
                    winsound.Beep(1000, 1000)
            except Exception as e:
                print(f"Error playing sound alert: {e}")
        else:
            # Console bell print fallback for non-Windows
            print("\a", end="")
            self.is_playing = True

    def stop_alarm(self):
        """Stop background sound loop."""
        if not self.is_playing:
            return
            
        if sys.platform == "win32" and winsound is not None:
            try:
                # Stop any looping playback
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception as e:
                print(f"Error stopping sound alert: {e}")
                
        self.is_playing = False
