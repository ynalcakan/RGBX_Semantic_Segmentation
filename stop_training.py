#!/usr/bin/env python3
import os
import sys
import signal
import argparse
import time
from pathlib import Path

def create_stop_flag():
    """Create a flag file that the training script can check to know when to stop"""
    flag_file = Path("stop_training.flag")
    flag_file.touch()
    print(f"Created stop flag: {flag_file.absolute()}")
    print("Training will stop after the current epoch completes.")
    print("The model checkpoint will be saved.")
    return flag_file

def find_training_process():
    """Find the Python process running train.py"""
    import psutil
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('train.py' in arg for arg in cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Safely stop training and save checkpoint")
    parser.add_argument("--send-signal", action="store_true", 
                        help="Also send SIGUSR1 to the training process if found")
    args = parser.parse_args()
    
    flag_file = create_stop_flag()
    
    if args.send_signal:
        try:
            training_proc = find_training_process()
            if training_proc:
                print(f"Sending signal to training process (PID: {training_proc.pid})")
                os.kill(training_proc.pid, signal.SIGUSR1)
            else:
                print("No training process found.")
        except Exception as e:
            print(f"Error sending signal: {e}")
    
    print("\nTo resume training later, delete the flag file with:")
    print(f"rm {flag_file.absolute()}")

if __name__ == "__main__":
    main() 