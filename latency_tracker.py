# -*- coding: utf-8 -*-
"""
Latency tracking for voicebot performance monitoring
"""
import time
from typing import List, Optional
from colorama import Fore, Style


class LatencyTracker:
    """Track and report latency metrics for voicebot interactions"""

    def __init__(self):
        # Timestamps
        self.user_speech_start: Optional[float] = None
        self.user_speech_end: Optional[float] = None
        self.bot_response_start: Optional[float] = None
        self.bot_response_end: Optional[float] = None
        self.search_start: Optional[float] = None
        self.search_end: Optional[float] = None

        # Historical data
        self.response_times: List[float] = []
        self.search_times: List[float] = []
        self.interruptions: int = 0
        self.total_interactions: int = 0

    def start_user_speech(self):
        """Mark start of user speech"""
        self.user_speech_start = time.perf_counter()

    def end_user_speech(self):
        """Mark end of user speech"""
        self.user_speech_end = time.perf_counter()

    def start_bot_response(self):
        """Mark start of bot response and return latency from user speech end"""
        self.bot_response_start = time.perf_counter()

        # Calculate and return Time-to-First-Audio (TTFA)
        if self.user_speech_end and self.bot_response_start:
            ttfa_ms = (self.bot_response_start - self.user_speech_end) * 1000
            return ttfa_ms
        return None

    def end_bot_response(self, was_interrupted: bool = False):
        """Mark end of bot response"""
        self.bot_response_end = time.perf_counter()

        if was_interrupted:
            self.interruptions += 1

        # Calculate time-to-first-audio (TTFA) - critical latency metric
        if self.user_speech_end and self.bot_response_start:
            ttfa = (self.bot_response_start - self.user_speech_end) * 1000  # Convert to ms
            self.response_times.append(ttfa)
            self.total_interactions += 1

    def start_search(self):
        """Mark start of knowledge base search"""
        self.search_start = time.perf_counter()

    def end_search(self):
        """Mark end of knowledge base search"""
        self.search_end = time.perf_counter()

        if self.search_start:
            search_latency = (self.search_end - self.search_start) * 1000  # Convert to ms
            self.search_times.append(search_latency)

    def print_summary(self):
        """Print comprehensive latency summary"""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"Latency Summary")
        print(f"{'='*70}{Style.RESET_ALL}\n")

        print(f"{Fore.YELLOW}Interactions: {self.total_interactions}")
        print(f"Interruptions: {self.interruptions}{Style.RESET_ALL}\n")

        if self.response_times:
            avg_response = sum(self.response_times) / len(self.response_times)
            min_response = min(self.response_times)
            max_response = max(self.response_times)

            print(f"{Fore.GREEN}Response Time (Time-to-First-Audio):")
            print(f"  Average: {avg_response:.2f}ms")
            print(f"  Min: {min_response:.2f}ms")
            print(f"  Max: {max_response:.2f}ms{Style.RESET_ALL}\n")

        if self.search_times:
            avg_search = sum(self.search_times) / len(self.search_times)
            min_search = min(self.search_times)
            max_search = max(self.search_times)

            print(f"{Fore.MAGENTA}Knowledge Base Search Time:")
            print(f"  Average: {avg_search:.2f}ms")
            print(f"  Min: {min_search:.2f}ms")
            print(f"  Max: {max_search:.2f}ms")
            print(f"  Total Searches: {len(self.search_times)}{Style.RESET_ALL}\n")
