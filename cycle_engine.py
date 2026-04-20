import numpy as np
from scipy import signal as scipy_signal
from scipy.optimize import curve_fit
import json
from datetime import datetime, timedelta

class CycleDetector:
    """
    Euler's Formula Based Cycle Detection Engine
    e^(iθ) = cos(θ) + i·sin(θ)
    
    Fourier Transform:
    F(ω) = Σ f(t) · e^(-iωt)
         = Σ f(t) · [cos(ωt) - i·sin(ωt)]
    """
    
    def __init__(self):
        self.cycles = []
        self.manual_points = []
        self.timeframe = "5m"
        self.analysis_result = {}
    
    def set_timeframe(self, tf):
        """Set timeframe: 5m, 15m, 1h, 4h, 1d"""
        self.timeframe = tf
        # Minutes per bar
        self.tf_minutes = {
            "1m": 1, "5m": 5, "15m": 15, 
            "1h": 60, "4h": 240, "1d": 1440
        }.get(tf, 5)
    
    def add_manual_point(self, datetime_str, price, point_type):
        """
        Add manual top/bottom point
        point_type: 'top' or 'bottom'
        datetime_str: '2025-04-20 09:30'
        """
        self.manual_points.append({
            'datetime': datetime_str,
            'price': float(price),
            'type': point_type,
            'timestamp': datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
        })
        # Sort by time
        self.manual_points.sort(key=lambda x: x['timestamp'])
    
    def clear_points(self):
        """Clear all manual points"""
        self.manual_points = []
        self.cycles = []
        self.analysis_result = {}
    
    def _generate_price_array(self):
        """
        Convert manual top/bottom points to continuous price array
        using linear interpolation between points
        """
        if len(self.manual_points) < 2:
            return np.array([]), np.array([])
        
        points = self.manual_points
        
        # Calculate total bars between first and last point
        total_minutes = (points[-1]['timestamp'] - points[0]['timestamp']).total_seconds() / 60
        total_bars = int(total_minutes / self.tf_minutes)
        
        if total_bars < 2:
            return np.array([]), np.array([])
        
        # Create time array
        time_array = np.linspace(0, total_bars - 1, total_bars)
        price_array = np.zeros(total_bars)
        
        # Interpolate between manual points
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            
            # Bar index for each point
            mins1 = (p1['timestamp'] - points[0]['timestamp']).total_seconds() / 60
            mins2 = (p2['timestamp'] - points[0]['timestamp']).total_seconds() / 60
            
            bar1 = int(mins1 / self.tf_minutes)
            bar2 = int(mins2 / self.tf_minutes)
            
            bar1 = max(0, min(bar1, total_bars - 1))
            bar2 = max(0, min(bar2, total_bars - 1))
            
            if bar2 > bar1:
                # Linear interpolation
                for j in range(bar1, bar2 + 1):
                    if j < total_bars:
                        ratio = (j - bar1) / (bar2 - bar1)
                        price_array[j] = p1['price'] + ratio * (p2['price'] - p1['price'])
        
        return time_array, price_array
    
    def _discrete_fourier_transform(self, price_data):
        """
        Manual DFT using Euler's Formula
        F(k) = Σ x(n) · e^(-i·2π·k·n/N)
             = Σ x(n) · [cos(2πkn/N) - i·sin(2πkn/N)]
        """
        N = len(price_data)
        
        # Remove trend (detrend)
        trend = np.linspace(price_data[0], price_data[-1], N)
        detrended = price_data - trend
        
        # DFT
        frequencies = []
        amplitudes = []
        phases = []
        
        for k in range(N // 2):
            real_sum = 0.0
            imag_sum = 0.0
            
            for n in range(N):
                # Euler's formula: e^(-i·2π·k·n/N)
                angle = 2.0 * np.pi * k * n / N
                real_sum += detrended[n] * np.cos(angle)   # Real part
                imag_sum -= detrended[n] * np.sin(angle)   # Imaginary part
            
            # Amplitude = |F(k)| = sqrt(real² + imag²)
            amplitude = np.sqrt(real_sum**2 + imag_sum**2) / N
            
            # Phase = atan2(imag, real)
            phase = np.arctan2(imag_sum, real_sum)
            
            # Frequency in cycles per bar
            freq = k / N
            
            # Period in bars
            period = N / k if k > 0 else float('inf')
            
            frequencies.append(freq)
            amplitudes.append(amplitude)
            phases.append(phase)
        
        return frequencies, amplitudes, phases
    
    def _scipy_fft(self, price_data):
        """
        Fast Fourier Transform using scipy (faster for large data)
        Based on same Euler formula but O(NlogN) algorithm
        """
        N = len(price_data)
        
        # Detrend
        trend = np.linspace(price_data[0], price_data[-1], N)
        detrended = price_data - trend
        
        # Apply window function (reduces spectral leakage)
        window = np.hanning(N)
        windowed = detrended * window
        
        # FFT
        fft_result = np.fft.fft(windowed)
        fft_freq = np.fft.fftfreq(N)
        
        # Only positive frequencies
        positive_mask = fft_freq > 0
        freqs = fft_freq[positive_mask]
        amps = np.abs(fft_result[positive_mask]) * 2 / N
        phases = np.angle(fft_result[positive_mask])
        
        return freqs, amps, phases
    
    def detect_cycles(self, method="fft", min_period=3, max_period=500, top_n=10):
        """
        Main cycle detection function
        
        method: 'fft' (fast) or 'dft' (manual Euler)
        min_period: minimum cycle period in bars
        max_period: maximum cycle period in bars
        top_n: how many top cycles to return
        """
        time_arr, price_arr = self._generate_price_array()
        
        if len(price_arr) < 4:
            return {"error": "Need at least 2 manual points"}
        
        N = len(price_arr)
        
        # Choose method
        if method == "dft" and N < 500:
            freqs, amps, phases = self._discrete_fourier_transform(price_arr)
        else:
            freqs, amps, phases = self._scipy_fft(price_arr)
        
        # Convert to periods and filter
        detected_cycles = []
        
        for i in range(len(freqs)):
            if freqs[i] > 0:
                period_bars = 1.0 / freqs[i]
                
                # Convert to time
                period_minutes = period_bars * self.tf_minutes
                period_hours = period_minutes / 60
                period_days = period_hours / 24
                
                if min_period <= period_bars <= max_period:
                    # Time string
                    if period_minutes < 60:
                        time_str = f"{period_minutes:.0f} min"
                    elif period_hours < 24:
                        time_str = f"{period_hours:.1f} hours"
                    else:
                        time_str = f"{period_days:.1f} days"
                    
                    detected_cycles.append({
                        'period_bars': round(period_bars, 1),
                        'period_time': time_str,
                        'period_minutes': round(period_minutes, 1),
                        'amplitude': round(float(amps[i]), 4),
                        'phase': round(float(phases[i]), 4),
                        'phase_degrees': round(float(np.degrees(phases[i])), 1),
                        'frequency': round(float(freqs[i]), 6),
                        'power': round(float(amps[i]**2), 6),
                        'strength_pct': 0  # Will calculate below
                    })
        
        # Sort by amplitude (strongest first)
        detected_cycles.sort(key=lambda x: x['amplitude'], reverse=True)
        
        # Calculate strength percentage
        total_power = sum(c['power'] for c in detected_cycles)
        if total_power > 0:
            for c in detected_cycles:
                c['strength_pct'] = round(c['power'] / total_power * 100, 1)
        
        # Top N cycles
        top_cycles = detected_cycles[:top_n]
        
        self.cycles = top_cycles
        
        return top_cycles
    
    def reconstruct_signal(self, num_cycles=5):
        """
        Reconstruct price using top N cycles
        Using inverse Euler: x(t) = Σ Ak · cos(2π·fk·t + φk)
        """
        time_arr, price_arr = self._generate_price_array()
        
        if len(price_arr) < 4 or len(self.cycles) == 0:
            return [], [], []
        
        N = len(price_arr)
        
        # Trend line
        trend = np.linspace(price_arr[0], price_arr[-1], N)
        
        # Reconstruct from top cycles
        cycles_to_use = min(num_cycles, len(self.cycles))
        reconstructed = np.zeros(N)
        
        for i in range(cycles_to_use):
            c = self.cycles[i]
            freq = c['frequency']
            amp = c['amplitude']
            phase = c['phase']
            
            # Euler's formula reconstruction
            # x(t) = A · cos(2π·f·t + φ)
            # = A · Re[e^(i(2π·f·t + φ))]
            for t in range(N):
                reconstructed[t] += amp * np.cos(2 * np.pi * freq * t + phase)
        
        # Add trend back
        reconstructed += trend
        
        return time_arr.tolist(), price_arr.tolist(), reconstructed.tolist()
    
    def predict_future(self, bars_ahead=50, num_cycles=5):
        """
        Project future price using detected cycles
        
        Future = Trend Extension + Σ Cycle Projections
        """
        time_arr, price_arr = self._generate_price_array()
        
        if len(price_arr) < 4 or len(self.cycles) == 0:
            return [], []
        
        N = len(price_arr)
        
        # Extend trend
        trend_slope = (price_arr[-1] - price_arr[0]) / N
        
        future_time = list(range(N, N + bars_ahead))
        future_price = np.zeros(bars_ahead)
        
        cycles_to_use = min(num_cycles, len(self.cycles))
        
        for i in range(cycles_to_use):
            c = self.cycles[i]
            freq = c['frequency']
            amp = c['amplitude']
            phase = c['phase']
            
            for t_idx in range(bars_ahead):
                t = N + t_idx
                # Euler's formula projection
                future_price[t_idx] += amp * np.cos(2 * np.pi * freq * t + phase)
        
        # Add trend extension
        for t_idx in range(bars_ahead):
            t = N + t_idx
            future_price[t_idx] += price_arr[-1] + trend_slope * (t_idx + 1)
        
        # Find future tops and bottoms
        future_tops = []
        future_bottoms = []
        
        for i in range(1, bars_ahead - 1):
            if future_price[i] > future_price[i-1] and future_price[i] > future_price[i+1]:
                future_minutes = (N + i) * self.tf_minutes
                start_time = self.manual_points[0]['timestamp']
                future_dt = start_time + timedelta(minutes=future_minutes)
                future_tops.append({
                    'bar': N + i,
                    'price': round(float(future_price[i]), 2),
                    'datetime': future_dt.strftime("%Y-%m-%d %H:%M"),
                    'type': 'top'
                })
            elif future_price[i] < future_price[i-1] and future_price[i] < future_price[i+1]:
                future_minutes = (N + i) * self.tf_minutes
                start_time = self.manual_points[0]['timestamp']
                future_dt = start_time + timedelta(minutes=future_minutes)
                future_bottoms.append({
                    'bar': N + i,
                    'price': round(float(future_price[i]), 2),
                    'datetime': future_dt.strftime("%Y-%m-%d %H:%M"),
                    'type': 'bottom'
                })
        
        return {
            'future_time': future_time,
            'future_price': future_price.tolist(),
            'future_tops': future_tops,
            'future_bottoms': future_bottoms,
            'bars_ahead': bars_ahead,
            'timeframe': self.timeframe
        }
    
    def get_cycle_confluence(self):
        """
        Find where multiple cycles align (confluence zones)
        These are HIGH PROBABILITY turning points!
        """
        if len(self.cycles) < 2:
            return []
        
        time_arr, price_arr = self._generate_price_array()
        N = len(price_arr)
        
        if N < 4:
            return []
        
        # For each bar, count how many cycles are near their top/bottom
        confluence_scores = np.zeros(N)
        
        for c in self.cycles[:5]:  # Top 5 cycles
            freq = c['frequency']
            amp = c['amplitude']
            phase = c['phase']
            
            for t in range(N):
                # Cycle value at this point
                cycle_val = np.cos(2 * np.pi * freq * t + phase)
                
                # Near top (cycle_val > 0.9) or near bottom (cycle_val < -0.9)
                if abs(cycle_val) > 0.85:
                    confluence_scores[t] += amp * abs(cycle_val)
        
        # Find top confluence points
        confluence_points = []
        for i in range(1, N - 1):
            if confluence_scores[i] > confluence_scores[i-1] and confluence_scores[i] > confluence_scores[i+1]:
                if confluence_scores[i] > np.mean(confluence_scores) * 1.5:
                    bar_minutes = i * self.tf_minutes
                    start_time = self.manual_points[0]['timestamp']
                    point_dt = start_time + timedelta(minutes=bar_minutes)
                    
                    # Determine if top or bottom
                    cycle_sum = 0
                    for c in self.cycles[:5]:
                        cycle_sum += np.cos(2 * np.pi * c['frequency'] * i + c['phase'])
                    
                    confluence_points.append({
                        'bar': i,
                        'datetime': point_dt.strftime("%Y-%m-%d %H:%M"),
                        'score': round(float(confluence_scores[i]), 4),
                        'type': 'top' if cycle_sum > 0 else 'bottom',
                        'price_estimate': round(float(price_arr[min(i, N-1)]), 2)
                    })
        
        confluence_points.sort(key=lambda x: x['score'], reverse=True)
        return confluence_points[:10]
    
    def full_analysis(self, bars_ahead=50, num_cycles=5):
        """Complete analysis - cycles + reconstruction + prediction + confluence"""
        
        # Detect cycles
        cycles = self.detect_cycles(top_n=10)
        
        if 'error' in cycles if isinstance(cycles, dict) else False:
            return cycles
        
        # Reconstruct
        time_arr, original, reconstructed = self.reconstruct_signal(num_cycles)
        
        # Predict future
        future = self.predict_future(bars_ahead, num_cycles)
        
        # Confluence
        confluence = self.get_cycle_confluence()
        
        # Summary stats
        if len(self.cycles) > 0:
            dominant_cycle = self.cycles[0]
            total_strength = sum(c['strength_pct'] for c in self.cycles[:num_cycles])
        else:
            dominant_cycle = None
            total_strength = 0
        
        self.analysis_result = {
            'manual_points': [{
                'datetime': p['datetime'],
                'price': p['price'],
                'type': p['type']
            } for p in self.manual_points],
            'timeframe': self.timeframe,
            'cycles': cycles,
            'time_array': time_arr,
            'original_price': original,
            'reconstructed_price': reconstructed,
            'future': future,
            'confluence': confluence,
            'dominant_cycle': dominant_cycle,
            'total_cycle_strength': round(total_strength, 1),
            'num_points': len(self.manual_points),
            'total_bars': len(original)
        }
        
        return self.analysis_result
