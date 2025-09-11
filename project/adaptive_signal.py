import time
import threading
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

@dataclass
class SignalState:
    """Represents the state of a traffic signal"""
    current_signal: str  # 'red', 'yellow', 'green'
    timer: int          # Time remaining in seconds
    direction: str      # 'north_south' or 'east_west'
    last_change: datetime = None

class AdaptiveSignalController:
    """Adaptive traffic signal controller using AI algorithms"""
    
    def __init__(self, config_file: str = 'config.json'):
        """
        Initialize the adaptive signal controller
        
        Args:
            config_file (str): Path to configuration file
        """
        self.config = self.load_config(config_file)
        
        # Initialize signal states
        self.signals = {
            'north_south': SignalState('green', 30, 'north_south', datetime.now()),
            'east_west': SignalState('red', 30, 'east_west', datetime.now())
        }
        
        # Timing constraints
        self.min_green_time = self.config.get('min_green_time', 15)
        self.max_green_time = self.config.get('max_green_time', 60)
        self.yellow_time = self.config.get('yellow_time', 3)
        self.red_clearance_time = self.config.get('red_clearance_time', 2)
        
        # Control flags
        self.emergency_override = False
        self.manual_mode = False
        self.running = False
        
        # Performance tracking
        self.cycle_history = []
        self.efficiency_metrics = {
            'total_cycles': 0,
            'avg_wait_time': 0.0,
            'throughput': 0.0,
            'fuel_savings': 0.0
        }
        
        # Threading
        self.control_thread = None
        self.stop_event = threading.Event()
    
    def load_config(self, config_file: str) -> Dict:
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            return {
                'min_green_time': 15,
                'max_green_time': 60,
                'yellow_time': 3,
                'red_clearance_time': 2,
                'adaptive_algorithm': 'vehicle_density',
                'emergency_priority': True
            }
    
    def calculate_adaptive_timing(self, vehicle_counts: Dict[str, int]) -> Dict[str, int]:
        """
        Calculate optimal signal timing based on vehicle counts
        
        Args:
            vehicle_counts (Dict[str, int]): Current vehicle counts by direction
            
        Returns:
            Dict[str, int]: Optimal green times for each direction pair
        """
        total_vehicles = sum(vehicle_counts.values())
        
        if total_vehicles == 0:
            # No vehicles detected, use minimum timing
            return {
                'north_south': self.min_green_time,
                'east_west': self.min_green_time
            }
        
        # Calculate vehicle densities
        ns_vehicles = vehicle_counts.get('north', 0) + vehicle_counts.get('south', 0)
        ew_vehicles = vehicle_counts.get('east', 0) + vehicle_counts.get('west', 0)
        
        # Calculate ratios
        ns_ratio = ns_vehicles / total_vehicles if total_vehicles > 0 else 0.5
        ew_ratio = ew_vehicles / total_vehicles if total_vehicles > 0 else 0.5
        
        # Base cycle time (total time for both directions)
        base_cycle_time = 90  # seconds
        
        # Calculate green times based on ratios
        ns_green = max(self.min_green_time, 
                      min(self.max_green_time, 
                          int(base_cycle_time * ns_ratio)))
        
        ew_green = max(self.min_green_time,
                      min(self.max_green_time,
                          int(base_cycle_time * ew_ratio)))
        
        # Apply traffic flow optimization
        ns_green, ew_green = self.optimize_flow(ns_green, ew_green, vehicle_counts)
        
        return {
            'north_south': ns_green,
            'east_west': ew_green
        }
    
    def optimize_flow(self, ns_time: int, ew_time: int, 
                     vehicle_counts: Dict[str, int]) -> tuple:
        """
        Apply advanced optimization to signal timing
        
        Args:
            ns_time (int): North-South green time
            ew_time (int): East-West green time
            vehicle_counts (Dict[str, int]): Current vehicle counts
            
        Returns:
            tuple: Optimized (ns_time, ew_time)
        """
        # Historical performance weighting
        if len(self.cycle_history) > 5:
            recent_performance = self.cycle_history[-5:]
            avg_efficiency = np.mean([cycle['efficiency'] for cycle in recent_performance])
            
            # Adjust based on recent performance
            if avg_efficiency < 0.8:  # Poor performance
                # Increase time for direction with more vehicles
                total_ns = vehicle_counts.get('north', 0) + vehicle_counts.get('south', 0)
                total_ew = vehicle_counts.get('east', 0) + vehicle_counts.get('west', 0)
                
                if total_ns > total_ew:
                    ns_time = min(self.max_green_time, ns_time + 10)
                else:
                    ew_time = min(self.max_green_time, ew_time + 10)
        
        # Peak hour adjustments
        current_hour = datetime.now().hour
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:
            # Rush hour - extend green times
            ns_time = min(self.max_green_time, ns_time + 5)
            ew_time = min(self.max_green_time, ew_time + 5)
        
        return ns_time, ew_time
    
    def update_signals(self, vehicle_counts: Dict[str, int]) -> None:
        """
        Update signal timings based on current traffic conditions
        
        Args:
            vehicle_counts (Dict[str, int]): Current vehicle counts
        """
        if self.emergency_override or self.manual_mode:
            return
        
        # Calculate optimal timings
        optimal_timings = self.calculate_adaptive_timing(vehicle_counts)
        
        # Apply new timings to currently green signal
        for direction, timing in optimal_timings.items():
            signal = self.signals[direction]
            if signal.current_signal == 'green':
                # Only update if significantly different and minimum time has passed
                time_since_change = datetime.now() - signal.last_change
                if (time_since_change.total_seconds() >= self.min_green_time and
                    abs(signal.timer - timing) > 5):
                    signal.timer = timing
    
    def emergency_vehicle_detected(self, direction: str) -> None:
        """
        Handle emergency vehicle priority
        
        Args:
            direction (str): Direction of emergency vehicle ('north_south' or 'east_west')
        """
        print(f"🚨 EMERGENCY VEHICLE DETECTED: {direction}")
        
        self.emergency_override = True
        current_time = datetime.now()
        
        # Immediately switch to green for emergency direction
        for dir_key, signal in self.signals.items():
            if dir_key == direction:
                if signal.current_signal != 'green':
                    signal.current_signal = 'green'
                    signal.last_change = current_time
                signal.timer = 45  # Extended time for emergency vehicle
            else:
                signal.current_signal = 'red'
                signal.timer = 45
                signal.last_change = current_time
        
        # Log emergency event
        self.cycle_history.append({
            'timestamp': current_time,
            'type': 'emergency',
            'direction': direction,
            'duration': 45
        })
    
    def clear_emergency_override(self) -> None:
        """Clear emergency override and return to normal operation"""
        print("✅ Emergency override cleared - returning to normal operation")
        self.emergency_override = False
        
        # Resume normal timing
        current_time = datetime.now()
        for signal in self.signals.values():
            signal.last_change = current_time
    
    def set_manual_mode(self, enabled: bool, 
                       ns_time: int = None, ew_time: int = None) -> None:
        """
        Enable or disable manual mode
        
        Args:
            enabled (bool): Whether to enable manual mode
            ns_time (int): North-South green time (if manual mode)
            ew_time (int): East-West green time (if manual mode)
        """
        self.manual_mode = enabled
        
        if enabled and ns_time and ew_time:
            # Apply manual timings
            current_time = datetime.now()
            
            for direction, timing in [('north_south', ns_time), ('east_west', ew_time)]:
                signal = self.signals[direction]
                if signal.current_signal == 'green':
                    signal.timer = timing
                    signal.last_change = current_time
    
    def run_signal_cycle(self) -> None:
        """Main signal control loop"""
        print("🚦 Starting adaptive signal control...")
        self.running = True
        cycle_start_time = datetime.now()
        
        while self.running and not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Update timers
                for direction, signal in self.signals.items():
                    if signal.timer > 0:
                        signal.timer -= 1
                    
                    # Handle signal transitions
                    if signal.timer <= 0:
                        self.transition_signal(direction, signal)
                
                # Calculate cycle efficiency
                if (current_time - cycle_start_time).total_seconds() >= 90:
                    self.calculate_cycle_efficiency()
                    cycle_start_time = current_time
                
                # Wait for next second
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in signal control loop: {e}")
                time.sleep(1)
    
    def transition_signal(self, direction: str, signal: SignalState) -> None:
        """Handle signal state transitions"""
        current_time = datetime.now()
        
        if signal.current_signal == 'green':
            # Green -> Yellow
            signal.current_signal = 'yellow'
            signal.timer = self.yellow_time
            signal.last_change = current_time
            
        elif signal.current_signal == 'yellow':
            # Yellow -> Red
            signal.current_signal = 'red'
            signal.last_change = current_time
            
            # Switch other direction to green
            other_direction = 'east_west' if direction == 'north_south' else 'north_south'
            other_signal = self.signals[other_direction]
            
            # Red clearance time before switching
            time.sleep(self.red_clearance_time)
            
            other_signal.current_signal = 'green'
            other_signal.timer = self.min_green_time  # Will be updated by adaptive algorithm
            other_signal.last_change = current_time
    
    def calculate_cycle_efficiency(self) -> None:
        """Calculate and store cycle efficiency metrics"""
        current_time = datetime.now()
        
        # Mock efficiency calculation (in real implementation, use actual traffic data)
        efficiency = np.random.uniform(0.7, 0.95)  # 70-95% efficiency
        
        cycle_data = {
            'timestamp': current_time,
            'type': 'normal',
            'efficiency': efficiency,
            'ns_green_time': self.signals['north_south'].timer if self.signals['north_south'].current_signal == 'green' else 0,
            'ew_green_time': self.signals['east_west'].timer if self.signals['east_west'].current_signal == 'green' else 0
        }
        
        self.cycle_history.append(cycle_data)
        
        # Keep only last 100 cycles
        if len(self.cycle_history) > 100:
            self.cycle_history = self.cycle_history[-100:]
        
        # Update performance metrics
        self.efficiency_metrics['total_cycles'] += 1
        recent_cycles = self.cycle_history[-10:] if len(self.cycle_history) >= 10 else self.cycle_history
        self.efficiency_metrics['avg_efficiency'] = np.mean([c['efficiency'] for c in recent_cycles])
    
    def start_control_system(self) -> None:
        """Start the signal control system in a separate thread"""
        if not self.running:
            self.stop_event.clear()
            self.control_thread = threading.Thread(target=self.run_signal_cycle)
            self.control_thread.daemon = True
            self.control_thread.start()
            print("✅ Adaptive signal control system started")
    
    def stop_control_system(self) -> None:
        """Stop the signal control system"""
        print("🛑 Stopping adaptive signal control system...")
        self.running = False
        self.stop_event.set()
        
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=5)
        
        print("✅ Signal control system stopped")
    
    def get_signal_status(self) -> Dict[str, Dict]:
        """Get current signal status for all directions"""
        return {
            direction: {
                'signal': signal.current_signal,
                'timer': signal.timer,
                'direction': signal.direction,
                'last_change': signal.last_change.isoformat() if signal.last_change else None
            } for direction, signal in self.signals.items()
        }
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return {
            'efficiency_metrics': self.efficiency_metrics.copy(),
            'cycle_count': len(self.cycle_history),
            'emergency_override': self.emergency_override,
            'manual_mode': self.manual_mode,
            'system_running': self.running
        }

def test_signal_controller():
    """Test function for AdaptiveSignalController"""
    controller = AdaptiveSignalController()
    
    print("Testing Adaptive Signal Controller...")
    print("Starting control system...")
    
    controller.start_control_system()
    
    try:
        # Simulate traffic for 30 seconds
        for i in range(30):
            # Simulate varying traffic
            vehicle_counts = {
                'north': np.random.randint(5, 15),
                'south': np.random.randint(5, 15),
                'east': np.random.randint(5, 15),
                'west': np.random.randint(5, 15)
            }
            
            print(f"Time: {i+1}s, Counts: {vehicle_counts}")
            print(f"Signals: {controller.get_signal_status()}")
            
            # Update signals based on traffic
            controller.update_signals(vehicle_counts)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        controller.stop_control_system()

if __name__ == "__main__":
    test_signal_controller()
