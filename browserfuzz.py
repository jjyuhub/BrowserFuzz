#!/usr/bin/env python3
"""
BrowserFuzz: AI-Guided IPC Fuzzing Framework for Browser Sandbox Testing
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import subprocess
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("browserfuzz.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BrowserFuzz")

@dataclass
class FuzzingState:
    """Represents the current state of the fuzzing process"""
    iterations: int = 0
    crashes_found: int = 0
    unique_crashes: List[str] = field(default_factory=list)
    last_reward: float = 0.0
    total_reward: float = 0.0
    last_input: Optional[Dict[str, Any]] = None
    browser_start_time: float = 0.0
    timeout_count: int = 0
    success_count: int = 0

@dataclass
class FuzzingConfig:
    """Configuration for the fuzzing process"""
    browser: str = "chrome"
    timeout: int = 30
    max_iterations: int = 10000
    reward_crash: float = 10.0
    reward_timeout: float = -1.0
    reward_normal: float = 0.1
    save_dir: str = "findings"
    use_rl: bool = True
    model_path: Optional[str] = None
    mutation_rate: float = 0.3

class BrowserEnvironment:
    """Represents the browser environment being fuzzed"""
    
    def __init__(self, config: FuzzingConfig):
        self.config = config
        self.browser_process = None
        self.state = FuzzingState()
        
        # Create directory for saving findings
        os.makedirs(config.save_dir, exist_ok=True)
    
    def start_browser(self, args: List[str] = None) -> bool:
        """Start the browser process with optional arguments"""
        if args is None:
            args = []
            
        browser_cmd = {
            "chrome": ["google-chrome", "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
            "firefox": ["firefox", "-headless"],
            "edge": ["msedge", "--disable-gpu", "--no-sandbox"],
            "safari": ["safaridriver", "--enable"]
        }.get(self.config.browser.lower(), ["google-chrome"])
        
        try:
            logger.info(f"Starting {self.config.browser} with args: {args}")
            self.state.browser_start_time = time.time()
            
            # In a real tool, you would use a more sophisticated approach to monitor the browser
            self.browser_process = subprocess.Popen(
                browser_cmd + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False
            )
            
            # Brief delay to ensure browser launches
            time.sleep(1)
            
            if self.browser_process.poll() is not None:
                logger.error(f"Browser failed to start, exit code: {self.browser_process.returncode}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            return False
    
    def stop_browser(self) -> Tuple[Optional[str], Optional[str], int]:
        """Stop the browser process and return its output"""
        if not self.browser_process:
            return None, None, 0
            
        # Check if browser is still running
        exit_code = self.browser_process.poll()
        if exit_code is None:
            logger.info("Terminating browser process")
            self.browser_process.terminate()
            try:
                self.browser_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Browser didn't terminate, killing forcefully")
                self.browser_process.kill()
        else:
            logger.info(f"Browser already terminated with exit code {exit_code}")
        
        stdout, stderr = None, None
        try:
            stdout, stderr = self.browser_process.communicate(timeout=5)
            if stdout:
                stdout = stdout.decode('utf-8', errors='ignore')
            if stderr:
                stderr = stderr.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error getting browser output: {e}")
        
        exit_code = self.browser_process.returncode
        self.browser_process = None
        
        return stdout, stderr, exit_code
    
    def execute_ipc_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an IPC message to the browser and observe the result
        
        In a real implementation, this would interact with the actual browser IPC mechanisms.
        This is a simplified placeholder that simulates possible outcomes.
        """
        self.state.last_input = message
        
        # In a real tool, you would need to implement browser-specific IPC mechanisms
        # This simulation just demonstrates the concept
        
        # Simulate possible outcomes for demonstration purposes
        result = {
            "success": True,
            "crash": False,
            "timeout": False,
            "memory_violation": False,
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "execution_time": 0,
        }
        
        # Start browser with the message
        message_args = [f"--ipc-test={json.dumps(message)}"]
        if not self.start_browser(message_args):
            result["success"] = False
            return result
        
        # Wait for result or timeout
        start_time = time.time()
        max_wait = self.config.timeout
        
        while time.time() - start_time < max_wait:
            if self.browser_process.poll() is not None:
                # Browser exited
                break
            time.sleep(0.1)
        
        execution_time = time.time() - start_time
        result["execution_time"] = execution_time
        
        # Check if we hit a timeout
        if self.browser_process.poll() is None:
            result["timeout"] = True
            self.state.timeout_count += 1
            logger.warning(f"Browser timeout after {execution_time:.2f}s")
        
        # Get output and exit code
        stdout, stderr, exit_code = self.stop_browser()
        result["stdout"] = stdout or ""
        result["stderr"] = stderr or ""
        result["exit_code"] = exit_code
        
        # Determine if this was a crash
        if exit_code != 0:
            result["crash"] = True
            self.state.crashes_found += 1
            
            # Check for signs of memory violations
            if stderr and any(x in stderr.lower() for x in ["segmentation fault", "memory violation", 
                                                           "access violation", "heap corruption"]):
                result["memory_violation"] = True
            
            crash_id = f"crash_{self.state.crashes_found}_{int(time.time())}"
            crash_info = {
                "id": crash_id,
                "input": message,
                "exit_code": exit_code,
                "stderr": stderr,
                "stdout": stdout,
                "time": time.time()
            }
            
            # Save crash information
            crash_path = os.path.join(self.config.save_dir, f"{crash_id}.json")
            with open(crash_path, "w") as f:
                json.dump(crash_info, f, indent=2)
            
            self.state.unique_crashes.append(crash_id)
            logger.info(f"Crash found! Saved to {crash_path}")
        else:
            self.state.success_count += 1
            
        return result

class IPCMessageGenerator:
    """Generates IPC messages for fuzzing"""
    
    def __init__(self, config: FuzzingConfig):
        self.config = config
        self.rng = random.Random(int(time.time()))
        
        # Common IPC message types to target
        self.message_types = [
            "RenderProcessHost", "BrowserProcessHost", "FileSystemAccess",
            "WebSocket", "SharedMemory", "ContentSettings",
            "ProcessLauncher", "MediaPermission", "NetworkRequest",
            "CookieAccess", "PluginProcess", "GPUProcess"
        ]
        
        # Common browser IPC methods
        self.ipc_methods = [
            "CreateChannel", "OpenFileSystem", "LaunchProcess",
            "AllocateSharedMemory", "RequestPermission", "SendMessage",
            "ReceiveMessage", "CloseHandle", "ConnectToPipe",
            "UpdatePreferences", "LoadPlugin", "RegisterCallback"
        ]
    
    def generate_random_message(self) -> Dict[str, Any]:
        """Generate a completely random IPC message"""
        message_type = self.rng.choice(self.message_types)
        method = self.rng.choice(self.ipc_methods)
        
        # Generate message with random fields
        message = {
            "type": message_type,
            "method": method,
            "id": self.rng.randint(1, 100000),
            "timestamp": time.time(),
        }
        
        # Add random parameters
        params = {}
        param_count = self.rng.randint(1, 10)
        
        for _ in range(param_count):
            param_type = self.rng.choice(["string", "int", "bool", "array", "object", "binary"])
            param_name = f"param_{self.rng.randint(1, 1000)}"
            
            if param_type == "string":
                # Generate strings of various forms
                string_type = self.rng.choice(["normal", "long", "format", "special", "path"])
                
                if string_type == "normal":
                    value = "".join(chr(self.rng.randint(32, 126)) for _ in range(self.rng.randint(1, 50)))
                elif string_type == "long":
                    value = "A" * self.rng.randint(1000, 10000)
                elif string_type == "format":
                    value = "%s" * self.rng.randint(1, 100)
                elif string_type == "special":
                    value = "".join(chr(self.rng.randint(1, 65535)) for _ in range(self.rng.randint(1, 20)))
                elif string_type == "path":
                    value = "/" + "/".join("dir" + str(i) for i in range(self.rng.randint(1, 10)))
                    if self.rng.random() < 0.3:
                        value = value + "/../" * self.rng.randint(1, 5)
                
                params[param_name] = value
                
            elif param_type == "int":
                int_type = self.rng.choice(["normal", "boundary", "negative", "large"])
                
                if int_type == "normal":
                    value = self.rng.randint(0, 1000)
                elif int_type == "boundary":
                    value = self.rng.choice([0, 1, -1, 255, 256, 65535, 65536, 2**31-1, 2**31])
                elif int_type == "negative":
                    value = -self.rng.randint(1, 10000)
                elif int_type == "large":
                    value = self.rng.randint(2**30, 2**32)
                    
                params[param_name] = value
                
            elif param_type == "bool":
                params[param_name] = self.rng.choice([True, False])
                
            elif param_type == "array":
                array_length = self.rng.randint(0, 50)
                if self.rng.random() < 0.8:  # Mostly uniform arrays
                    array_type = self.rng.choice(["int", "string", "bool"])
                    if array_type == "int":
                        value = [self.rng.randint(-1000, 1000) for _ in range(array_length)]
                    elif array_type == "string":
                        value = ["".join(chr(self.rng.randint(32, 126)) for _ in range(self.rng.randint(1, 20))) 
                                for _ in range(array_length)]
                    elif array_type == "bool":
                        value = [self.rng.choice([True, False]) for _ in range(array_length)]
                else:  # Mixed type arrays (could trigger type confusion)
                    value = []
                    for _ in range(array_length):
                        elem_type = self.rng.choice(["int", "string", "bool", "null", "object"])
                        if elem_type == "int":
                            value.append(self.rng.randint(-1000, 1000))
                        elif elem_type == "string":
                            value.append("".join(chr(self.rng.randint(32, 126)) 
                                               for _ in range(self.rng.randint(1, 20))))
                        elif elem_type == "bool":
                            value.append(self.rng.choice([True, False]))
                        elif elem_type == "null":
                            value.append(None)
                        elif elem_type == "object":
                            value.append({"nested": self.rng.randint(1, 100)})
                            
                params[param_name] = value
                
            elif param_type == "object":
                depth = self.rng.randint(1, 3)
                value = self._generate_nested_object(depth)
                params[param_name] = value
                
            elif param_type == "binary":
                # Generate binary data (base64 encoded for JSON compatibility)
                import base64
                binary_length = self.rng.randint(1, 1000)
                binary_data = bytes(self.rng.randint(0, 255) for _ in range(binary_length))
                value = base64.b64encode(binary_data).decode('ascii')
                params[param_name] = {"type": "binary", "data": value}
        
        message["params"] = params
        return message
    
    def _generate_nested_object(self, depth: int) -> Dict[str, Any]:
        """Generate a nested object with the specified depth"""
        if depth <= 0:
            return {"value": self.rng.randint(1, 100)}
            
        obj = {}
        prop_count = self.rng.randint(1, 5)
        
        for _ in range(prop_count):
            prop_name = f"prop_{self.rng.randint(1, 1000)}"
            
            if self.rng.random() < 0.7 and depth > 1:
                # Nested object
                obj[prop_name] = self._generate_nested_object(depth - 1)
            else:
                # Primitive value
                val_type = self.rng.choice(["int", "string", "bool", "array"])
                
                if val_type == "int":
                    obj[prop_name] = self.rng.randint(-1000, 1000)
                elif val_type == "string":
                    obj[prop_name] = "".join(chr(self.rng.randint(32, 126)) 
                                           for _ in range(self.rng.randint(1, 20)))
                elif val_type == "bool":
                    obj[prop_name] = self.rng.choice([True, False])
                elif val_type == "array":
                    array_length = self.rng.randint(0, 10)
                    obj[prop_name] = [self.rng.randint(1, 100) for _ in range(array_length)]
        
        return obj
    
    def mutate_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an existing message for guided fuzzing"""
        # Copy message to avoid modifying the original
        mutated = json.loads(json.dumps(message))
        
        mutation_type = self.rng.choice([
            "change_type", "change_method", "modify_param", 
            "add_param", "remove_param", "mutate_nested"
        ])
        
        if mutation_type == "change_type":
            mutated["type"] = self.rng.choice(self.message_types)
            
        elif mutation_type == "change_method":
            mutated["method"] = self.rng.choice(self.ipc_methods)
            
        elif mutation_type == "modify_param":
            if "params" in mutated and mutated["params"]:
                param_keys = list(mutated["params"].keys())
                if param_keys:
                    param_to_change = self.rng.choice(param_keys)
                    current_value = mutated["params"][param_to_change]
                    
                    if isinstance(current_value, str):
                        string_mod = self.rng.choice(["append", "prepend", "replace", "inject"])
                        
                        if string_mod == "append":
                            mutated["params"][param_to_change] += "A" * self.rng.randint(1, 1000)
                        elif string_mod == "prepend":
                            mutated["params"][param_to_change] = "A" * self.rng.randint(1, 1000) + current_value
                        elif string_mod == "replace":
                            mutated["params"][param_to_change] = "".join(
                                chr(self.rng.randint(1, 65535)) for _ in range(self.rng.randint(1, 20))
                            )
                        elif string_mod == "inject":
                            inject_pos = self.rng.randint(0, len(current_value))
                            inject_str = "".join(chr(self.rng.randint(1, 255)) for _ in range(self.rng.randint(1, 10)))
                            mutated["params"][param_to_change] = (
                                current_value[:inject_pos] + inject_str + current_value[inject_pos:]
                            )
                            
                    elif isinstance(current_value, int):
                        int_mod = self.rng.choice(["flip_sign", "boundary", "increment", "bitflip"])
                        
                        if int_mod == "flip_sign":
                            mutated["params"][param_to_change] = -current_value
                        elif int_mod == "boundary":
                            mutated["params"][param_to_change] = self.rng.choice([0, 1, -1, 255, 256, 65535, 65536, 2**31-1, 2**31])
                        elif int_mod == "increment":
                            delta = self.rng.randint(-1000, 1000)
                            mutated["params"][param_to_change] += delta
                        elif int_mod == "bitflip":
                            bit_pos = self.rng.randint(0, 31)
                            mutated["params"][param_to_change] ^= (1 << bit_pos)
                            
                    elif isinstance(current_value, bool):
                        mutated["params"][param_to_change] = not current_value
                        
                    elif isinstance(current_value, list):
                        list_mod = self.rng.choice(["add", "remove", "change", "clear", "duplicate"])
                        
                        if list_mod == "add" and current_value:
                            if all(isinstance(x, int) for x in current_value):
                                current_value.append(self.rng.randint(-1000, 1000))
                            elif all(isinstance(x, str) for x in current_value):
                                current_value.append("".join(chr(self.rng.randint(32, 126)) 
                                                           for _ in range(self.rng.randint(1, 20))))
                            else:
                                current_value.append(self.rng.choice([1, "string", True, None]))
                                
                        elif list_mod == "remove" and current_value:
                            idx = self.rng.randint(0, len(current_value) - 1)
                            del current_value[idx]
                            
                        elif list_mod == "change" and current_value:
                            idx = self.rng.randint(0, len(current_value) - 1)
                            if isinstance(current_value[idx], int):
                                current_value[idx] = self.rng.randint(-1000, 1000)
                            elif isinstance(current_value[idx], str):
                                current_value[idx] = "".join(chr(self.rng.randint(32, 126)) 
                                                           for _ in range(self.rng.randint(1, 20)))
                            else:
                                current_value[idx] = self.rng.choice([42, "mutated", False, {"x": 1}])
                                
                        elif list_mod == "clear":
                            current_value.clear()
                            
                        elif list_mod == "duplicate" and current_value:
                            current_value.extend(current_value)
                            
                    elif isinstance(current_value, dict):
                        dict_keys = list(current_value.keys())
                        if dict_keys:
                            dict_mod = self.rng.choice(["add_key", "remove_key", "change_value", "nested_change"])
                            
                            if dict_mod == "add_key":
                                new_key = f"new_key_{self.rng.randint(1, 1000)}"
                                current_value[new_key] = self.rng.choice([42, "value", True, [1, 2, 3]])
                                
                            elif dict_mod == "remove_key":
                                key_to_remove = self.rng.choice(dict_keys)
                                del current_value[key_to_remove]
                                
                            elif dict_mod == "change_value":
                                key_to_change = self.rng.choice(dict_keys)
                                current_value[key_to_change] = self.rng.choice([99, "changed", False, {"x": "y"}])
                                
                            elif dict_mod == "nested_change" and any(isinstance(current_value[k], dict) for k in dict_keys):
                                nested_keys = [k for k in dict_keys if isinstance(current_value[k], dict)]
                                if nested_keys:
                                    nested_key = self.rng.choice(nested_keys)
                                    nested_dict = current_value[nested_key]
                                    if nested_dict:
                                        nested_dict_keys = list(nested_dict.keys())
                                        if nested_dict_keys:
                                            key_to_change = self.rng.choice(nested_dict_keys)
                                            nested_dict[key_to_change] = self.rng.choice([1, "nested", True])
                        
        elif mutation_type == "add_param":
            if "params" not in mutated:
                mutated["params"] = {}
                
            new_param_name = f"fuzzy_param_{self.rng.randint(1, 1000)}"
            new_param_type = self.rng.choice(["string", "int", "bool", "array", "object"])
            
            if new_param_type == "string":
                mutated["params"][new_param_name] = "".join(chr(self.rng.randint(32, 126)) 
                                                          for _ in range(self.rng.randint(1, 100)))
            elif new_param_type == "int":
                mutated["params"][new_param_name] = self.rng.randint(-10000, 10000)
            elif new_param_type == "bool":
                mutated["params"][new_param_name] = self.rng.choice([True, False])
            elif new_param_type == "array":
                array_length = self.rng.randint(0, 20)
                mutated["params"][new_param_name] = [self.rng.randint(1, 100) for _ in range(array_length)]
            elif new_param_type == "object":
                mutated["params"][new_param_name] = {"key": "value", "num": 42}
                
        elif mutation_type == "remove_param":
            if "params" in mutated and mutated["params"]:
                param_keys = list(mutated["params"].keys())
                if param_keys:
                    param_to_remove = self.rng.choice(param_keys)
                    del mutated["params"][param_to_remove]
                    
        elif mutation_type == "mutate_nested":
            if "params" in mutated and mutated["params"]:
                self._mutate_nested(mutated["params"])
        
        return mutated
    
    def _mutate_nested(self, obj: Dict[str, Any], depth: int = 3) -> None:
        """Recursively mutate nested objects"""
        if depth <= 0 or not isinstance(obj, dict):
            return
            
        keys = list(obj.keys())
        if not keys:
            return
            
        # Choose a key to modify
        key = self.rng.choice(keys)
        value = obj[key]
        
        if isinstance(value, dict):
            if self.rng.random() < 0.7:
                # Recurse into nested dict
                self._mutate_nested(value, depth - 1)
            else:
                # Replace with something else
                obj[key] = self.rng.choice([42, "mutated", True, [1, 2, 3]])
                
        elif isinstance(value, list) and value:
            if self.rng.random() < 0.5:
                # Modify a list element
                idx = self.rng.randint(0, len(value) - 1)
                if isinstance(value[idx], dict):
                    self._mutate_nested(value[idx], depth - 1)
                else:
                    value[idx] = self.rng.choice([99, "changed", False, {"nested": True}])
            else:
                # Do something with the list
                list_action = self.rng.choice(["add", "remove", "clear", "duplicate"])
                
                if list_action == "add":
                    value.append(self.rng.choice([1, "new", True, {"x": 1}]))
                elif list_action == "remove" and value:
                    idx = self.rng.randint(0, len(value) - 1)
                    del value[idx]
                elif list_action == "clear":
                    value.clear()
                elif list_action == "duplicate" and value:
                    value.extend(value)
                    
        elif isinstance(value, str):
            # Modify string value
            str_action = self.rng.choice(["append", "prepend", "replace", "format"])
            
            if str_action == "append":
                obj[key] = value + "X" * self.rng.randint(1, 10)
            elif str_action == "prepend":
                obj[key] = "X" * self.rng.randint(1, 10) + value
            elif str_action == "replace":
                obj[key] = "".join(chr(self.rng.randint(32, 126)) for _ in range(self.rng.randint(1, 20)))
            elif str_action == "format":
                obj[key] = value + "%" * self.rng.randint(1, 5)
                
        elif isinstance(value, int):
            # Modify integer value
            int_action = self.rng.choice(["flip", "boundary", "random"])
            
            if int_action == "flip":
                obj[key] = -value
            elif int_action == "boundary":
                obj[key] = self.rng.choice([0, 1, -1, 255, 256, 65535, 65536, 2**31-1, 2**31])
            elif int_action == "random":
                obj[key] = self.rng.randint(-10000, 10000)
                
        elif isinstance(value, bool):
            # Flip boolean
            obj[key] = not value

class ReinforcementLearningAgent:
    """Reinforcement learning agent to guide the fuzzing process"""
    
    def __init__(self, config: FuzzingConfig, input_dim: int = 100, output_dim: int = 10):
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize model
        self.model = self._build_model()
        
        # Experience replay buffer
        self.memory = []
        self.memory_size = 1000
        
        # Exploration parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
    def _build_model(self):
        """Build a neural network model for the RL agent"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.input_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(self.output_dim, activation='linear')
        ])
        
        model.compile(optimizer=keras.optimizers.Adam(0.001),
                     loss='mse')
        
        return model
    
    def vectorize_message(self, message: Dict[str, Any]) -> np.ndarray:
        """
        Convert an IPC message to a vector representation for the model.
        This is a simplified representation and would be more sophisticated in a real implementation.
        """
        vector = np.zeros(self.input_dim)
        
        # Encode message type
        type_id = hash(message.get("type", "")) % 20
        vector[type_id] = 1
        
        # Encode method
        method_id = hash(message.get("method", "")) % 20
        vector[20 + method_id] = 1
        
        # Encode parameters
        params = message.get("params", {})
        param_keys = list(params.keys())
        
        for i, key in enumerate(param_keys[:10]):  # Limit to first 10 parameters
            key_id = hash(key) % 5
            vector[40 + i*5 + key_id] = 1
            
            value = params[key]
            if isinstance(value, str):
                vector[40 + i*5] = min(len(value) / 1000, 1.0)  # Normalize string length
            elif isinstance(value, int):
                vector[40 + i*5 + 1] = min(abs(value) / 10000, 1.0)  # Normalize integer value
            elif isinstance(value, bool):
                vector[40 + i*5 + 2] = 1 if value else 0
            elif isinstance(value, list):
                vector[40 + i*5 + 3] = min(len(value) / 100, 1.0)  # Normalize list length
            elif isinstance(value, dict):
                vector[40 + i*5 + 4] = min(len(value) / 20, 1.0)  # Normalize dict size
        
        return vector
    
    def get_action(self, message: Dict[str, Any]) -> int:
        """Determine next action (mutation type) based on current state"""
        if self.config.use_rl and np.random.rand() > self.epsilon:
            # Use model for prediction
            state = self.vectorize_message(message)
            q_values = self.model.predict(np.array([state]), verbose=0)[0]
            return np.argmax(q_values)
        else:
            # Random exploration
            return np.random.randint(0, self.output_dim)
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def replay(self, batch_size: int = 32):
        """Train the model using experience replay"""
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([m[0] for m in minibatch])
        actions = np.array([m[1] for m in minibatch])
        rewards = np.array([m[2] for m in minibatch])
        next_states = np.array([m[3] for m in minibatch])
        dones = np.array([m[4] for m in minibatch])
        
        # Current Q-values
        targets = self.model.predict(states, verbose=0)
        
        # Future Q-values
        next_q_values = self.model.predict(next_states, verbose=0)
        max_next_q = np.amax(next_q_values, axis=1)
        
        # Update targets for the actions taken
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * max_next_q[i]
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def calculate_reward(self, result: Dict[str, Any]) -> float:
        """Calculate reward based on fuzzing result"""
        reward = 0.0
        
        if result.get("crash", False):
            # Higher reward for crashes
            reward += self.config.reward_crash
            
            # Extra reward for memory violations (potential security issues)
            if result.get("memory_violation", False):
                reward += self.config.reward_crash * 2
                
            # Analyze crash severity based on output
            stderr = result.get("stderr", "")
            if stderr:
                if any(x in stderr.lower() for x in ["memory corruption", "heap overflow", 
                                                    "use after free", "double free"]):
                    # These are high-value security bugs
                    reward += self.config.reward_crash * 3
        
        elif result.get("timeout", False):
            # Timeouts might indicate hangs or infinite loops, but are less valuable
            reward += self.config.reward_timeout
        else:
            # Small reward for successful executions (to encourage exploration)
            reward += self.config.reward_normal
            
        return reward
    
    def save(self, filepath: str):
        """Save the model to a file"""
        self.model.save(filepath)
        
    def load(self, filepath: str):
        """Load the model from a file"""
        self.model = keras.models.load_model(filepath)

class BrowserFuzz:
    """Main fuzzing class that orchestrates the fuzzing process"""
    
    def __init__(self, config: FuzzingConfig):
        self.config = config
        self.env = BrowserEnvironment(config)
        self.generator = IPCMessageGenerator(config)
        
        # Initialize RL agent if enabled
        if config.use_rl:
            self.agent = ReinforcementLearningAgent(config)
            if config.model_path and os.path.exists(config.model_path):
                logger.info(f"Loading existing model from {config.model_path}")
                self.agent.load(config.model_path)
    
    def run(self):
        """Run the fuzzing process"""
        logger.info(f"Starting fuzzing with config: {self.config}")
        
        # Initial random message
        current_message = self.generator.generate_random_message()
        
        for iteration in range(self.config.max_iterations):
            logger.info(f"Iteration {iteration+1}/{self.config.max_iterations}")
            
            # Execute the current message
            result = self.env.execute_ipc_message(current_message)
            
            if self.config.use_rl:
                # Calculate reward
                reward = self.agent.calculate_reward(result)
                self.env.state.last_reward = reward
                self.env.state.total_reward += reward
                
                # Current state vector
                current_state = self.agent.vectorize_message(current_message)
                
                # Determine next action
                action = self.agent.get_action(current_message)
                
                # Create next message based on action
                if action == 0 or iteration == 0:
                    # Generate brand new message
                    next_message = self.generator.generate_random_message()
                else:
                    # Different mutation strategies
                    next_message = self.generator.mutate_message(current_message)
                
                # Next state vector
                next_state = self.agent.vectorize_message(next_message)
                
                # Store experience
                done = (iteration == self.config.max_iterations - 1)
                self.agent.remember(current_state, action, reward, next_state, done)
                
                # Train the model
                if iteration % 10 == 0:
                    self.agent.replay()
                
                # Save model periodically
                if self.config.model_path and iteration % 100 == 0:
                    model_path = self.config.model_path
                    self.agent.save(model_path)
                    logger.info(f"Saved model to {model_path}")
                
            else:
                # Non-RL fuzzing - alternate between random and mutation
                if self.rng.random() < self.config.mutation_rate and iteration > 0:
                    next_message = self.generator.mutate_message(current_message)
                else:
                    next_message = self.generator.generate_random_message()
            
            # Update current message
            current_message = next_message
            
            # Print progress
            if (iteration + 1) % 10 == 0 or result.get("crash", False):
                logger.info(f"Progress: {iteration+1}/{self.config.max_iterations}, "
                           f"Crashes: {self.env.state.crashes_found}, "
                           f"Unique: {len(self.env.state.unique_crashes)}, "
                           f"Timeouts: {self.env.state.timeout_count}, "
                           f"Successful: {self.env.state.success_count}")
                
                if self.config.use_rl:
                    logger.info(f"Epsilon: {self.agent.epsilon:.4f}, "
                               f"Last reward: {self.env.state.last_reward:.2f}, "
                               f"Total reward: {self.env.state.total_reward:.2f}")
            
            # Check for stop conditions
            if os.path.exists("stop_fuzzing"):
                logger.info("Stop file detected, ending fuzzing run")
                break
        
        # Final report
        total_time = time.time() - self.env.state.browser_start_time
        logger.info(f"Fuzzing complete after {iteration+1} iterations")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Total crashes found: {self.env.state.crashes_found}")
        logger.info(f"Unique crashes: {len(self.env.state.unique_crashes)}")
        logger.info(f"Timeouts: {self.env.state.timeout_count}")
        logger.info(f"Successful executions: {self.env.state.success_count}")
        
        if self.env.state.unique_crashes:
            logger.info("Unique crashes:")
            for crash_id in self.env.state.unique_crashes:
                logger.info(f" - {crash_id}")

def main():
    parser = argparse.ArgumentParser(description="BrowserFuzz: AI-Guided IPC Fuzzing Framework")
    parser.add_argument("--browser", type=str, default="chrome", 
                       help="Browser to target (chrome, firefox, edge, safari)")
    parser.add_argument("--timeout", type=int, default=30, 
                       help="Timeout for each test in seconds")
    parser.add_argument("--iterations", type=int, default=10000, 
                       help="Number of fuzzing iterations")
    parser.add_argument("--save-dir", type=str, default="findings", 
                       help="Directory to save findings")
    parser.add_argument("--use-rl", action="store_true", 
                       help="Use reinforcement learning to guide fuzzing")
    parser.add_argument("--model-path", type=str, default=None, 
                       help="Path to save/load RL model")
    parser.add_argument("--mutation-rate", type=float, default=0.3, 
                       help="Rate of mutation vs random generation")
    
    args = parser.parse_args()
    
    config = FuzzingConfig(
        browser=args.browser,
        timeout=args.timeout,
        max_iterations=args.iterations,
        save_dir=args.save_dir,
        use_rl=args.use_rl,
        model_path=args.model_path,
        mutation_rate=args.mutation_rate
    )
    
    fuzzer = BrowserFuzz(config)
    fuzzer.run()

if __name__ == "__main__":
    main()
