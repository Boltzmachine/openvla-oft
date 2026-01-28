import time
import torch
from contextlib import ContextDecorator
from transformers.cache_utils import Cache, DynamicCache

all_timers = {}

class BaseTimer:
    def __init__(self, name, cuda_only=False):
        self.name = name
        self.start_time = None
        
        self.total = 0.0
        self.count = 0
        self.cuda_only = cuda_only
        
    def start(self):
        if self.cuda_only:
            self.start_time = torch.cuda.Event(enable_timing=True)
            self.end_time = torch.cuda.Event(enable_timing=True)
            self.start_time.record()
        else:
            self.start_time = time.time()
    
    def stop(self):
        if self.cuda_only:
            self.end_time.record()
            torch.cuda.synchronize()
            elapsed_time = self.start_time.elapsed_time(self.end_time)
        else:
            elapsed_time = (time.time() - self.start_time) * 1000
        self.total += elapsed_time
        self.count += 1
        self.start_time = None
        
    # def pause(self):
    #     assert self.start_time is not None, "Timer is not running"
    #     elapsed_time = time.time() - self.start_time
    #     self.total += elapsed_time
    #     self.start_time = None

    # def resume(self):
    #     assert self.start_time is None, "Timer is already running"
    #     self.start_time = time.time()
        
    def print(self):
        print(f"{self.name}: step {self.count} | ", end='')
        if self.cuda_only:
            print(f"Avg CUDA latency {self.total/self.count:.10f}")
        else:
            print(f"Avg latency {self.total/self.count:.10f}")
        
        
class TimerCollection:
    def __init__(self, timers):
        self.timers = timers
    
    def start(self):
        for timer in self.timers:
            timer.start()
            
    def stop(self):
        for timer in self.timers:
            timer.stop()
        
    def print(self):
        for timer in self.timers:
            timer.print()

class Timers(ContextDecorator):
    def __init__(self, name, modes=['cpu', 'cuda']):
        self.name = name
        if name is None:
            return
        if name.startswith("-"):
            assert name[1:] in all_timers
            self.timer = all_timers[name[1:]]
        else:
            if name not in all_timers:
                timers = []
                for mode in modes:
                    if mode == 'cpu':
                        kwargs = {'cuda_only': False}
                    elif mode == 'cuda':
                        kwargs = {'cuda_only': True}
                    else:
                        raise ValueError(f"Unknown mode: {mode}")
                    timers.append(BaseTimer(name, **kwargs))
                all_timers[name] = TimerCollection(timers)
            self.timer = all_timers[name]
    
    def __enter__(self):
        if self.name is None:
            return
        if self.name.startswith("-"):
            self.timer.pause()
        else:
            self.timer.start()
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.name is None:
            return
        if self.name.startswith("-"):
            self.timer.resume()
        else:
            self.timer.stop()
            self.timer.print()
        

def repeat_batch(tensor, batch_size):
    if tensor is None:
        return None
    if tensor.dim() == 1:
        return tensor.expand(batch_size).clone()
    elif tensor.dim() == 2:
        return tensor.expand(batch_size, -1).clone()
    elif tensor.dim() == 3:
        return tensor.expand(batch_size, -1, -1).clone()
    elif tensor.dim() == 4:
        return tensor.expand(batch_size, -1, -1, -1).clone()
    elif tensor.dim() == 5:
        return tensor.expand(batch_size, -1, -1, -1, -1).clone()
    else:
        raise
    
def repeat_input(input_ids, kwargs, batch_size = 16):
    input_ids = repeat_batch(input_ids, batch_size)
    if "attention_mask" in kwargs:
        kwargs["attention_mask"] = repeat_batch(kwargs["attention_mask"], batch_size)
    if "pixel_values" in kwargs:
        kwargs["pixel_values"] = repeat_batch(kwargs["pixel_values"], batch_size)
    if kwargs.get("past_key_values", None) is not None:
        cache_cls = tuple
        if isinstance(kwargs["past_key_values"], Cache):
            cache_cls = type(kwargs["past_key_values"])
            kwargs["past_key_values"] = kwargs["past_key_values"].to_legacy_cache()

        kwargs['past_key_values'] = list(kwargs["past_key_values"])
        for i, layer in enumerate(kwargs["past_key_values"]):
            keys, values = layer
            kwargs["past_key_values"][i] = (
                repeat_batch(keys, batch_size),
                repeat_batch(values, batch_size)
            )
        kwargs["past_key_values"] = tuple(kwargs["past_key_values"])
        
        if cache_cls is not tuple:
            kwargs["past_key_values"] = cache_cls.from_legacy_cache(kwargs["past_key_values"])
    if "inputs_embeds" in kwargs:
        kwargs["inputs_embeds"] = repeat_batch(kwargs["inputs_embeds"], batch_size)
    return input_ids, kwargs

def squeeze_output(output):
    for key, value in output.items():
        if key == "sequences":
            setattr(output, key, value[0:1])
        elif key == "attentions":
            setattr(output, key, tuple(
                tuple(attn[0:1] if attn.dim() > 1 else attn for attn in attention) for attention in value
            ))
        elif key == "past_key_values":
            if isinstance(value, DynamicCache):
                value.key_cache = list(
                    cache[0:1] for cache in value.key_cache
                )
                value.value_cache = list(
                    cache[0:1] for cache in value.value_cache
                )
            elif isinstance(value, tuple):
                setattr(output, key, tuple(
                    (key_cache[0:1], value_cache[0:1]) for key_cache, value_cache in value
                ))
            else:
                raise NotImplementedError(f"Unknown past_key_values type: {type(value)}")
        else:
            raise NotImplementedError(f"Unknown output key: {key}")
    return output
                
            
def break_back(time_gaps: list):
    res = []
    for i in time_gaps:
        res.append(1)
        for _ in range(i):
            res.append(0)
    return res


def calculate_flops(time_gaps_0, time_gaps_1):
    total = 0
    reduced = 0
    assert len(time_gaps_0) == len(time_gaps_1)
    for time_gap_0, time_gap_1 in zip(time_gaps_0, time_gaps_1):
        if time_gap_0 == 1:
            assert time_gap_1 == 1
        total += 405.50 + 57.96
        reduced += 405.50 + 57.96

        if time_gap_0 == 1 and time_gap_1 == 1:
            total += 3726.64
            reduced += 3726.64
        elif time_gap_0 == 0 and time_gap_1 == 1:
            total += 3726.64
            reduced += 1969.04
        elif time_gap_0 == 0 and time_gap_1 == 0:
            total += 3726.64
            reduced += 555.03


    return reduced, total