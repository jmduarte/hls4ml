from hls4ml.model.graph import HLSConfig, ModelGraph

try:
    from hls4ml.model import profiling

    __profiling_enabled__ = True
except ImportError:
    __profiling_enabled__ = False
