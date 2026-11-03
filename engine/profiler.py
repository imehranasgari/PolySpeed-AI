import cProfile
import pstats
import io
import time

def profile_python_code(code_str: str) -> dict:
    profiler = cProfile.Profile()
    globals_dict = {"__builtins__": __builtins__}
    
    start_wall = time.perf_counter()
    success = True
    err_msg = ""
    
    try:
        profiler.enable()
        exec(code_str, globals_dict)
        profiler.disable()
    except Exception as e:
        success = False
        err_msg = str(e)
        profiler.disable()
    
    wall_duration = time.perf_counter() - start_wall
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    hotspots = s.getvalue()
    
    return {
        "success": success,
        "wall_time": wall_duration,
        "hotspots": hotspots,
        "error": err_msg
    }
