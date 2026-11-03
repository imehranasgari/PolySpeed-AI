import io
import sys
import time
from .compiler import NativeCompiler
from .generator import CodeTranspiler
from .profiler import profile_python_code

def run_python_isolated(code: str) -> tuple[bool, str, float]:
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    start = time.perf_counter()
    success = True
    try:
        exec(code, {"__builtins__": __builtins__})
        output = buffer.getvalue()
    except Exception as e:
        success = False
        output = f"Python Execution Error: {str(e)}"
    finally:
        sys.stdout = old_stdout
        elapsed = time.perf_counter() - start
    return success, output, elapsed

class OptimizationAgent:
    def __init__(self, transpiler: CodeTranspiler, compiler: NativeCompiler, max_retries: int = 3):
        self.transpiler = transpiler
        self.compiler = compiler
        self.max_retries = max_retries

    def optimize_and_verify(self, python_code: str, target_lang: str, selected_model_name: str):
        profile_res = profile_python_code(python_code)
        if not profile_res["success"]:
            return {"error": profile_res["error"], "stage": "python"}

        py_ok, py_out, py_time = run_python_isolated(python_code)
        if not py_ok:
            return {"error": py_out, "stage": "python"}

        err_feedback = None
        code = ""
        build_ok = False
        build_msg = ""

        for attempt in range(self.max_retries + 1):
            try:
                code = self.transpiler.generate(
                    python_code=python_code,
                    language=target_lang,
                    selected_model_name=selected_model_name,
                    error_feedback=err_feedback,
                    profile_stats=profile_res["hotspots"]
                )
            except Exception as e:
                return {"error": f"LLM Dispatch Error: {str(e)}", "stage": "llm"}

            self.compiler.write_source(code, target_lang)
            build_ok, build_msg = self.compiler.compile(target_lang)
            if build_ok:
                break
            err_feedback = build_msg

        if not build_ok:
            return {
                "error": f"Build failed after {self.max_retries} self-healing cycles:\n{build_msg}",
                "code": code,
                "stage": "compile"
            }

        exec_ok, native_out, native_time = self.compiler.execute()
        if not exec_ok:
            return {"error": native_out, "code": code, "stage": "execute"}

        speedup = py_time / native_time if native_time > 0 else 1.0

        return {
            "success": True,
            "code": code,
            "py_output": py_out,
            "py_time": py_time,
            "native_output": native_out,
            "native_time": native_time,
            "speedup": speedup,
            "hotspots": profile_res["hotspots"]
        }
