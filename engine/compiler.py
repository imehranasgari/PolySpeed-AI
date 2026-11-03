import subprocess
import os
import time

class NativeCompiler:
    def __init__(self, work_dir="workspace"):
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)

    def write_source(self, code: str, language: str) -> str:
        ext = "rs" if language.lower() == "rust" else "cpp"
        file_path = os.path.join(self.work_dir, f"main.{ext}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        return file_path

    def compile(self, language: str) -> tuple[bool, str]:
        ext = "rs" if language.lower() == "rust" else "cpp"
        src_path = os.path.join(self.work_dir, f"main.{ext}")
        exe_path = os.path.join(self.work_dir, "main.exe" if os.name == "nt" else "main")

        if language.lower() == "rust":
            cmd = [
                "rustc", src_path,
                "-C", "opt-level=3",
                "-C", "target-cpu=native",
                "-C", "codegen-units=1",
                "-C", "lto=fat",
                "-C", "panic=abort",
                "-o", exe_path
            ]
        else: # C++17 with full SIMD, OpenMP & Math Linkers
            cmd = [
                "g++", "-std=c++17", "-O3", "-march=native",
                "-fopenmp",
                "-DNDEBUG",
                src_path,
                "-static-libgcc", "-static-libstdc++",
                "-lm",
                "-o", exe_path
            ]

        try:
            res = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=90)
            return True, "Build successful."
        except subprocess.CalledProcessError as e:
            return False, e.stderr
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out after 90 seconds."
        except Exception as e:
            return False, str(e)

    def execute(self, timeout=60) -> tuple[bool, str, float]:
        exe_path = os.path.join(self.work_dir, "main.exe" if os.name == "nt" else "main")
        if not os.path.exists(exe_path):
            return False, "Executable not found.", 0.0

        try:
            start = time.perf_counter()
            res = subprocess.run([exe_path], check=True, text=True, capture_output=True, timeout=timeout)
            elapsed = time.perf_counter() - start
            return True, res.stdout, elapsed
        except subprocess.CalledProcessError as e:
            return False, e.stderr, 0.0
        except subprocess.TimeoutExpired:
            return False, f"Execution timed out after {timeout} seconds.", 0.0
        except Exception as e:
            return False, str(e), 0.0
