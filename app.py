import gradio as gr
from dotenv import load_dotenv
from system_info import retrieve_system_info
from engine.compiler import NativeCompiler
from engine.generator import CodeTranspiler, AVAILABLE_MODELS
from engine.runner import OptimizationAgent

load_dotenv(override=True)
sys_info = retrieve_system_info()
compiler = NativeCompiler()
transpiler = CodeTranspiler(system_info=sys_info)
agent = OptimizationAgent(transpiler=transpiler, compiler=compiler)

PRESETS = {
    "Monte Carlo Pi (Iterative Simulation)": """# Monte-Carlo Pi Estimation (High Iteration Count)
import random
import time

def estimate_pi(samples=15_000_000):
    inside = 0
    for _ in range(samples):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            inside += 1
    return (inside / samples) * 4.0

start = time.perf_counter()
pi = estimate_pi()
duration = time.perf_counter() - start

print(f"Estimated Pi: {pi:.8f}")
print(f"Execution Time: {duration:.4f} seconds")
""",
    "Max Subarray LCG (O(N^2) Nested Loops)": """# Maximum Subarray Sum with Linear Congruential Generator
import time

def lcg(seed, a=1664525, c=1013904223, m=2**32):
    val = seed
    while True:
        val = (a * val + c) % m
        yield val

def compute_max_subarray(n=8000, seed=42):
    gen = lcg(seed)
    nums = [next(gen) % 21 - 10 for _ in range(n)]
    max_sum = float('-inf')
    for i in range(n):
        curr = 0
        for j in range(i, n):
            curr += nums[j]
            if curr > max_sum:
                max_sum = curr
    return max_sum

start = time.perf_counter()
result = compute_max_subarray()
duration = time.perf_counter() - start

print(f"Max Subarray Sum: {result}")
print(f"Execution Time: {duration:.4f} seconds")
""",
    "Leibniz Formula (SIMD-Friendly Series)": """# Leibniz Series for Pi (200 Million Iterations)
import time

def calculate_pi(iterations=100_000_000):
    result = 1.0
    for i in range(1, iterations + 1):
        j1 = i * 4 - 1
        j2 = i * 4 + 1
        result -= (1.0 / j1)
        result += (1.0 / j2)
    return result * 4.0

start = time.perf_counter()
pi = calculate_pi()
duration = time.perf_counter() - start

print(f"Result: {pi:.12f}")
print(f"Execution Time: {duration:.4f} seconds")
"""
}

model_choices = list(AVAILABLE_MODELS.keys())

CUSTOM_CSS = """
:root {
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --bg-dark: #070a13;
    --card-bg: rgba(20, 26, 38, 0.8);
    --border-color: rgba(255, 255, 255, 0.08);
}

body, .gradio-container {
    background-color: var(--bg-dark) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
}

.hero-box {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(168, 85, 247, 0.12));
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 22px;
    backdrop-filter: blur(12px);
}

.hero-box h1 {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.badge-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 12px;
}

.badge-item {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid var(--border-color);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    color: #cbd5e1;
}

.badge-item strong {
    color: #38bdf8;
}

.action-btn {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border-radius: 12px !important;
    border: none !important;
    padding: 14px !important;
    box-shadow: 0 4px 18px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.25s ease !important;
}

.action-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.6) !important;
}

.speed-metric input {
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    text-align: center !important;
    color: #10b981 !important;
    background: rgba(16, 185, 129, 0.12) !important;
    border: 1px solid rgba(16, 185, 129, 0.4) !important;
    border-radius: 14px !important;
}
"""

def update_preset(preset_name):
    return PRESETS.get(preset_name, "")

def run_synthesis(python_code, target_lang, model_name):
    res = agent.optimize_and_verify(python_code, target_lang, model_name)
    if "error" in res:
        return res.get("code", ""), f"Stage [{res['stage'].upper()}] Failure:\n{res['error']}", "", "0.0x", ""
    
    speedup_label = f"{res['speedup']:.1f}x"
    metrics_summary = (
        f"⏱️ Python Baseline Time: {res['py_time']:.4f}s\n"
        f"🚀 Native {target_lang} Time: {res['native_time']:.4f}s\n"
        f"⚡ Acceleration Speedup: {res['speedup']:.1f}x Faster\n"
        f"🎯 Hardware Tuning: OpenMP Threading + Target Native Vectorization"
    )
    return res["code"], res["py_output"], f"{res['native_output']}\n\n=============================\n{metrics_summary}", speedup_label, res["hotspots"]

with gr.Blocks(title="PolySpeed-AI Enterprise Studio", css=CUSTOM_CSS, theme=gr.themes.Default()) as demo:
    with gr.Column(elem_classes=["hero-box"]):
        gr.Markdown(
            f"""
            # ⚡ PolySpeed-AI: Hardware-Aware Native Acceleration Engine
            **Autonomous LLM Transpiler for CPU-Bound Algorithms & High-Performance Kernels**
            
            <div class="badge-strip">
                <span class="badge-item">💻 Host: <strong>{sys_info['cpu']['brand']}</strong></span>
                <span class="badge-item">⚙️ Threads: <strong>{sys_info['cpu']['cores_logical']} Logical ({sys_info['cpu']['cores_physical']} Physical)</strong></span>
                <span class="badge-item">⚡ G++ OpenMP: <strong>{'Active' if sys_info['compilers']['g++'] else 'Not Found'}</strong></span>
                <span class="badge-item">🦀 Rust Toolchain: <strong>{'Ready' if sys_info['compilers']['rustc'] else 'Not Found'}</strong></span>
            </div>
            """
        )

    with gr.Row():
        preset_selector = gr.Dropdown(
            choices=list(PRESETS.keys()),
            value=list(PRESETS.keys())[0],
            label="📚 Compute-Heavy Benchmark Presets (Load Ready-to-Test Algorithms)",
            scale=8
        )

    with gr.Row():
        with gr.Column(scale=6):
            py_input = gr.Code(label="🐍 Python Source Code (Baseline)", value=PRESETS[list(PRESETS.keys())[0]], language="python", lines=18)
        with gr.Column(scale=6):
            native_output = gr.Code(label="⚡ Native Transpiled Code (Rust / C++)", language="cpp", lines=18)

    with gr.Row(equal_height=True):
        model_dropdown = gr.Dropdown(
            choices=model_choices,
            value="Gemma 4 26B A4B QAT (Local LM Studio)",
            label="🤖 LLM Synthesis Engine",
            scale=5
        )
        lang_choice = gr.Dropdown(
            choices=["Rust", "C++"],
            value="Rust",
            label="🎯 Target Native Language",
            scale=2
        )
        speedup_metric = gr.Textbox(label="🚀 Acceleration Factor", value="--", elem_classes=["speed-metric"], scale=2)

    with gr.Row():
        run_btn = gr.Button("🚀 Transpile, Auto-Heal & Run Hardware Benchmark", elem_classes=["action-btn"])

    with gr.Row():
        with gr.Column(scale=6):
            py_console = gr.TextArea(label="🐍 Python Console Output", lines=7)
        with gr.Column(scale=6):
            native_console = gr.TextArea(label="⚡ Native Benchmark & Telemetry Breakdown", lines=7)

    with gr.Accordion("🔍 Python Runtime Profiler Hotspots (cProfile)", open=False):
        profiler_console = gr.TextArea(label="Hotspot Function Analysis", lines=6)

    preset_selector.change(fn=update_preset, inputs=[preset_selector], outputs=[py_input])

    run_btn.click(
        fn=run_synthesis,
        inputs=[py_input, lang_choice, model_dropdown],
        outputs=[native_output, py_console, native_console, speedup_metric, profiler_console]
    )

if __name__ == "__main__":
    demo.launch()
