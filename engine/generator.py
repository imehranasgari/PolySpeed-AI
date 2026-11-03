import os
from openai import OpenAI

AVAILABLE_MODELS = {
    "Gemma 4 26B A4B QAT (Local LM Studio)": {
        "provider": "lmstudio",
        "model_id": "google/gemma-4-26b-a4b-qat"
    },
    "Gemini 2.5 Flash (Google Cloud)": {
        "provider": "gemini",
        "model_id": "gemini-2.5-flash"
    },
    "Gemini 3.1 Flash Lite (Google Cloud)": {
        "provider": "gemini",
        "model_id": "gemini-3.1-flash-lite"
    },
    "Llama 3.3 70B Versatile (Groq Cloud)": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile"
    },
    "Llama 3.1 8B Instant (Groq Fast)": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant"
    },
    "Qwen 2.5 Coder 32B (OpenRouter)": {
        "provider": "openrouter",
        "model_id": "qwen/qwen-2.5-coder-32b-instruct"
    },
    "DeepSeek R1 (OpenRouter / Reasoning)": {
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-r1"
    },
    "Qwen 2.5 Coder (Local Ollama)": {
        "provider": "ollama",
        "model_id": "qwen2.5-coder"
    },
    "DeepSeek Coder V2 (Local Ollama)": {
        "provider": "ollama",
        "model_id": "deepseek-coder-v2"
    }
}

class CodeTranspiler:
    def __init__(self, system_info: dict):
        self.system_info = system_info
        self.clients = {}

        # 1. LM Studio Local
        lmstudio_url = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
        self.clients["lmstudio"] = OpenAI(
            api_key="lm-studio",
            base_url=lmstudio_url,
            timeout=300.0
        )

        # 2. Google Gemini
        if os.getenv("GOOGLE_API_KEY"):
            self.clients["gemini"] = OpenAI(
                api_key=os.getenv("GOOGLE_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )

        # 3. Groq Cloud
        if os.getenv("GROQ_API_KEY"):
            self.clients["groq"] = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )

        # 4. OpenRouter
        if os.getenv("OPENROUTER_API_KEY"):
            self.clients["openrouter"] = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1"
            )

        # 5. Ollama Local
        self.clients["ollama"] = OpenAI(
            api_key="ollama",
            base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1")
        )

    def generate(self, python_code: str, language: str = "Rust", selected_model_name: str = None, error_feedback: str = None, profile_stats: str = None) -> str:
        model_meta = AVAILABLE_MODELS.get(selected_model_name)
        if not model_meta:
            model_meta = AVAILABLE_MODELS["Gemma 4 26B A4B QAT (Local LM Studio)"]

        provider = model_meta["provider"]
        model_id = model_meta["model_id"]
        client = self.clients.get(provider)

        if not client:
            raise ValueError(f"Provider '{provider}' is not reachable. Verify .env credentials or local server status.")

        system_prompt = (
            f"You are a World-Class Principal Systems Engineer and HPC Specialist.\n"
            f"Task: Transpile the compute-heavy Python code into production-grade, standalone {language} code.\n"
            f"Target System Architecture:\n{self.system_info}\n\n"
            f"CRITICAL COMPILATION CONSTRAINTS (ZERO EXTERNAL DEPENDENCIES):\n"
            f"1. RUST CONSTRAINTS: The code is compiled directly with standalone `rustc main.rs` without Cargo!\n"
            f"   - DO NOT USE external crates like `rand`, `rayon`, `rand_xoshiro`, or `lazy_static`.\n"
            f"   - Use ONLY Rust standard library (`std::thread`, `std::sync`, `std::time::Instant`, `std::io`).\n"
            f"   - For random numbers, implement a lightweight, ultra-fast inline PRNG such as XorShift64/128 or LCG:\n"
            f"     `struct Rng(u64); impl Rng {{ fn next_f64(&mut self) -> f64 {{ self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 as f64) / (u64::MAX as f64) }} }}`\n"
            f"   - For multi-threading in Rust, spawn threads using `std::thread::spawn` with `join()`.\n"
            f"2. C++ CONSTRAINTS: Compiled with `g++ -std=c++17 -O3 -march=native -fopenmp`. OpenMP headers (<omp.h>) are fully available.\n"
            f"3. INTEGER PRECISION & TYPES: Use 64-bit integers (`int64_t`/`uint64_t` in C++; `i64`/`u64` in Rust) to prevent arithmetic overflow.\n"
            f"4. IDENTICAL OUTPUT: Produce identical numerical output and formatting to the Python script.\n"
            f"5. RETURN ONLY CODE: Return strictly code inside markdown blocks. Do not write explanations outside code."
        )

        user_content = f"Python source code to optimize and transpile into {language}:\n```python\n{python_code}\n```"
        if profile_stats:
            user_content += f"\n\nRuntime Profiler Feedback (Hotspots):\n{profile_stats}"
        if error_feedback:
            user_content += f"\n\nCRITICAL COMPILER ERROR (Must be resolved):\n{error_feedback}\nFix all missing crates and module errors by using strictly std library and inline PRNG."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=0.1,
            max_tokens=4096
        )
        content = response.choices[0].message.content
        cleaned = content.replace("```rust", "").replace("```cpp", "").replace("```", "").strip()
        return cleaned
