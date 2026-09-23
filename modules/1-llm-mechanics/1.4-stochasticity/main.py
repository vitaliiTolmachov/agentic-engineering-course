"""Stochasticity demo for Module 1 (Lecture 1.4).

Запускає той самий промпт з трьома значеннями temperature (0.0, 0.7, 1.0)
і показує різницю у варіативності відповідей.
"""

import os
import sys
from difflib import SequenceMatcher

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"

PROMPT = (
    "Поясни різницю між Task і ValueTask у C#, та наведи приклади, "
    "коли використання ValueTask дійсно виправдане для оптимізації пам'яті."
)

RUNS_PER_TEMP = 3


def require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        sys.exit(1)


def call(client: Anthropic, prompt: str, temperature: float) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def run_batch(
    client: Anthropic, label: str, prompt: str, temperature: float
) -> list[str]:
    print("=" * 80)
    print(f"--- {label} (T={temperature}, {RUNS_PER_TEMP} runs) ---")
    print("=" * 80)

    outputs = []
    for i in range(1, RUNS_PER_TEMP + 1):
        out = call(client, prompt, temperature)
        outputs.append(out)

        print(f"\n>>> RUN {i} (Temperature = {temperature}) <<<\n")
        print(out)
        print("-" * 40)

    pairs = [
        (i + 1, j + 1, similarity(outputs[i], outputs[j]))
        for i in range(len(outputs))
        for j in range(i + 1, len(outputs))
    ]
    avg = sum(s for _, _, s in pairs) / max(len(pairs), 1)
    pair_str = ", ".join(f"{i}~{j}={s:.2f}" for i, j, s in pairs)

    print(f"\n[Pairwise similarity (1.00 = identical): {pair_str}, avg={avg:.2f}]\n")
    return outputs


def main() -> None:
    require_api_key()
    client = Anthropic()

    print("\nSTOCHASTICITY DEMO FOR C# PROMPT")
    print(f"Model: {MODEL}")
    print(f'Prompt: "{PROMPT}"\n')

    # Виконуємо 3 прогони для кожної з трьох температур
    run_batch(client, "Temperature 0.0", PROMPT, 0.0)
    run_batch(client, "Temperature 0.7", PROMPT, 0.7)
    run_batch(client, "Temperature 1.0", PROMPT, 1.0)


if __name__ == "__main__":
    main()