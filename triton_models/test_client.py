"""Test the Triton inference server with sample transliteration requests."""

from pathlib import Path

import numpy as np
import tritonclient.http as httpclient

TRITON_URL = "localhost:8000"
MODEL_NAME = "lstm_translit"
HTML_OUTPUT = Path(__file__).parent / "test_results.html"

TEST_CASES = [
    ("به نام خداوند جان و خرد", "fa2tg"),
    ("سلام دنیا", "fa2tg"),
    ("Ба номи Худованд ҷону хирад", "tg2fa"),
    ("Тоҷикистон кишвари зебост", "tg2fa"),
    ("دوستی و محبت", "fa2tg"),
]


def _infer_batch(client, texts: list[str], direction: str) -> list[str]:
    text_array = np.array([t.encode("utf-8") for t in texts], dtype=object).reshape(-1, 1)
    dir_array = np.array([direction.encode("utf-8")], dtype=object).reshape(1, 1)

    input_text = httpclient.InferInput("text", text_array.shape, "BYTES")
    input_text.set_data_from_numpy(text_array)

    input_dir = httpclient.InferInput("direction", dir_array.shape, "BYTES")
    input_dir.set_data_from_numpy(dir_array)

    output_tensor = httpclient.InferRequestedOutput("transliteration")
    response = client.infer(MODEL_NAME, inputs=[input_text, input_dir], outputs=[output_tensor])
    return [
        r.decode("utf-8") if isinstance(r, bytes) else r
        for r in response.as_numpy("transliteration").flatten()
    ]


def _write_html(results: list[tuple[str, str, str]]) -> None:
    rows = ""
    for direction, src, tgt in results:
        src_dir = "rtl" if direction == "fa2tg" else "ltr"
        tgt_dir = "ltr" if direction == "fa2tg" else "rtl"
        rows += (
            f'<tr><td class="tag">[{direction}]</td>'
            f'<td dir="{src_dir}">{src}</td>'
            f'<td dir="{tgt_dir}">{tgt}</td></tr>\n'
        )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Transliteration results</title>
<style>
  body {{ font-family: sans-serif; padding: 2em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border: 1px solid #ccc; padding: 0.5em 1em; }}
  th {{ background: #f0f0f0; }}
  .tag {{ color: #888; font-family: monospace; }}
</style>
</head>
<body>
<table>
<tr><th>Direction</th><th>Input</th><th>Output</th></tr>
{rows}</table>
</body>
</html>"""
    HTML_OUTPUT.write_text(html, encoding="utf-8")
    print(f"HTML written to {HTML_OUTPUT}")


def main() -> None:
    client = httpclient.InferenceServerClient(url=TRITON_URL)
    print(f"Server ready: {client.is_server_ready()}")
    print(f"Model ready: {client.is_model_ready(MODEL_NAME)}\n")

    results = []
    for text, direction in TEST_CASES:
        (result,) = _infer_batch(client, [text], direction)
        print(f"[{direction}] {text}")
        print(f"       -> {result}\n")
        results.append((direction, text, result))

    _write_html(results)


if __name__ == "__main__":
    main()
