"""The filter table build without threads, for a WebAssembly build without pthreads.

reSIDfp builds the four filter tables (summer, mixer, gain, resonance) on four threads and joins
them. Each lambda writes its own table and reads none of the others, so running them in turn
computes the same tables. emscripten without pthreads cannot spawn a thread, and pthreads in a
browser needs SharedArrayBuffer with the cross-origin isolation headers, which GitHub Pages does
not set. This rewrites the four thread objects into four calls under __EMSCRIPTEN__.

    python3 single_thread.py <tree>/src/residfp/FilterModelConfig6581.cpp <...8580.cpp>
"""
import re
import sys
from pathlib import Path

BLOCK = re.compile(
    r"#if defined\(HAVE_CXX20\) && defined\(__cpp_lib_jthread\)\n"
    r"    using sidThread = std::jthread;\n"
    r"#else\n"
    r"    using sidThread = std::thread;\n"
    r"#endif\n\n"
    r"((?:    sidThread \w+\([\w]+\);\n)+)\n"
    r"#if !defined\(HAVE_CXX20\) \|\| !defined\(__cpp_lib_jthread\)\n"
    r"((?:    \w+\.join\(\);\n)+)"
    r"#endif\n"
)
LAMBDA = re.compile(r"    sidThread \w+\((\w+)\);")


def rewrite(match: re.Match) -> str:
    calls = "".join(f"    {name}();\n" for name in LAMBDA.findall(match.group(1)))
    return (
        "#if defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)\n"
        "    // The table builds are independent and deterministic, so one thread computes the\n"
        "    // same tables. emscripten without pthreads has no thread to spawn.\n"
        + calls
        + "#else\n"
        + match.group(0)
        + "#endif\n"
    )


def main() -> int:
    for name in sys.argv[1:]:
        path = Path(name)
        source = path.read_text()
        if "__EMSCRIPTEN_PTHREADS__" in source:
            print(f"{path}: already rewritten")
            continue
        patched, count = BLOCK.subn(rewrite, source)
        if count != 1:
            raise SystemExit(f"{path}: the thread block was found {count} times")
        path.write_text(patched)
        print(f"{path}: the table builds run in turn under emscripten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
