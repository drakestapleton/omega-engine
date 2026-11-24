# omega-engine Agent Rules

- `omega-mv.lock` pins the exact `omega-mv` commit that provides `om/gpu/engine/` and `seed/out/omc`.
- Never modify or delete files under `evidence/`.
- Always disclose Linux kernel open modules (`nvidia`/`nvidia-uvm`) and signed GSP firmware when describing Zero-CUDA execution.
- Land and verify changes on Interplane first (`spark:git/omega-engine.git`), then push to GitHub.
