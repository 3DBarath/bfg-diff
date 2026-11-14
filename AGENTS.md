# AGENTS.md

## Rules

1. **Update the research docs frequently and accordingly.** Every change to
   model mechanics, hyperparameters, CLI defaults, benchmark definitions, or
   measured results must update the matching doc in the same change:
   - `docs/research.md` — design and mechanism changes.
   - `docs/findings.md` — new measurements, bugs, fixes and limitations, with
     date, command and environment.
   - `docs/step-log.md` — append what was done and how it was verified.
   Do not leave the docs describing an older version of the code.

2. **Verify before claiming completion.** `python train.py --smoke_test` must
   print `PASS`, and `python benchmark.py --smoke_test` must run. Put the
   observed numbers in `docs/findings.md`; quote only real output.

3. **Keep the mask alias intact.** `<|extra_0|>` resolves via
   `setup_tokenizer` to a spare row of the existing Qwen2.5 embedding table
   (id 151665 of 151936). Do not resize or reinitialise embeddings for Qwen.

4. **Never commit `runs/`.** Checkpoints are multi-GB and already ignored;
   commit code and docs only.
