# Ryzanstein LLM Crash Debug Session — Continuation State

## EXECUTIVE SUMMARY
- **Date**: April 7, 2026
- **Status**: CRASH ISOLATED, waiting for granular intra-function diagnostics
- **Build**: ✅ COMPLETE (ryzen_llm_bindings.pyd 404.5 KB)
- **Runtime**: ❌ CRASH (forward() crashes inside attention_layer())
- **Crash Location**: DEFINITIVE — inside attention_layer() (engine.cpp lines 687-820)
- **Exact Operation**: AFTER Q-projection matvec completes (verified: all 32 rows), BEFORE attention_layer() returns

## BREAKTHROUGH DIAGNOSTIC EVIDENCE
Most recent test output proves:
```
[FWD] embedding done ✅
[FWD] layer 0 attn_norm done ✅
[DIAG] matmul complete, all 32 rows done ✅ (Q-projection)
[FWD] layer 0 attention done ❌ NEVER PRINTS (CRASH)
```

## NEXT EXACT STEPS (NO LOOPING)
1. Read attention_layer() in engine.cpp (lines ~687-820)
2. Add granular cerr diagnostics AFTER each sub-op inside function:
   - After input quantization
   - IMMEDIATELY AFTER K-projection matvec (~line 711)
   - IMMEDIATELY AFTER V-projection matvec (~line 717)
   - IMMEDIATELY AFTER O-projection matvec (~line 808)
3. Incremental rebuild (command saved)
4. Test with $env:OMP_NUM_THREADS = "1" (command saved)
5. Identify which operation crashes
6. Fix root cause
7. Clean diagnostics
8. Final test

## COMMANDS READY TO USE
- Full rebuild: `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 && pwsh -NoProfile -ExecutionPolicy Bypass -File "s:\Ryot\scripts\compile_bindings.ps1"'`
- Incremental: `cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 && cmake --build s:\Ryot\RYZEN-LLM\build --config Release -j 16 2>&1' | Select-Object -Last 3`
- Test: `$env:OMP_NUM_THREADS = "1"; python -c "import sys; sys.path.insert(0, r's:\Ryot\RYZEN-LLM\python'); from ryzanstein_llm.ryzen_llm_bindings import BitNetEngine, ModelConfig; cfg = ModelConfig(); cfg.vocab_size = 64; cfg.hidden_size = 32; cfg.intermediate_size = 86; cfg.num_layers = 1; cfg.num_heads = 2; cfg.head_dim = 16; cfg.max_seq_length = 32; engine = BitNetEngine(cfg); engine.load_weights('_'); output = engine.forward(0, 0); print(f'Done: {len(output)} logits')" 2>&1`

## FULL STATE FILE
`s:\Ryot\CRASH_DEBUG_SESSION_CONTINUATION.txt` — comprehensive reference with all details, environment setup, and command documentation
