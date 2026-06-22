import sys, time, traceback
sys.path.insert(0, ".")
import app

# Tiny synthetic preference set (code-flavored chosen/rejected on the same prompt).
pairs = [
    {"prompt": "Write a Rust fn that adds two i64s.",
     "chosen": "pub fn add(a: i64, b: i64) -> i64 { a + b }",
     "rejected": "pub fn add(a,b){a+b}  // not valid Rust"},
    {"prompt": "Reverse a String in Rust.",
     "chosen": "pub fn rev(s: &str) -> String { s.chars().rev().collect() }",
     "rejected": "pub fn rev(s){ return s.reverse() }"},
    {"prompt": "Is 7 prime? Rust fn.",
     "chosen": "pub fn is_prime(n:u64)->bool{ if n<2 {return false} (2..n).all(|d| n%d!=0) }",
     "rejected": "pub fn is_prime(n){ n%2!=0 }"},
    {"prompt": "Sum a slice of i64 in Rust.",
     "chosen": "pub fn sum(v:&[i64])->i64{ v.iter().sum() }",
     "rejected": "pub fn sum(v){ let mut s; for x in v {s+=x} s }"},
]
print(f"[smoke] starting run_dpo on mlx-community/gemma-3-1b-it-4bit, {len(pairs)} pairs", flush=True)
t0=time.time()
try:
    out = app.run_dpo("mlx-community/gemma-3-1b-it-4bit", pairs,
                      {"epochs":1, "lora_rank":8, "lora_alpha":16, "max_seq_length":512})
    print("[smoke] RESULT:", out, flush=True)
except Exception:
    print("[smoke] FAILED:\n" + traceback.format_exc(), flush=True)
print(f"[smoke] wall={time.time()-t0:.0f}s", flush=True)
