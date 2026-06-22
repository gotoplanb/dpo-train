import sys, time, traceback
sys.path.insert(0, ".")
import app

BASE = "mlx-community/Llama-3.2-3B-Instruct-4bit"
# Adventure-engine-flavored preference pairs: immersive 2nd-person narration
# (chosen) vs flat/meta/jargon narration (rejected) on the same scene prompt.
pairs = [
    {"prompt": "Scene: the player opens a creaking door into a dark cellar.",
     "chosen": "The door groans open. Cold air curls around your ankles as you step down; somewhere below, water drips in the dark.",
     "rejected": "You open the door. There is a cellar. It is dark. The game state is now CELLAR."},
    {"prompt": "Scene: the player meets a suspicious merchant at a crossroads.",
     "chosen": "The merchant's eyes flick to your coin purse, then up to your face. 'Traveler,' he says, too warmly. 'You look like someone who needs... options.'",
     "rejected": "A merchant NPC appears. He wants to trade. Select an option from the menu to continue."},
    {"prompt": "Scene: the player's torch begins to gutter in a long tunnel.",
     "chosen": "Your torch sputters, throwing jittering shadows that lunge and retreat along the wet stone. The dark ahead seems to lean closer.",
     "rejected": "Torch durability is low. Light radius decreased. Warning: torch will expire soon."},
    {"prompt": "Scene: the player finds an old journal on a corpse.",
     "chosen": "The journal is swollen with damp, its last entry a single line gouged deep: 'It hears the light.' The hand that wrote it lies still beside you.",
     "rejected": "You found item: journal. It contains lore text. Reading it grants +1 to investigation."},
]
print(f"[adv] run_dpo on {BASE}, {len(pairs)} adventure pairs", flush=True)
t0=time.time()
try:
    out = app.run_dpo(BASE, pairs, {"epochs":1, "lora_rank":8, "lora_alpha":16, "max_seq_length":512})
    print("[adv] RESULT:", out, flush=True)
except Exception:
    print("[adv] FAILED:\n"+traceback.format_exc(), flush=True)
print(f"[adv] wall={time.time()-t0:.0f}s", flush=True)
