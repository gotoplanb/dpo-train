import sys, time, traceback
sys.path.insert(0, ".")
import app

BASE = "mlx-community/Llama-3.2-3B-Instruct-4bit"
scenes = [
    ("a creaking door into a dark cellar","The door groans open; cold air curls around your ankles as you step down, water dripping somewhere below.","You open the door. There is a cellar. It is dark. State=CELLAR."),
    ("a suspicious merchant at a crossroads","His eyes flick to your coin purse, then up. 'Traveler,' he says, too warmly, 'you look like someone who needs options.'","A merchant NPC appears. He wants to trade. Select an option."),
    ("your torch guttering in a long tunnel","Your torch sputters, throwing jittering shadows that lunge and retreat along the wet stone; the dark ahead leans closer.","Torch durability low. Light radius decreased. Warning: torch expiring."),
    ("an old journal on a corpse","The journal is swollen with damp, its last entry gouged deep: 'It hears the light.' The hand that wrote it lies beside you.","Found item: journal. Contains lore. Reading grants +1 investigation."),
    ("a frozen lake under a green sky","The ice moans beneath your boots, hairline cracks racing out like white veins; far below, something pale turns over.","You are on a lake. It is frozen. The sky is green. Walk forward?"),
    ("a market stall of caged songbirds","A hundred small throats go silent as you pass, then erupt at once; the vendor smiles without looking up from her knife.","Birds for sale. Price: 5 gold. Buy? Y/N."),
    ("a collapsed mineshaft, lantern dying","Dust still sifts from the broken beams. Your lantern throws a shrinking coin of light; beyond it, the rubble ticks as it settles.","Mineshaft collapsed. Lantern at 12%. Exits: blocked."),
    ("a letter slid under your door at midnight","The paper is warm, as if held a long time. Three words, no signature: 'They know now.'","You receive a letter. It advances the quest. +1 clue."),
]
pairs = []
for i in range(5):  # 8 scenes x5 = 40 pairs
    for s, chosen, rejected in scenes:
        pairs.append({"prompt": f"Scene: the player encounters {s}.",
                      "chosen": chosen, "rejected": rejected})
print(f"[stress] run_dpo on {BASE}, {len(pairs)} pairs (resident models freed)", flush=True)
t0=time.time()
try:
    out = app.run_dpo(BASE, pairs, {"epochs":1, "lora_rank":16, "lora_alpha":32, "max_seq_length":512})
    print("[stress] RESULT:", out, flush=True)
except Exception:
    print("[stress] FAILED:\n"+traceback.format_exc(), flush=True)
print(f"[stress] wall={time.time()-t0:.0f}s", flush=True)
