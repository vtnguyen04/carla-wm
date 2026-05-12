import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "runs/wm_twister/metrics.jsonl"

header = f"{'Step':>6} | {'WM_loss':>10} | {'Recon':>10} | {'KL':>8} | {'Reward':>8} | {'Actor':>10} | {'Critic':>8} | {'Entropy':>10} | {'Smooth':>10} | {'AccMean':>8} | {'SteerMn':>8} | {'Neut%':>6}"
print(header)
print("-" * len(header))

with open(path) as f:
    for line in f:
        d = json.loads(line)
        if "train/wm/loss" not in d:
            continue
        step = d["step"]
        wm = d.get("train/wm/loss", 0)
        rec = d.get("train/wm/reconstruction", 0)
        kl = d.get("train/wm/kl", 0)
        rew = d.get("train/wm/reward", 0)
        act = d.get("train/actor/loss", 0)
        crit = d.get("train/critic/loss", 0)
        ent = d.get("train/actor/actor_entropy", 0)
        sm = d.get("train/actor/actor_smoothness", 0)
        acc_m = d.get("report/Stats/Action_Acceleration_Mean", "-")
        st_m = d.get("report/Stats/Action_Steering_Mean", "-")
        neut = d.get("report/Stats/Action_Neutral_Fraction", "-")
        if isinstance(neut, float):
            neut = f"{neut*100:.1f}%"
        if isinstance(acc_m, float):
            acc_m = f"{acc_m:.3f}"
        if isinstance(st_m, float):
            st_m = f"{st_m:.3f}"
        print(f"{step:>6} | {wm:>10.1f} | {rec:>10.1f} | {kl:>8.3f} | {rew:>8.3f} | {act:>10.6f} | {crit:>8.4f} | {ent:>10.6f} | {sm:>10.2e} | {str(acc_m):>8} | {str(st_m):>8} | {str(neut):>6}")
