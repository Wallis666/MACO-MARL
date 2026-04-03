"""解析 TensorBoard 事件文件，输出关键指标摘要。"""
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

path = sys.argv[1] if len(sys.argv) > 1 else "runs/phase3_ctx/tb"

ea = EventAccumulator(path)
ea.Reload()

tags = sorted(ea.Tags().get("scalars", []))
print(f"标量标签 ({len(tags)}):")
for t in tags:
    print(f"  {t}")
print()

for tag in tags:
    events = ea.Scalars(tag)
    if not events:
        continue
    vals = [e.value for e in events]
    steps = [e.step for e in events]
    n_tail = max(1, len(vals) // 5)
    tail_avg = sum(vals[-n_tail:]) / n_tail
    print(f"--- {tag} ---")
    print(f"  数据点: {len(vals)}  步数: {steps[0]} -> {steps[-1]}")
    print(f"  初始: {vals[0]:.6f}  最终: {vals[-1]:.6f}  后20%均值: {tail_avg:.6f}")
    print(f"  最小: {min(vals):.6f}  最大: {max(vals):.6f}")
    print()

print("=" * 100)
print("趋势表")
print("=" * 100)
key_tags = [t for t in tags if t in [
    "return/cheetah_run", "return/cheetah_run_backwards",
    "train/avg_return", "train/dynamics_loss",
    "train/reward_loss", "train/ctx_loss",
]]
milestones = list(range(0, int(steps[-1]) + 100000, 100000))
header = f"{'step':>8}"
for tag in key_tags:
    short = tag.split("/")[-1][:14]
    header += f"  {short:>14}"
print(header)
print("-" * len(header))
for ms in milestones:
    row = f"{ms:>8}"
    for tag in key_tags:
        events = ea.Scalars(tag)
        best = None
        for e in events:
            if best is None or abs(e.step - ms) < abs(best.step - ms):
                best = e
        if best and abs(best.step - ms) < 50000:
            row += f"  {best.value:>14.4f}"
        else:
            row += f"  {'N/A':>14}"
    print(row)
