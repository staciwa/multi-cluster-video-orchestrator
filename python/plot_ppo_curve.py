import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def plot_tensorboard_data():
    log_dir = "./ppo_tensorboard"
    
    # Find the latest MaskablePPO_x folder inside ppo_tensorboard
    subdirs = glob.glob(os.path.join(log_dir, "MaskablePPO_*"))
    if not subdirs:
        print("No training directories found in", log_dir)
        return
        
    latest_subdir = max(subdirs, key=os.path.getmtime)
    print(f"Reading logs from: {latest_subdir}")
    
    event_acc = EventAccumulator(latest_subdir)
    event_acc.Reload()
    
    # Extract mean reward
    tag = 'rollout/ep_rew_mean'
    if tag not in event_acc.Tags()['scalars']:
        print(f"Tag {tag} not found in logs.")
        return
        
    events = event_acc.Scalars(tag)
    steps = [e.step for e in events]
    rewards = [e.value for e in events]
    
    # Styling for master thesis (scientific and clear style)
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plotting smoothed and raw line
    plt.plot(steps, rewards, color='royalblue', alpha=0.3, linewidth=2, label="Raw mean reward")
    
    # Moving Average smoothing
    window_size = max(1, len(rewards) // 20)
    import numpy as np
    smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    smooth_steps = steps[(window_size-1):]
    plt.plot(smooth_steps, smoothed, color='navy', linewidth=3, label=f"Smoothed trend")
    
    plt.title('MaskablePPO Learning Curve (Episodic Reward)', pad=20, fontweight='bold')
    plt.xlabel('Environment Training Steps')
    plt.ylabel('Mean Cumulative Reward per Episode')
    plt.legend(loc="lower right")
    
    # Axis formatting
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    
    out_path = "learning_curve_ppo.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.savefig("learning_curve_ppo.png", dpi=300, bbox_inches='tight')
    print(f"Saved plots as {out_path} and .png!")

if __name__ == "__main__":
    plot_tensorboard_data()
