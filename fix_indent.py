with open('scripts/train/train_milestone_b.py', 'r') as f:
    content = f.read()
content = content.replace(
    '                        print(f"  [val @ step {step}] {json.dumps(val_metrics)}")',
    '                        print(f"  [val @ step {step}] {json.dumps(val_metrics)}")'
)
with open('scripts/train/train_milestone_b.py', 'w') as f:
    f.write(content)
print('Fixed')