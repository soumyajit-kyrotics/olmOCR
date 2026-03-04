#!/usr/bin/env python3
"""Combined YAML generation stats: session speed + per-folder coverage + ETA."""
import re, os, glob, subprocess
from datetime import datetime

# ── Session stats ──────────────────────────────────────────────────────────────
with open('synth_output/logs/yaml_progress.log', 'r') as f:
    lines = f.readlines()

last_session_start = None
for i, line in enumerate(lines):
    if '=== YAML SESSION START ===' in line:
        last_session_start = i

session_time = None
for line in lines[last_session_start:last_session_start+5]:
    m = re.search(r'Time\s+:\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    if m:
        session_time = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        break

session_lines = lines[last_session_start:]
done_count = sum(1 for l in session_lines if '[DONE ]' in l)
fail_count = sum(1 for l in session_lines if '[FAIL ]' in l)
skip_count = sum(1 for l in session_lines if '[SKIP ]' in l)

times, tokens = [], []
for line in session_lines:
    m = re.search(r'\[DONE \].*?(\d+\.\d+)s\b.*?completion=(\d+) total=(\d+)', line)
    if m:
        times.append(float(m.group(1)))
        tokens.append(int(m.group(3)))

last10_t   = times[-10:] if len(times) >= 10 else times
last10_tok = tokens[-10:] if len(tokens) >= 10 else tokens
avg_t      = sum(last10_t) / len(last10_t) if last10_t else 0
avg_tok    = sum(last10_tok) // len(last10_tok) if last10_tok else 0

last_done = None
for line in reversed(session_lines):
    m = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\].*\[DONE \]', line)
    if m:
        last_done = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
        break

now     = datetime.now()
elapsed = (now - session_time).total_seconds()
speed   = done_count / (elapsed / 60) if elapsed > 0 else 0

print(f'Session started  : {session_time}')
print(f'Elapsed          : {elapsed/60:.1f} min')
print(f'Pages DONE       : {done_count}')
print(f'Pages FAILED     : {fail_count}')
print(f'Pages SKIPPED    : {skip_count}')
print(f'Speed            : {speed:.2f} pages/min ({speed*60:.0f} pages/hr)')
print(f'Avg page time    : {avg_t:.1f}s  (last 10)')
print(f'Avg total tokens : {avg_tok}  (last 10)')
print(f'Last DONE at     : {last_done}')

# ── Per-folder coverage ────────────────────────────────────────────────────────
print()
print('Counting folder coverage (takes ~30s)...')

all_yamls = set(os.path.basename(f) for f in glob.glob('synth_output/yaml/*.yaml'))

def count_folder(folder):
    total, done = 0, 0
    for pdf in glob.glob(f'data/{folder}/**/*.pdf', recursive=True):
        pdf_name = re.sub(r'\.pdf$', '', os.path.basename(pdf), flags=re.IGNORECASE)
        try:
            r = subprocess.run(['pdfinfo', pdf], capture_output=True, text=True)
            for l in r.stdout.splitlines():
                if l.startswith('Pages:'):
                    n = int(l.split(':')[1].strip())
                    total += n
                    done  += sum(1 for pg in range(1, n+1)
                                 if f'{pdf_name}_page_{pg}.yaml' in all_yamls)
                    break
        except:
            pass
    return total, done

grand_total = grand_done = 0
results = {}
for folder in ['hc', 'hi', 'sc']:
    t, d = count_folder(folder)
    results[folder] = (t, d)
    grand_total += t
    grand_done  += d

print()
print(f"{'Folder':<8} {'Total pages':>12} {'YAMLs done':>12} {'Remaining':>10} {'Coverage':>10}")
print('-' * 56)
for folder, (t, d) in results.items():
    print(f"{folder:<8} {t:>12} {d:>12} {t-d:>10} {d/t*100:>9.1f}%")
print('-' * 56)
print(f"{'TOTAL':<8} {grand_total:>12} {grand_done:>12} {grand_total-grand_done:>10} {grand_done/grand_total*100:>9.1f}%")

# ETA based on sc remaining and current speed
sc_t, sc_d = results['sc']
sc_remaining = sc_t - sc_d
eta_hrs = sc_remaining / (speed * 60) if speed > 0 else float('inf')
print()
print(f'SC remaining     : {sc_remaining}')
print(f'ETA (sc only)    : {eta_hrs:.1f} hrs  ({eta_hrs/24:.1f} days) at current speed')
