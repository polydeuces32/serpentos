# Running SerpentOS as a self-running agent

The learning core imports nothing platform-specific, so the agent runs anywhere Python does — no terminal, no keyboard, no display. This document covers operating it as a long-lived process.

```bash
python -m serpentos bot [options]
serpentos bot [options]          # if installed with pip
```

---

## Flags

### How long to run

| Flag | Meaning |
|------|---------|
| `--episodes N` | Run N episodes, then exit (default 1000) |
| `--forever` | Run until stopped by a signal or `--max-seconds` |
| `--max-seconds S` | Wall-clock budget, checked between episodes |
| `--eval N` | Run N greedy episodes with no learning and no writes |

### What to run

| Flag | Meaning |
|------|---------|
| `--preset {DEFAULT,BOLD,CAREFUL,SPARSE}` | Hyperparameter profile |
| `--rows N` / `--cols N` | Grid size (default 22x78, about a standard terminal) |
| `--max-steps N` | Step cap per episode (default 1800) |
| `--shaping` / `--no-shaping` | Override the preset's distance shaping |
| `--seed N` | Make the whole run reproducible |

### State and output

| Flag | Meaning |
|------|---------|
| `--data-dir PATH` | Where to keep the Q-table, logs and lock (default `~/.serpentos`) |
| `--fresh` | Ignore any saved Q-table and start empty |
| `--checkpoint-every N` | Episodes between saves (default 100) |
| `--log-every N` | Episodes between progress lines (default 100) |
| `--metrics PATH` | Append one JSON object per episode |
| `--json` | Print the run summary as JSON on stdout |
| `--quiet` | Only log warnings and errors |

### Policy exchange

| Flag | Meaning |
|------|---------|
| `--bench` | Score the policy on the frozen benchmark and exit |
| `--import-policy PATH` | Load a shared policy instead of the saved checkpoint |
| `--export-policy PATH` | Write the resulting policy to a shareable file |
| `--name NAME` | Name recorded inside the exported policy |

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Finished normally, including a clean signal shutdown |
| 1 | Bad arguments, unreadable policy, or storage failure |
| 2 | The data directory is locked by another process |

---

## Lifecycle guarantees

**Signals.** `SIGINT` and `SIGTERM` set a stop flag. The agent finishes the episode it is in, writes a checkpoint, releases the lock and exits 0. A second signal exits immediately. Worst case you lose one episode.

**Checkpoints.** Written to a temp file, fsynced, then `os.replace`d over the target. A crash mid-write leaves the previous checkpoint intact. Nothing is written per step, so I/O does not throttle training.

**Locking.** The data directory takes an exclusive lock recording the owning PID. A second agent pointed at the same directory exits 2 rather than racing. A lock left behind by a killed process is reclaimed automatically once its PID is gone.

**Logs.** `training_log.csv` rotates to `.1` at 5 MB, so a permanently running agent cannot fill the disk.

---

## systemd

```ini
# /etc/systemd/system/serpentos.service
[Unit]
Description=SerpentOS self-training agent
After=network.target

[Service]
Type=simple
User=serpentos
ExecStart=/usr/bin/python3 -m serpentos bot --forever \
    --preset BOLD \
    --data-dir /var/lib/serpentos \
    --metrics /var/lib/serpentos/metrics.jsonl \
    --log-every 5000
Restart=always
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=30

# The agent only needs its own state directory.
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/serpentos

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now serpentos
journalctl -u serpentos -f
```

`Restart=always` is safe because progress is checkpointed and resumed: a restarted agent continues from its saved episode count and exploration rate.

---

## Docker

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir .
VOLUME /data
ENV PYTHONUNBUFFERED=1
STOPSIGNAL SIGTERM
ENTRYPOINT ["python", "-m", "serpentos", "bot"]
CMD ["--forever", "--data-dir", "/data"]
```

```bash
docker build -t serpentos .
docker run -d --name serpentos -v serpentos-data:/data serpentos
docker logs -f serpentos
docker stop serpentos          # SIGTERM: finishes the episode and saves
```

Mount a volume at `/data`. The container filesystem is ephemeral, and an agent that cannot persist its Q-table relearns from zero on every restart.

---

## Scheduled training

```cron
# Train for 10 minutes every hour
0 * * * * /usr/bin/python3 -m serpentos bot --forever --max-seconds 600 --data-dir /var/lib/serpentos --quiet
```

`--max-seconds` is checked between episodes, so a cron run never overruns into the next one.

---

## GitHub Actions

```yaml
- name: Train and publish a policy
  run: |
    python -m serpentos bot --forever --max-seconds 300 --preset BOLD \
      --data-dir state --export-policy policy.json --name "$GITHUB_ACTOR"
- uses: actions/upload-artifact@v4
  with:
    name: policy
    path: policy.json
```

Cache or commit the `state` directory between runs and the agent keeps improving across CI runs instead of restarting each time.

---

## Running several agents at once

Each agent needs its own data directory — the lock enforces this.

```bash
for preset in DEFAULT BOLD CAREFUL SPARSE; do
  python -m serpentos bot --episodes 20000 --preset "$preset" \
    --data-dir "runs/$preset" --seed 1 --quiet &
done
wait

for preset in DEFAULT BOLD CAREFUL SPARSE; do
  echo -n "$preset "
  python -m serpentos bot --bench --data-dir "runs/$preset" --quiet | grep mean_score
done
```

With `--seed` fixed, every agent sees the same sequence, so differences between them come from the hyperparameters rather than luck.

---

## Reading the output

Progress lines go to stderr, so `--json` summaries can be piped safely:

```
episode 5000 | score 7 | avg50 6.42 | best 31 | eps 0.050 | states 1583
```

`--metrics` writes one JSON object per episode:

```json
{"ts": 1756288, "episode": 5000, "score": 7, "steps": 412, "avg50": 6.42,
 "epsilon": 0.05, "reason": "body", "states": 1583, "mode": "bot"}
```

`reason` is why the episode ended: `wall`, `body`, `truncated` (hit the step cap) or `won` (filled the grid). A rising share of `body` deaths is the normal sign of a working agent — it means the snake is living long enough to run into itself.

The run summary breaks the same information down:

```json
{
  "episodes": 20000,
  "avg_score": 6.1,
  "best_score": 34,
  "endings": {"wall": 3120, "body": 16791, "truncated": 89, "won": 0},
  "episodes_per_second": 812.4,
  "stopped_by": "signal"
}
```

---

## Tuning notes

- **`--preset BOLD` first, then `CAREFUL`.** High exploration finds the food quickly; low learning rates stabilise what was found.
- **Grid size matters.** A policy trained on 22x78 is not tuned for 10x30. The state is mostly local, so it transfers, but not perfectly. Train at the size you play at.
- **`SPARSE` needs far more episodes.** Without distance shaping the agent only learns from eating and dying, so budget an order of magnitude more.
- **Watch `states`, not just score.** A table that has stopped growing means exploration has collapsed; raising epsilon or using `--fresh` may beat grinding more episodes.
- **The state has eight fields and no memory of the body's shape.** Tabular Q-learning plateaus once the snake is long enough to trap itself — the agent cannot see the coil it is building. Getting past that plateau needs a richer state or a function approximator, not more episodes.
