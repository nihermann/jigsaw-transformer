import argparse
import base64
import json
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

import main as m


def tensor_image_to_data_url(image_tensor: torch.Tensor) -> str:
    img = image_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1)
    arr = (img.numpy() * 255).astype("uint8")
    pil_img = Image.fromarray(arr)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_reordered_slot_to_scr(pred_positions_row: list[int], n_patches: int) -> list[int]:
    # Mirrors main.reorder_patches_from_predictions: last writer to a slot wins.
    mapping = [-1] * n_patches
    for scr_idx, slot in enumerate(pred_positions_row):
        mapping[slot] = scr_idx
    return mapping


def build_view_data(run_dir: Path, num_images: int) -> dict:
    with (run_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = m.Config(**json.load(f))
    m.set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = torch.load(run_dir / "patch_puzzle_transformer.pth", map_location=device)
    model = m.PatchPuzzleTransformer(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_chans=cfg.num_channels,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(weights)
    model.eval()

    val_loader = m.prepare_dataloader(cfg, train=False)
    images, _ = next(iter(val_loader))
    images = images[:num_images].to(device)

    patches = m.images_to_patches(images, patch_size=cfg.patch_size)
    scrambled_patches, targets = m.scramble_patches(patches)
    logits, attn_layers = model(scrambled_patches, return_attention=True)
    pred_positions = logits.argmax(dim=-1)

    reordered_patches = m.reorder_patches_from_predictions(scrambled_patches, pred_positions)

    scrambled_images = m.patches_to_images(
        scrambled_patches,
        patch_size=cfg.patch_size,
        image_size=cfg.image_size,
        num_channels=cfg.num_channels,
    )
    reordered_images = m.patches_to_images(
        reordered_patches,
        patch_size=cfg.patch_size,
        image_size=cfg.image_size,
        num_channels=cfg.num_channels,
    )

    # Accumulate attention over blocks and heads: (B, N, N)
    # attn layer shape: (B, num_heads, N, N)
    stacked_attn = torch.stack(attn_layers, dim=0)
    attn_accum = stacked_attn.mean(dim=(0, 2))
    attn_accum = attn_accum / (attn_accum.max(dim=-1, keepdim=True).values + 1e-8)

    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    grid_size = cfg.image_size // cfg.patch_size

    samples = []
    for idx in range(images.size(0)):
        targets_row = targets[idx].detach().cpu().tolist()
        pred_row = pred_positions[idx].detach().cpu().tolist()
        orig_to_scr = [0] * num_patches
        for scr_idx, orig_idx in enumerate(targets_row):
            orig_to_scr[orig_idx] = scr_idx

        slot_to_scr = build_reordered_slot_to_scr(pred_row, num_patches)
        patch_acc = sum(int(pred_row[i] == targets_row[i]) for i in range(num_patches)) / num_patches

        samples.append(
            {
            "index": idx,
            "original": tensor_image_to_data_url(images[idx]),
            "scrambled": tensor_image_to_data_url(scrambled_images[idx]),
            "reordered": tensor_image_to_data_url(reordered_images[idx]),
            "targets": targets_row,
            "pred": pred_row,
            "orig_to_scr": orig_to_scr,
            "reordered_slot_to_scr": slot_to_scr,
            "attention": attn_accum[idx].detach().cpu().tolist(),
            "patch_acc": patch_acc,
            }
        )

    return {
        "meta": {
            "image_size": cfg.image_size,
            "patch_size": cfg.patch_size,
            "grid_size": grid_size,
            "num_patches": num_patches,
            "num_images": len(samples),
            "run_dir": str(run_dir),
        },
        "samples": samples,
    }


def render_html(payload: dict) -> str:
    payload_json = json.dumps(payload)
    return f"""<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Patch Attention Explorer</title>
    <style>
        :root {{
            --bg: #f4f6f8;
            --card: #ffffff;
            --text: #1b1f24;
            --muted: #5d6773;
            --border: #d5dde5;
            --accent: #007f8b;
            --heat: 29, 105, 150;
            --highlight: #ff5a1f;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            margin: 0;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
            color: var(--text);
            background: radial-gradient(circle at 20% -10%, #ffffff 0%, #eef3f6 45%, #e8eef2 100%);
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ margin: 0 0 8px 0; font-size: 1.8rem; }}
        .subtitle {{ color: var(--muted); margin-bottom: 20px; }}
        .sample-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(390px, 1fr));
            gap: 16px;
        }}
        .sample {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        }}
        .sample-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 10px;
            font-size: 0.92rem;
        }}
        .score {{ color: var(--accent); font-weight: 600; }}
        .panels {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }}
        .panel-title {{
            font-size: 0.78rem;
            margin-bottom: 4px;
            color: var(--muted);
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .image-wrap {{
            position: relative;
            width: 100%;
            aspect-ratio: 1 / 1;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
            background: #e9eef2;
        }}
        .image-wrap img {{
            width: 100%;
            height: 100%;
            display: block;
            image-rendering: pixelated;
        }}
        .grid-overlay,
        .heat-overlay,
        .cells-overlay {{
            position: absolute;
            inset: 0;
            display: grid;
            pointer-events: none;
        }}
        .grid-overlay {{
            background-image:
                linear-gradient(to right, rgba(255,255,255,0.45) 1px, transparent 1px),
                linear-gradient(to bottom, rgba(255,255,255,0.45) 1px, transparent 1px);
            background-size: calc(100% / var(--grid-size)) calc(100% / var(--grid-size));
            mix-blend-mode: screen;
            z-index: 1;
        }}
        .heat-cell {{
            background: rgba(var(--heat), 0);
            transition: background 80ms linear;
        }}
        .cells-overlay {{ pointer-events: auto; z-index: 3; }}
        .hover-cell {{
            border: 1px solid rgba(255,255,255,0.18);
            background: transparent;
            cursor: crosshair;
        }}
        .hover-cell.active {{
            border: 2px solid var(--highlight);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.9);
        }}
        .legend {{
            margin-top: 16px;
            font-size: 0.9rem;
            color: var(--muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-bar {{
            width: 140px;
            height: 10px;
            border-radius: 999px;
            background: linear-gradient(to right, rgba(var(--heat), 0.05), rgba(var(--heat), 0.85));
            border: 1px solid var(--border);
        }}
        .hint {{ margin-top: 4px; color: var(--muted); font-size: 0.85rem; }}
    </style>
</head>
<body>
    <div class=\"container\">
        <h1>Patch Attention Explorer</h1>
        <div class=\"subtitle\">Hover any patch. Attention scores are accumulated over all transformer blocks and heads, then synchronized across original, scrambled, and reordered views.</div>
        <div id=\"sample-grid\" class=\"sample-grid\"></div>
        <div class=\"legend\"><span>low</span><div class=\"legend-bar\"></div><span>high attention</span></div>
        <div class=\"hint\">Dataset slice: first 8 images from validation loader.</div>
    </div>

    <script>
        const payload = {payload_json};
        const meta = payload.meta;
        document.documentElement.style.setProperty('--grid-size', String(meta.grid_size));

        function mk(tag, className) {{
            const el = document.createElement(tag);
            if (className) el.className = className;
            return el;
        }}

        function createPanel(sample, panelType, title, imageSrc) {{
            const panel = mk('div', 'panel');
            const t = mk('div', 'panel-title');
            t.textContent = title;
            panel.appendChild(t);

            const wrap = mk('div', 'image-wrap');
            const img = mk('img');
            img.src = imageSrc;
            img.alt = title;
            wrap.appendChild(img);

            const grid = mk('div', 'grid-overlay');
            wrap.appendChild(grid);

            const heat = mk('div', 'heat-overlay');
            heat.style.gridTemplateColumns = `repeat(${{meta.grid_size}}, 1fr)`;
            heat.style.gridTemplateRows = `repeat(${{meta.grid_size}}, 1fr)`;

            const cells = mk('div', 'cells-overlay');
            cells.style.gridTemplateColumns = `repeat(${{meta.grid_size}}, 1fr)`;
            cells.style.gridTemplateRows = `repeat(${{meta.grid_size}}, 1fr)`;

            const heatCells = [];
            const hoverCells = [];
            for (let i = 0; i < meta.num_patches; i++) {{
                const hc = mk('div', 'heat-cell');
                heat.appendChild(hc);
                heatCells.push(hc);

                const c = mk('div', 'hover-cell');
                c.dataset.patchIndex = String(i);
                c.dataset.panelType = panelType;
                cells.appendChild(c);
                hoverCells.push(c);
            }}

            wrap.appendChild(heat);
            wrap.appendChild(cells);
            panel.appendChild(wrap);
            return {{ panel, heatCells, hoverCells }};
        }}

        function scrambleIndexFromPanel(sample, panelType, panelPatchIndex) {{
            if (panelType === 'scrambled') return panelPatchIndex;
            if (panelType === 'original') return sample.orig_to_scr[panelPatchIndex];
            if (panelType === 'reordered') return sample.reordered_slot_to_scr[panelPatchIndex];
            return -1;
        }}

        function heatValue(v) {{
            const clamped = Math.max(0, Math.min(1, v));
            return `rgba(${{getComputedStyle(document.documentElement).getPropertyValue('--heat')}}, ${{(0.85 * clamped).toFixed(4)}})`;
        }}

        function renderSampleCard(sample) {{
            const card = mk('div', 'sample');

            const header = mk('div', 'sample-header');
            const title = mk('div');
            title.textContent = `Sample #${{sample.index}}`;
            const score = mk('div', 'score');
            score.textContent = `Patch acc: ${{(sample.patch_acc * 100).toFixed(1)}}%`;
            header.appendChild(title);
            header.appendChild(score);
            card.appendChild(header);

            const panels = mk('div', 'panels');
            const original = createPanel(sample, 'original', 'Original (patchified)', sample.original);
            const scrambled = createPanel(sample, 'scrambled', 'Scrambled', sample.scrambled);
            const reordered = createPanel(sample, 'reordered', 'Predicted reorder', sample.reordered);

            panels.appendChild(original.panel);
            panels.appendChild(scrambled.panel);
            panels.appendChild(reordered.panel);
            card.appendChild(panels);

            const panelMap = {{
                original,
                scrambled,
                reordered,
            }};

            function clearAll() {{
                for (const p of Object.values(panelMap)) {{
                    for (const hc of p.heatCells) hc.style.background = 'rgba(0,0,0,0)';
                    for (const c of p.hoverCells) c.classList.remove('active');
                }}
            }}

            function applyHeat(panelType, scoreByPatch) {{
                const p = panelMap[panelType];
                for (let panelPatch = 0; panelPatch < meta.num_patches; panelPatch++) {{
                    const scrIdx = scrambleIndexFromPanel(sample, panelType, panelPatch);
                    const v = scrIdx >= 0 ? scoreByPatch[scrIdx] : 0;
                    p.heatCells[panelPatch].style.background = heatValue(v);
                }}
            }}

            function handleHover(panelType, panelPatchIndex) {{
                const scrIdx = scrambleIndexFromPanel(sample, panelType, panelPatchIndex);
                if (scrIdx < 0) {{
                    clearAll();
                    return;
                }}

                const scores = sample.attention[scrIdx];
                applyHeat('original', scores);
                applyHeat('scrambled', scores);
                applyHeat('reordered', scores);

                for (const p of Object.values(panelMap)) {{
                    for (const c of p.hoverCells) c.classList.remove('active');
                }}
                panelMap.scrambled.hoverCells[scrIdx].classList.add('active');
                panelMap.original.hoverCells[sample.targets[scrIdx]].classList.add('active');
                panelMap.reordered.hoverCells[sample.pred[scrIdx]].classList.add('active');
            }}

            for (const [panelType, panelObj] of Object.entries(panelMap)) {{
                for (const c of panelObj.hoverCells) {{
                    c.addEventListener('mouseenter', () => handleHover(panelType, Number(c.dataset.patchIndex)));
                    c.addEventListener('mouseleave', () => clearAll());
                }}
            }}

            return card;
        }}

        const grid = document.getElementById('sample-grid');
        for (const sample of payload.samples) grid.appendChild(renderSampleCard(sample));
    </script>
</body>
</html>
"""


def save_html(run_dir: Path, output_html: Path, num_images: int) -> Path:
    payload = build_view_data(run_dir=run_dir, num_images=num_images)
    html = render_html(payload)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    return output_html


def notebook_iframe_html(output_html: Path, width: int = 1200, height: int = 920) -> str:
    return (
        f'<iframe src="{output_html.as_posix()}" width="{width}" height="{height}" '
        'style="border:1px solid #d5dde5;border-radius:8px;"></iframe>'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build interactive patch-attention viewer as an HTML page.")
    parser.add_argument("--run-dir", help="Path to W&B run files directory")
    parser.add_argument("--output", default="attention_viewer.html", help="Output HTML file")
    parser.add_argument("--num-images", type=int, default=9, help="How many validation images to render")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_html = Path(args.output)
    result = save_html(run_dir=run_dir, output_html=output_html, num_images=args.num_images)
    result_path = result.resolve()
    result_path_str = result_path.as_posix()

    print("Patch attention viewer saved.")
    print(f"HTML file: {result_path_str}")
    print("Open it in a browser.")


if __name__ == "__main__":
    main()
