# THE 1000-SCREEN MOSAIC

A live, GPU-rendered cinematic animation. Thousands of miniature photographic
screens drift in the dark, swarm, and assemble into the portrait in
`assets/reference/target.png`, then the camera pulls back to reveal the mosaic.

## Run

```bash
pip install -r requirements.txt
python main.py
```

The window opens immediately and starts playing (about 19 s, then it holds on the
final frame). Nothing is pre-rendered and no video is written.

| Key | Action |
|-----|--------|
| SPACE | Pause / resume |
| R | Restart |
| F | Toggle fullscreen |
| ESC | Exit |
| Q | Cycle quality (low / medium / high) |
| B | Toggle post-FX (bloom, depth of field) |

Options: `--quality low|medium|high`, `--count N` (number of screens),
`--scale 0.7` (internal render resolution), `--width/--height`, `--fullscreen`.
The frame rate is shown in the window title. If the frame rate drops, the program
lowers its internal resolution automatically (`--no-adaptive` turns that off).

Requires OpenGL 3.3 (any GPU from the last ~10 years).

## Your photographs

* `assets/photos/` - every `.jpg/.png/.webp/.bmp` here is loaded automatically.
  The program cuts 270 different crops (face close-ups, details, wider shots,
  flips, light colour grades) from them and spreads those across the screens.
  **Two photos are included. Add 20-100 more stills of the same person for far
  less repetition.** If the folder is empty, procedural placeholders are used.
* `assets/reference/target.png` - the final portrait. Replace it to build a
  different picture (any aspect ratio is kept).
* `assets/reference/target.json` - `focus` = `[x, y, rx, ry]` (0..1) marks the face
  so more screens are spent on it. Delete the file to auto-detect the face.

## How it works

* **Layout:** a quadtree over the target image. Flat areas get large screens;
  eyes, nose, mouth, beard and the face outline get the smallest ones.
* **Screens:** instanced 3D boxes with a bezel, thickness, glass reflections and
  coloured rim light, all drawn from one mip-mapped photo atlas.
* **Colour lock:** as each screen lands, its photo is tone-matched to the reference
  at that spot while keeping its own detail, so the portrait reads from a distance
  and the individual photos are still visible up close.
* **Sequence:** darkness (0-3 s), swarm (3-7), reconstruction head-to-clothing
  (7-13), reveal with golden glow and light sweep (13-17), title (17-19).
* **Look:** HDR buffer, MSAA, motion trails, depth of field, bloom, tone mapping,
  vignette, film grain.

## Testing without a display

`python main.py --dump out_folder --times 1,5,9,12,15,18 --width 1280 --height 720`
renders those moments to PNG using offscreen OpenGL (EGL).
