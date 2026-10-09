# Token icons

Every token has an icon that shows its rough meaning. The icons illustrate inference in the write-up. A row of them (a "tape") is also the project logo.

There are up to 224 icons, one per token (2 sides × 16 sizes × 7 price-move classes). The style is a "comet": a coloured dot with a tail.

## How to read an icon

| Token field | Visual channel | Rule |
|---|---|---|
| Side | Colour | Blue for a buy aggressor, orange for a sell aggressor |
| Size bin (1 to 16) | Dot radius | Grows linearly with the bin index |
| Price move | Tail | Tail direction and length; no tail when the price did not move |

The tail trails behind the dot, as if the dot were moving. A rising price puts the dot high with a tail below it. A falling price puts the dot low with a tail above it. A longer tail is a larger move.

The move classes run from a large fall to a large rise. A fall and a rise of the same magnitude class have the same tail length.

| Class | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| Meaning | down large | down mid | down 1 tick | flat | up 1 tick | up mid | up large |
| Tail length $T$ | 14 | 8 | 4 | none | 4 | 8 | 14 |

## Geometry

Each icon is drawn on a $32\times32$ canvas. The tile is a rounded square of corner radius 5. Let $s\in\{0,\dots,15\}$ be the size bin (0-based) and $m\in\{0,\dots,6\}$ the move class. Let $d$ be the move direction: $-1$ for classes 0 to 2, $0$ for class 3, $+1$ for classes 4 to 6. The dot sits on the vertical centre line $x=16$. Heights $y$ are measured downwards from the top of the tile.

- Dot radius: $r = 1.6 + 0.42\,s$.
- Dot centre height: $c_y = 16 - d\,T/2$.
- Tail tip height: $y_{\text{tip}} = c_y + d\,(r + T)$.
- Tail shape: a triangle with its base across the dot centre (half-width $0.8\,r$) and its apex at $y_{\text{tip}}$.

The tail therefore always sticks out a distance $T$ beyond the dot edge, whatever the dot size. If the tail were measured from the dot centre, a large dot would hide most of it. The same move would then look smaller on a large order. The dot and tail together are centred vertically. The largest case ($s=15$, $T=14$) still fits inside the tile.

## Colours

| Use | Light background | Dark background |
|---|---|---|
| Buy dot and tail | `#2563EB` | `#5B8DEF` |
| Sell dot and tail | `#E8590C` | `#FF8A3D` |
| Tile | `#E9ECF2` | `#1B2333` |

The tail is the dot colour at 50% opacity. The logo tape uses the dark column on a `#0F1420` strip. Page text is `#14181F` on `#F4F5F7`. Fonts in the prototype were IBM Plex Sans and IBM Plex Mono.

## Uses

- **Inference figure.** Show a short run of consecutive events as a row of icons, with the model's surprisal for each event underneath. The reader can see what kind of event surprised the model.
- **Logo.** A tape of 30 to 40 icons. The prototype used a made-up sequence that goes calm, stress, calm. Stress shows as larger dots, longer tails and more switching between blue and orange. A real sequence from the test month could replace it.
- **README banner.** `docs/assets/tape.svg` is a paper ticker tape of 21 icons, 800 px wide, shown at the top of the README. It is a generated figure from a made-up calm, stress, calm sequence.
- **Key.** Two $16\times7$ grids, one per side, document the full vocabulary. Rows are size bins and columns are move classes.

## Notes

- A buy usually comes with a flat or rising price, and a sell with a flat or falling price. All combinations exist as tokens. A blue dot with a downward move is rare but valid.
- The size bins are log-spaced in BTC. The dot radius is therefore linear in the logarithm of the order size.
- Sixteen is the maximum number of size bins. If the tail rule merges bins, index the radius by the merged bin number. Then the largest bin keeps the largest radius.
- Colour carries the side and nothing else. A figure that must work in greyscale should add a text label for the side.
- The formulas above are the specification. The prototypes were built as SVG in a design canvas, and they are not stored in the repo. Four styles were compared (candle, comet, heat tile, depth bars) and the comet was chosen.
