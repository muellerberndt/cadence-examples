# 06 · Sign

Show the net a sign, and it writes what it sees with a two-joint arm. This rung ties
perception to action: pixels in, a motor program out, and the same free/nudged rule as
every other rung. Read [How a patch net learns](../HOW_IT_LEARNS.md) first.

```bash
pip install "cadence-net>=0.2"
python train.py                    # under ten minutes on a laptop core
python build_page.py               # embeds net.json into index.html; open it and watch it write
python train.py --verify receipt.json
```

## 1. The arm, the canvas, the signs

A 16×16 canvas sits in the workspace of a planar arm with two unit links anchored at the
origin. The pen sits on a canvas cell and moves, each step, to one of its eight
neighbours or stays: nine actions. The joints follow the pen by inverse kinematics, the
way a spinal reflex follows an intended hand position, and they are what the page draws.
Every cell the pen visits is inked.

A sign is one of six single-stroke shapes (bar, dash, slash, L, V, Z) at a random size and
place, rasterised to an ordered list of cells: its stroke. The pen starts on the stroke's
first cell.

## 2. What the net sees, and what it does

The net sees pixels only, through a 7×7 window centred on its pen: the sign's cells in that
window and the cells it has already inked, 98 input owners. It does not see the whole
canvas and it does not see its joint angles. The window is what makes the task learnable
by a net with no convolutions: the rule "head for the next sign cell that is not yet
inked" looks the same wherever the pen is, so one set of seams serves every position.

Nine output owners are the nine pen moves. To act, the net settles under the window's
clamp and takes the most active output. Forty steps write a sign.

## 3. How it learns

A teacher writes every sign perfectly: at each step it heads for the first stroke cell not
yet inked, and stays when none is left. The net imitates it: 2,160 demonstrated signs,
half of them with random slips of the pen so that the demonstrations contain recoveries
(the label is always what the teacher would do from the state the pen is actually in).
That is 86,400 rows of (window, teacher's move), and learning is classification with
nine classes, exactly section 3 of [How a patch net learns](../HOW_IT_LEARNS.md): a free
settlement, a settlement nudged toward the teacher's move and one away, and every seam
moving on its own two endpoints. Eight epochs.

The first version of this rung asked the net to choose joint-angle changes directly, from
the whole canvas plus a map of its pen. Neither the patch net nor an MLP learned it: the
right joint motion for a given pen direction depends on where in the workspace the arm is,
a geometry that one hidden layer does not extract from pixels in an evening. Moving the
action to pen directions on the grid, letting the joints follow, and giving the net a
pen-centred window is what turned an unlearnable task into a solved one. The receipt
is for the version that works.

## 4. The numbers

From `receipt.json`: validation on 120 fresh signs chose 64 hidden owners (both sizes
reached 0.999 overlap with the teacher's writing); 240 held-out signs written once.

| policy | parameters | epochs | training | overlap with the sign | overlap with the teacher | agreement with the teacher |
|---|---|---|---|---|---|---|
| patch net 98-64-9, free/nudged rule | 7,055 | 8 | 157 s | 0.988 | 1.000 | 1.000 |
| MLP 98-64-9, Adam | 6,921 | 8 | 2 s | 0.988 | 1.000 | 1.000 |
| the teacher | | | | 0.988 | 1 | 1 |

Both learners reproduce the teacher move for move on fresh signs; the 0.988 against the
sign is the teacher's own score (the rasterised Z sometimes has a cell the stroke walk
skips). Per shape: bar, dash, slash, L, and V at 1.000, Z at 0.927. Wall-clock is the usual
factor, here seventy.

## 5. What is different from the MLP

Nothing in the outcome, everything in the mechanism: the MLP maps the window to a move in
one pass and learned from a backward pass; the patch net settles to rest under the window
and learned from a second settlement under a nudge. Section 6 of
[How a patch net learns](../HOW_IT_LEARNS.md) has the comparison.

## 6. The page

Press "New sign" and "Write it". The page rasterises a fresh sign the way `arm.py` does,
clamps the 7×7 window onto the net's input owners, settles the net in JavaScript for every
step, and moves the pen on the brightest output owner; the arm follows by inverse
kinematics. The right column shows what the net sees and its nine output owners at rest.

## 7. Things to try

- A pen that starts at a random cell of the stroke, not its first, and a teacher that
  heads for the nearest un-inked cell: the net must then choose a direction at junctions.
- More shapes (a triangle, a digit) in `polyline`; the window rule does not change.
- `K = 5` or `K = 3`: how small a window still writes a V?
