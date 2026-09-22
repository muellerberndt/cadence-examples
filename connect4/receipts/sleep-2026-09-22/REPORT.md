# Connect Four: records by day, slow parameters by night

School: `runs/connect4/school/s1.npz`, 3000 games of seed 1, 67399 positions (the README's 50,000 games were reduced for the time budget: the generator makes about 23 positions per game at 2.5 games per second, and one night at 4,096 cells costs about 3 s per 1,000 rows). Held-out: `runs/connect4/school/test.npz`, 1500 games of seed 99; 27613 distinct positions occur in no school file, split as train.py splits them into a validation half of 13806 (chooses the checkpoint) and a reported half of 13807; 1887 held-out positions occur in the school.

Every arm: hidden 256, 32 active cells, seed 0, slow rate 0.3, batches and cues of 256 rows, one pass of the school = 263 updates. A day writes every school position once, in random order, with `write=True` at slow rate 0. A night is `RecordPatchNet.sleep` on cues drawn as train.py draws its batches (every band of stones equally often), one pass over the fixed dreams, `dawn_passes=2`, `backtrack` off. The store-off score is the emptied copy of the patch (snapshot, restore, `records.tables['y']` zeroed); it equals the slow readout alone. The score is the sign accuracy on decided positions, as in train.py.

## What a night costs

| cells | table | projection | day, s per 1,000 rows | night, s per 1,000 rows | read, s per 1,000 rows |
| --- | --- | --- | --- | --- | --- |
| 4,096 | 0.03 MB | 11 MB | 1.3 | 3.0 | 0.6 |
| 16,384 | 0.13 MB | 45 MB | 3.1 | 9.0 | 1.3 |
| 65,536 | 0.52 MB | 178 MB | 12.6 | 38.3 | 12.2 |

1,024 synthetic board readings, hidden 256, cue batches of 256, passes 1, dawn_passes 2, OMP_NUM_THREADS=2, load average 8.8 before the runs; the 65536 read was one imagine call of 1,024 rows

## Validation half, after each pass or night

| regime | point | slow updates | held-out sign, store off | store on | 3-way off | 3-way on | seconds | what the day and the night cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| school | pass 1 | 263 | 0.785 | 0.785 | 0.547 | 0.547 | 20 | no records written |
| school | pass 2 | 526 | 0.788 | 0.788 | 0.548 | 0.548 | 44 | no records written |
| school | pass 3 | 789 | 0.794 | 0.794 | 0.540 | 0.540 | 67 | no records written |
| school | pass 4 | 1052 | 0.801 | 0.801 | 0.552 | 0.552 | 91 | no records written |
| school | pass 5 | 1315 | 0.802 | 0.802 | 0.555 | 0.555 | 114 | no records written |
| school | pass 6 | 1578 | 0.803 | 0.803 | 0.569 | 0.569 | 140 | no records written |
| sleep 4,096 cells | start | 0 | 0.550 | 0.550 | 0.139 | 0.139 | 25 | settled, table empty |
| sleep 4,096 cells | night 1 bedtime | 0 | 0.550 | 0.733 | 0.139 | 0.520 | 66 | day: 67399 rows written in 34 s; store 3646 cells nonzero, rms 0.470 |
| sleep 4,096 cells | night 1 dawn | 263 | 0.755 | 0.744 | 0.521 | 0.537 | 213 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.2048 to 0.0741, 134656 dawn writes, 140 s; store after dawn 3744 cells nonzero, rms 0.382 |
| sleep 4,096 cells | night 2 bedtime | 263 | 0.755 | 0.749 | 0.521 | 0.530 | 261 | day: 67399 rows written in 40 s; store 3747 cells nonzero, rms 0.505 |
| sleep 4,096 cells | night 2 dawn | 526 | 0.759 | 0.755 | 0.515 | 0.542 | 406 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.0980 to 0.0692, 134656 dawn writes, 138 s; store after dawn 3767 cells nonzero, rms 0.430 |
| sleep 4,096 cells | night 3 bedtime | 526 | 0.759 | 0.750 | 0.515 | 0.546 | 452 | day: 67399 rows written in 38 s; store 3768 cells nonzero, rms 0.543 |
| sleep 4,096 cells | night 3 dawn | 789 | 0.768 | 0.763 | 0.543 | 0.556 | 597 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.1123 to 0.0602, 134656 dawn writes, 138 s; store after dawn 3771 cells nonzero, rms 0.458 |
| sleep 4,096 cells, 3 passes a night | start | 0 | 0.550 | 0.550 | 0.139 | 0.139 | 33 | settled, table empty |
| sleep 4,096 cells, 3 passes a night | night 1 bedtime | 0 | 0.550 | 0.733 | 0.139 | 0.520 | 92 | day: 67399 rows written in 49 s; store 3646 cells nonzero, rms 0.470 |
| sleep 4,096 cells, 3 passes a night | night 1 dawn | 789 | 0.764 | 0.744 | 0.524 | 0.536 | 311 | night: 263 cues of 256 (67328 rows dreamed), 789 updates, dream loss 0.2048 to 0.0690, 134656 dawn writes, 210 s; store after dawn 3742 cells nonzero, rms 0.378 |
| sleep 4,096 cells, one day | start | 0 | 0.550 | 0.550 | 0.139 | 0.139 | 34 | settled, table empty |
| sleep 4,096 cells, one day | night 1 bedtime | 0 | 0.550 | 0.733 | 0.139 | 0.520 | 92 | day: 67399 rows written in 49 s; store 3646 cells nonzero, rms 0.470 |
| sleep 4,096 cells, one day | night 1 dawn | 263 | 0.755 | 0.744 | 0.521 | 0.537 | 267 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.2048 to 0.0741, 134656 dawn writes, 167 s; store after dawn 3744 cells nonzero, rms 0.382 |
| sleep 4,096 cells, one day | night 2 dawn | 526 | 0.754 | 0.743 | 0.536 | 0.537 | 430 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.0574 to 0.0503, 134656 dawn writes, 154 s; store after dawn 3760 cells nonzero, rms 0.375 |
| sleep 4,096 cells, one day | night 3 dawn | 789 | 0.754 | 0.745 | 0.531 | 0.533 | 611 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.0480 to 0.0442, 134656 dawn writes, 170 s; store after dawn 3768 cells nonzero, rms 0.366 |
| sleep 4,096 cells, averaging (floor 0.5) | start | 0 | 0.550 | 0.550 | 0.139 | 0.139 | 30 | settled, table empty |
| sleep 4,096 cells, averaging (floor 0.5) | night 1 bedtime | 0 | 0.550 | 0.732 | 0.139 | 0.520 | 75 | day: 67399 rows written in 39 s; store 3646 cells nonzero, rms 0.481 |
| sleep 4,096 cells, averaging (floor 0.5) | night 1 dawn | 263 | 0.755 | 0.744 | 0.521 | 0.537 | 215 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.2050 to 0.0741, 134656 dawn writes, 132 s; store after dawn 3744 cells nonzero, rms 0.394 |
| sleep 4,096 cells, averaging (floor 0.5) | night 2 bedtime | 263 | 0.755 | 0.749 | 0.521 | 0.529 | 259 | day: 67399 rows written in 37 s; store 3747 cells nonzero, rms 0.518 |
| sleep 4,096 cells, averaging (floor 0.5) | night 2 dawn | 526 | 0.760 | 0.755 | 0.514 | 0.541 | 407 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.0983 to 0.0694, 134656 dawn writes, 141 s; store after dawn 3767 cells nonzero, rms 0.443 |
| sleep 4,096 cells, averaging (floor 0.5) | night 3 bedtime | 526 | 0.760 | 0.749 | 0.514 | 0.548 | 462 | day: 67399 rows written in 46 s; store 3768 cells nonzero, rms 0.555 |
| sleep 4,096 cells, averaging (floor 0.5) | night 3 dawn | 789 | 0.769 | 0.762 | 0.543 | 0.555 | 621 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.1120 to 0.0600, 134656 dawn writes, 151 s; store after dawn 3770 cells nonzero, rms 0.471 |
| sleep 4,096 cells, averaging (floor 0.02) | start | 0 | 0.550 | 0.550 | 0.139 | 0.139 | 46 | settled, table empty |
| sleep 4,096 cells, averaging (floor 0.02) | night 1 bedtime | 0 | 0.550 | 0.778 | 0.139 | 0.511 | 108 | day: 67399 rows written in 52 s; store 3646 cells nonzero, rms 0.237 |
| sleep 4,096 cells, averaging (floor 0.02) | night 1 dawn | 263 | 0.775 | 0.795 | 0.507 | 0.538 | 299 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.1288 to 0.0232, 134656 dawn writes, 181 s; store after dawn 3737 cells nonzero, rms 0.197 |
| sleep 4,096 cells, averaging (floor 0.02) | night 2 bedtime | 263 | 0.775 | 0.797 | 0.507 | 0.550 | 357 | day: 67399 rows written in 48 s; store 3739 cells nonzero, rms 0.243 |
| sleep 4,096 cells, averaging (floor 0.02) | night 2 dawn | 526 | 0.776 | 0.800 | 0.521 | 0.555 | 575 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.0213 to 0.0227, 134656 dawn writes, 207 s; store after dawn 3755 cells nonzero, rms 0.230 |
| sleep 16,384 cells | start | 0 | 0.550 | 0.550 | 0.139 | 0.139 | 93 | settled, table empty |
| sleep 16,384 cells | night 1 bedtime | 0 | 0.550 | 0.753 | 0.139 | 0.537 | 238 | day: 67399 rows written in 110 s; store 11215 cells nonzero, rms 0.338 |
| sleep 16,384 cells | night 1 dawn | 263 | 0.768 | 0.772 | 0.532 | 0.555 | 725 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.2139 to 0.0825, 134656 dawn writes, 462 s; store after dawn 11898 cells nonzero, rms 0.296 |
| sleep 16,384 cells | night 2 bedtime | 263 | 0.768 | 0.761 | 0.532 | 0.542 | 865 | day: 67399 rows written in 117 s; store 11955 cells nonzero, rms 0.394 |
| sleep 16,384 cells | night 2 dawn | 526 | 0.779 | 0.766 | 0.515 | 0.552 | 1378 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.1039 to 0.0787, 134656 dawn writes, 486 s; store after dawn 12117 cells nonzero, rms 0.356 |
| sleep 16,384 cells, 3 passes a night, averaging (floor 0.02) | start | 0 | 0.550 | 0.550 | 0.139 | 0.139 | 133 | settled, table empty |
| sleep 16,384 cells, 3 passes a night, averaging (floor 0.02) | night 1 bedtime | 0 | 0.550 | 0.789 | 0.139 | 0.512 | 337 | day: 67399 rows written in 173 s; store 11215 cells nonzero, rms 0.224 |
| sleep 16,384 cells, 3 passes a night, averaging (floor 0.02) | night 1 dawn | 789 | 0.779 | 0.801 | 0.501 | 0.544 | 1002 | night: 263 cues of 256 (67328 rows dreamed), 789 updates, dream loss 0.1326 to 0.0223, 134656 dawn writes, 641 s; store after dawn 11903 cells nonzero, rms 0.199 |
| sleep 65,536 cells | start | 0 | 0.555 | 0.555 | 0.144 | 0.144 | 296 | settled, table empty |
| sleep 65,536 cells | night 1 bedtime | 0 | 0.555 | 0.778 | 0.144 | 0.559 | 794 | day: 67399 rows written in 453 s; store 29363 cells nonzero, rms 0.212 |
| sleep 65,536 cells | night 1 dawn | 263 | 0.772 | 0.783 | 0.530 | 0.569 | 3063 | night: 263 cues of 256 (67328 rows dreamed), 263 updates, dream loss 0.2265 to 0.0897, 134656 dawn writes, 2204 s; store after dawn 32068 cells nonzero, rms 0.199 |

## Reported half, the checkpoint validation chose

| regime | kept at | slow updates | sign, store off | store on | mse off | mse on | 0-8 stones off/on | 8-16 | 16-24 | 24-43 | seen positions off/on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| school, 3 passes | batch 789 | 789 | 0.792 | 0.792 | 0.447 | 0.447 | 0.796/0.796 | 0.803/0.803 | 0.785/0.785 | 0.792/0.792 | 0.899/0.899 |
| school, 6 passes | batch 1578 | 1578 | 0.802 | 0.802 | 0.442 | 0.442 | 0.796/0.796 | 0.810/0.810 | 0.801/0.801 | 0.801/0.801 | 0.917/0.917 |
| sleep 4,096 cells | night 3 | 789 | 0.764 | 0.768 | 0.512 | 0.536 | 0.743/0.719 | 0.777/0.762 | 0.773/0.770 | 0.759/0.769 | 0.867/0.889 |
| sleep 4,096 cells, 3 passes a night | night 1 | 789 | 0.762 | 0.750 | 0.491 | 0.578 | 0.820/0.719 | 0.761/0.750 | 0.754/0.752 | 0.765/0.749 | 0.886/0.896 |
| sleep 4,096 cells, one day | night 1 | 263 | 0.753 | 0.750 | 0.504 | 0.575 | 0.832/0.743 | 0.748/0.747 | 0.751/0.755 | 0.753/0.748 | 0.884/0.888 |
| sleep 4,096 cells, one day | night 3 (last) | 789 | 0.753 | 0.751 | 0.531 | 0.564 | 0.796/0.719 | 0.763/0.743 | 0.749/0.761 | 0.751/0.749 | 0.883/0.886 |
| sleep 4,096 cells, averaging (floor 0.5) | night 3 | 789 | 0.767 | 0.769 | 0.508 | 0.534 | 0.737/0.719 | 0.781/0.767 | 0.774/0.772 | 0.761/0.769 | 0.870/0.889 |
| sleep 4,096 cells, averaging (floor 0.02) | night 2 | 526 | 0.776 | 0.797 | 0.472 | 0.440 | 0.790/0.760 | 0.786/0.810 | 0.769/0.793 | 0.778/0.797 | 0.892/0.925 |
| sleep 16,384 cells | night 2 | 526 | 0.782 | 0.766 | 0.465 | 0.511 | 0.754/0.689 | 0.803/0.761 | 0.773/0.769 | 0.783/0.767 | 0.888/0.905 |
| sleep 16,384 cells, 3 passes a night, averaging (floor 0.02) | night 1 | 789 | 0.774 | 0.797 | 0.470 | 0.434 | 0.754/0.754 | 0.766/0.789 | 0.770/0.795 | 0.778/0.801 | 0.897/0.937 |
| sleep 65,536 cells | night 1 | 263 | 0.768 | 0.780 | 0.473 | 0.493 | 0.665/0.701 | 0.770/0.778 | 0.771/0.775 | 0.769/0.784 | 0.895/0.937 |

## Control: the schooled parameters (3 passes) with one day of writes on top

| store | validation sign, store off | store on | reported sign, store off | store on | seen positions off/on | seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 4,096 cells, write rate 0.5 | 0.794 | 0.765 | 0.792 | 0.765 | 0.899/0.906 | 85 |
| 4,096 cells, averaging (floor 0.02) | 0.794 | 0.807 | 0.792 | 0.806 | 0.899/0.928 | 83 |
| 16,384 cells, averaging (floor 0.02) | 0.794 | 0.811 | 0.792 | 0.812 | 0.899/0.945 | 289 |

## Play, every move graded by the solver

| brain | opponent | first (w-d-l) | second (w-d-l) | optimal moves | blunders when not lost | of moves | 0-8 | 8-16 | 16-24 | 24-42 | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| school_p3 | one_ply | 10-0-0 | 10-0-0 | 0.753 | 0.044 | 160 | 0.077 | 0.019 | 0.000 | 0.000 | 238 |
| school_p3 | solver_070 | 6-0-4 | 7-1-2 | 0.753 | 0.163 | 98 | 0.268 | 0.208 | 0.000 | 0.000 | 686 |
| school_p3 | solver_100 | 0-0-10 | 0-0-10 | 0.752 | 0.500 | 20 | 0.500 | - | - | - | 363 |
| sleep_c16384 | one_ply | 10-0-0 | 10-0-0 | 0.680 | 0.109 | 174 | 0.215 | 0.021 | 0.031 | 0.000 | 467 |
| sleep_c16384 | solver_070 | 10-0-0 | 7-0-3 | 0.737 | 0.165 | 158 | 0.333 | 0.190 | 0.028 | 0.000 | 748 |
| sleep_c16384 | solver_100 | 0-0-10 | 0-0-10 | 0.755 | 0.333 | 30 | 0.333 | - | - | - | 724 |
| sleep_c16384_avg_low_p3 | one_ply | 10-0-0 | 10-0-0 | 0.768 | 0.060 | 166 | 0.101 | 0.018 | 0.043 | 0.000 | 241 |
| sleep_c16384_avg_low_p3 | solver_070 | 9-0-1 | 8-0-2 | 0.797 | 0.124 | 170 | 0.073 | 0.333 | 0.148 | 0.000 | 704 |
| sleep_c16384_avg_low_p3 | solver_100 | 0-0-10 | 0-0-10 | 0.879 | 0.228 | 57 | 0.000 | 0.765 | - | - | 894 |


## What was found

- The regime works on this task. With the corpus closed, one night of 263 slow updates from the store's own dreams takes the slow readout from 0.550 (chance on the sign of decided positions) to 0.755 at 4,096 cells, 0.768 at 16,384, 0.772 at 65,536 and 0.775 at 4,096 cells with an averaging store (validation half). The school reaches 0.785 with the same 263 updates from the school's outcomes.
- At matched slow updates the sleep-trained slow readout is below the school in every arm. Validation half, store off: 263 updates, 0.755 to 0.775 against 0.785; 526 updates, 0.759 (4,096), 0.776 (4,096 averaging) and 0.779 (16,384) against 0.788; 789 updates, 0.768 (three nights), 0.764 (three passes in one night), 0.769 (averaging at the default floor) and 0.779 (16,384 averaging, three passes) against 0.794. Reported half, kept checkpoints: 0.782 (16,384 cells, 526 updates) and 0.774 (16,384 averaging, 789 updates) against the school's 0.792 at 789 and 0.802 at 1,578.
- The night's outcome follows the day's store. What the store reads at bedtime on never-seen positions is what the dreams carry, and it orders the dawns: a bedtime store at 0.733 (4,096 cells) gives a dawn readout of 0.755; 0.753 (16,384) gives 0.768; 0.778 (65,536, and 4,096 averaging) gives 0.772 and 0.775; 0.789 (16,384 averaging) gives 0.779 with three passes. A store that reads unseen positions at 0.73 to 0.79 teaches that accuracy; the school teaches from exact outcomes.
- The dreams limit more than the updates do. Three passes over one night's fixed dreams (789 updates) gave 0.764 where one pass gave 0.755; three nights with a day before each gave 0.768. One day followed by three nights of the same store gave 0.755, 0.754 and 0.754, with the dream loss at bedtime 0.205, 0.057 and 0.048: after the first night the store's dreams are what the weights already say, and the dawn rewrite leaves the store holding what they did not take.
- The store's write rule decides what a day holds. At the patch's write rate 0.5 a cell holds its last few writers; a day of 67,399 writes into 4,096 cells (about 526 writes per cell) leaves a store that reads unseen positions at 0.733. With `record_averaging` at floor 0.02 the same day leaves 0.778, equal to the 65,536-cell last-writers store at a sixteenth of the table, and at 16,384 cells 0.789. `record_averaging` at the patch's default floor 0.5 changes nothing here (0.767 against 0.764 on the reported half): every cell is written hundreds of times a day, so the floor is the rate. The averaging store's second night adds 0.001: its dawn writes move the store by 0.02 per write, so the store keeps its residuals against the weights of the day before, and the second night's dreams are the first night's.
- The averaging store does not drown. One day of writes on top of the schooled parameters (0.792 on the reported half) lifts the read to 0.806 at 4,096 cells and 0.812 at 16,384; the last-writers store at 4,096 cells lowers it to 0.765, which is the README's result at this school size. The sleep arms with their store present read 0.797 (4,096 averaging, 526 updates) and 0.798 (16,384 averaging, 789 updates), above the schooled slow readout alone and below the schooled readout with an averaging store. The day carries more than the night transfers.
- Play. `deploy.py` carries the parameters into an empty store, so the store-off patch is what plays. Twenty games per opponent, every move graded by the solver, the same opponent seeds for each brain. The schooled patch (789 updates, 0.792) and the 16,384-cell sleep patch (526 updates, 0.782) both beat one_ply 20-0; the schooled brain's moves are optimal more often (0.753 against 0.680) and it blunders less when not lost (0.044 against 0.109; in the first eight stones 0.077 against 0.215). Against the solver at 70% the sleep-trained brain scored 17-0-3 and the schooled one 13-1-6, at the same blunder rate when not lost (0.165 and 0.163) and 0.737 against 0.753 optimal moves; four games out of twenty is inside the noise of the draw. Against the perfect solver both lost all twenty. The averaging brain (16,384 cells, floor 0.02, three passes in one night, 789 updates, 0.774 on the reported half with the store off) beat one_ply 20-0 with 0.768 of its moves optimal and 0.060 blunders when not lost, scored 17-0-3 against the solver at 70% (0.797 optimal, 0.124 blunders), and lost all twenty to the perfect solver. Its blunder rate lies between the two others against one_ply and below both against the 70% solver; with twenty games an opponent the three brains are not separated by their game scores, and the schooled brain keeps the lowest blunder rate against the weakest opponent.
- Cost. With OMP_NUM_THREADS=2 on a machine at load average 12 to 20, a night over 67,328 rows took 140 s at 4,096 cells, 462 to 486 s at 16,384 and 2,204 s at 65,536 (2.1, 7.0 and 33 s per 1,000 rows; 181 to 207 s with averaging at 4,096 and 641 s for three passes at 16,384), a day 34 to 52 s, 110 to 173 s and 453 s; the school's pass of 263 updates took 20 to 26 s. One night is six to a hundred times the school's pass at the same updates, and the day comes before it. The table is one float64 column: 0.03, 0.13 and 0.52 MB; the fixed projection the store codes with is 11, 45 and 178 MB.

## The big school: 4,444,571 positions of 200,000 solver games, on a 16-core instance

The same arms on the school the deployed brain is taught with (seed 0; held-out `big_test.npz`, 1,500 games of seed 99: 21,403 unseen positions, half validation and half reported). The sleep arms take the school as 67 days of about 66,300 positions each, with a night after every day: 259 cues of 256 rows dreamed once (`--shards 67 --nights 67`), one pass or three over the fixed dreams, 4,096 cells, last-writers store at rate 0.5. Files: `big/<arm>.training.json` and `big/<arm>.log`.

| arm | slow updates | reported unseen sign, store off | store on | 0-8 stones | 8-16 | 16-24 | 24-43 | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| school, 3 passes | 52,084 | 0.896 | 0.896 | 0.926 | 0.879 | 0.881 | 0.904 | 3,151 |
| school, 6 passes (deployed as `brain/v2.npz`) | 104,169 | 0.906 | 0.906 | 0.926 | 0.884 | 0.895 | 0.912 | 6,299 |
| sleep, 67 days and nights, one pass a night | 17,353 | 0.819 | 0.789 | 0.778 | 0.821 | 0.812 | 0.820 | 3,340 |
| sleep, 67 days and nights, three passes a night | 52,059 | 0.838 | 0.814 | 0.704 | 0.815 | 0.832 | 0.839 | 4,010 |

At the same 52,000 updates the school is 5.8 points ahead of the sleep regime (0.896 against 0.838), where it was one to three points ahead on the 67,399-position school: the gap grows with the school. Each night teaches what its day's store holds, and a day's store of 66,300 writes in 4,096 last-writer cells reads unseen positions at about 0.78 at bedtime, so 67 nights of such dreams do not add up to what 67 days of the outcomes themselves teach. Two arms with an averaging store (floor 0.02, 4,096 and 16,384 cells, three passes a night) were started on the same instance and are reported here when they end.

## Where the example's code changed

- `connect4/train_sleep.py` is new. It reuses `train.py`'s loading, held-out split and metric, and adds the emptied-store read (snapshot, restore, `records.tables["y"]` zeroed, checked equal to the slow part of the live read), the day at slow rate 0 through `ValuePatch.learn(..., rate=0.0, write=True)`, and the night through `RecordPatchNet.sleep` on cues of 256 rows and one moment each. The library needed no change.
- `connect4/patch.py`: `ValuePatch.__init__` takes `record_averaging` and passes it to the library. Nothing else changed. `deploy.py` carries a sleep-trained checkpoint's parameters into an empty store as it does the schooled one (checked: parameters equal, table empty).
- The README's sentence that bulk writes drown the store holds at write rate 0.5 and does not hold for an averaging store at floor 0.02 at this school size. The README was left as it is.
- The school is 3,000 games (67,399 positions) in place of the README's 50,000, for the time budget; the held-out file is 1,500 games of seed 99. Both are regenerated from their seeds by `school.py`.


Receipt: `runs/connect4/sleep/receipt.json` (every setting and number, the per-arm `training.json` files inlined, the bench receipts).
