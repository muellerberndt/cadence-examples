"""One declared circuit, different stimulation ports and an ablation. No retraining."""
import numpy as np
import cadence as cd

wiring = cd.Wiring.from_edges(4, pre=[0, 1, 2, 3], post=[1, 2, 3, 0], sign=[.6, .6, -.6, .6])
net = cd.Settlement(wiring, cd.GradedRule(gain=1, slope=2, threshold=0, leak=1, dt=1))

for stimulated in (0, 2):
    drive = np.zeros(4)
    drive[stimulated] = 1
    state = net.settle(drive, steps=100, tolerance=1e-10)
    ablated = net.settle(drive, mask=np.array([1, 1, 0, 1]), steps=100, tolerance=1e-10)
    print(f"Drive owner {stimulated}: {state.activation.round(4)}")
    print(f"Remove owner 2:       {ablated.activation.round(4)}")

print("These predictions follow the supplied circuit rule; they are not learned biology.")
