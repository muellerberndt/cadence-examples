"""The alpha-zero-general Connect Four net in PyTorch and its NNetWrapper."""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from . import import_azg

import_azg()
from NeuralNet import NeuralNet  # noqa: E402
from utils import AverageMeter, dotdict  # noqa: E402

ARGS = dotdict({"lr": 0.001, "dropout": 0.3, "epochs": 10, "batch_size": 64, "cuda": torch.cuda.is_available(), "num_channels": 128})


class Connect4NNet(nn.Module):
    """Four 3x3 convolutions (two padded, two not) and two dense layers, the Othello net of
    alpha-zero-general on the 6 by 7 board: the unpadded convolutions leave 2 by 3."""

    def __init__(self, game, args) -> None:
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        super().__init__()
        ch = args.num_channels
        self.conv1 = nn.Conv2d(1, ch, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ch, ch, 3, stride=1)
        self.conv4 = nn.Conv2d(ch, ch, 3, stride=1)
        self.bn1, self.bn2, self.bn3, self.bn4 = (nn.BatchNorm2d(ch) for _ in range(4))
        self.fc1 = nn.Linear(ch * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)
        return F.log_softmax(self.fc3(s), dim=1), torch.tanh(self.fc4(s))


class NNetWrapper(NeuralNet):
    """``default_args`` is what the Coach's arena copy (``__class__(game)``) is built with;
    ``load_checkpoint`` rebuilds the net when the checkpoint's width differs."""

    default_args = ARGS

    def __init__(self, game, args=None) -> None:
        self.args = args if args is not None else type(self).default_args
        self.game = game
        self.nnet = Connect4NNet(game, self.args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        self.nnet.to(self.device)

    def train(self, examples) -> None:
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)
        for epoch in range(self.args.epochs):
            self.nnet.train()
            pi_losses, v_losses = AverageMeter(), AverageMeter()
            for _ in range(int(len(examples) / self.args.batch_size)):
                ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64)).to(self.device)
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64)).to(self.device)
                out_pi, out_v = self.nnet(boards)
                l_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
                l_v = torch.sum((target_vs - out_v.view(-1)) ** 2) / target_vs.size()[0]
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                optimizer.zero_grad()
                (l_pi + l_v).backward()
                optimizer.step()
            print(f"epoch {epoch + 1}/{self.args.epochs}: pi loss {pi_losses.avg:.4f} v loss {v_losses.avg:.4f}", flush=True)

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64)).to(self.device).view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar") -> None:
        os.makedirs(folder, exist_ok=True)
        torch.save({"state_dict": self.nnet.state_dict(), "args": dict(self.args)}, os.path.join(folder, filename))

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar") -> None:
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"no model in path {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        saved = dotdict({**self.args, **{k: v for k, v in checkpoint.get("args", {}).items() if k in ("num_channels", "dropout")}})
        if saved.num_channels != self.args.num_channels:
            self.args = saved
            self.nnet = Connect4NNet(self.game, saved).to(self.device)
        self.nnet.load_state_dict(checkpoint["state_dict"])
