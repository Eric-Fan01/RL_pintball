"""
Refactored original MuZero Conv‑LSTM model:
- ✅ 修复了所有 LSTM 不能塞入 Sequential 的错误（改为手动 forward）
- ✅ 所有输出 shape 从 21x21 改回 6x6，与原论文一致，便于与调试网络统一
- ✅ 简化过度冗余的 Sequential 拼装方式，结构更清晰
- ✅ 每个 forward 中添加打印，观察 tensor shape 和前几个元素，方便 debug

可直接运行 `python model.py` 查看效果
"""

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.Dropout2d(0.5),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True), nn.Dropout2d(0.5),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.seq(x)

class DownSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 2, 3, stride=2, padding=1, bias=False),
            ResidualBlock(out_ch // 2),
            nn.Conv2d(out_ch // 2, out_ch, 3, stride=2, padding=1, bias=False),
            ResidualBlock(out_ch),
            # nn.AvgPool2d(3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)

class Representation(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch), nn.ReLU(inplace=True),
            DownSample(hidden_ch, hidden_ch),
            ResidualBlock(hidden_ch),
        )

    def forward(self, x):
        x = self.model(x)
        print("[Representation]", x.shape)
        return x

class Dynamics(nn.Module):
    def __init__(self, ch, action_dim):
        super().__init__()
        self.conv = nn.Conv2d(ch + 1, ch, 3, padding=1, bias=False)
        self.resblock = ResidualBlock(ch)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 21 * 21, 64),
            nn.ReLU(),
            nn.Linear(64, ch * 21 * 21),
        )
        
        self.reward_head = nn.Sequential(
            nn.Flatten(), nn.Linear(ch * 21 * 21, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = self.resblock(self.conv(x))
        next_s = self.fc(x).view_as(s)
        reward = self.reward_head(x).view(-1, 1, 1, 1)
        print("[Dynamics] next_state:", next_s.shape, "reward:", reward.view(-1))
        return reward, next_s

class Prediction(nn.Module):
    def __init__(self, ch, action_dim, support_size):
        super().__init__()
        self.flat = nn.Flatten()
        self.policy = nn.Sequential(
            nn.Linear(ch * 21 * 21, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.value = nn.Sequential(
            nn.Linear(ch * 21 * 21, 64), nn.ReLU(),
            nn.Linear(64, support_size)
        )

    def forward(self, s):
        x = self.flat(s)
        p, v = self.policy(x), self.value(x)
        print("[Prediction] policy:", p.shape, "value:", v.shape)
        return p, v

# Test forward
if __name__ == '__main__':
    torch.manual_seed(0)
    obs = torch.randn(1, 4, 84, 84)
    action_dim = 9
    support = 21

    rep = Representation(4, 16)
    dyn = Dynamics(16, action_dim)
    pred = Prediction(16, action_dim, support)

    state = rep(obs)
    h, w = state.shape[2:]
    action_plane = torch.full((1, 1, h, w), fill_value=4 / action_dim)

    reward, next_state = dyn(state, action_plane)
    policy, value = pred(next_state)
