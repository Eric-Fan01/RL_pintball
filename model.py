import torch
import torch.nn as nn
import math
# ------------------------------ Utility: weight init ------------------------------

def constant_init(module, val=0.02):
    """Fill all Conv2d / Linear weights with the same constant for deterministic debugging."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.constant_(m.weight, val)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

def normal_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)

# ------------------------------ Network blocks ------------------------------

class ResidualBlock(nn.Module):
    """Lightweight ResNet‑V2 style block for debugging with deterministic weights."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )
        constant_init(self.block)

    def forward(self, x):
        return x + self.block(x)

class DownSample(nn.Module):
    """Two‑stage strided conv + residual stack, roughly 1/4 spatial size."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.stage = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//2, 3, stride=2, padding=1, bias=False),
            ResidualBlock(out_ch//2),
            nn.Conv2d(out_ch//2, out_ch, 3, stride=2, padding=1, bias=False),
            ResidualBlock(out_ch),
        )
        constant_init(self.stage)

    def forward(self, x):
        return self.stage(x)

# ------------------------------ Representation ------------------------------

class Representation(nn.Module):
    def __init__(self, obs_shape, hidden_ch=16):
        """
        obs_shape: (H, W, C) as in Gym frame stack (e.g. 84,84,4)
        hidden_ch: channels after first conv.
        """
        super().__init__()
        in_ch = obs_shape[2]
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_ch),
            DownSample(hidden_ch, hidden_ch),
        )
        constant_init(self.trunk)

    def forward(self, x):
        return self.trunk(x)

# ------------------------------ Dynamics ------------------------------

class Dynamics(nn.Module):
    def __init__(self, state_ch: int, action_dim: int, lstm_hidden: int = 32):
        super().__init__()
        # Simple conv to blend action channel in
        self.action_conv = nn.Conv2d(state_ch + 1, state_ch, 3, padding=1, bias=False)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_ch * 21 * 21, lstm_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(lstm_hidden, state_ch * 21 * 21),
            nn.ReLU(inplace=True),
        )
        constant_init(self)

    def forward(self, state, action_plane):
        x = torch.cat([state, action_plane], 1)
        x = self.action_conv(x)
        next_state = self.head(x).view_as(state)
        # Dummy scalar reward: mean over spatial map for reproducible scalar
        reward = next_state.mean(dim=(1, 2, 3), keepdim=True)
        return reward, next_state

# ------------------------------ Prediction ------------------------------

class Prediction(nn.Module):
    def __init__(self, state_ch: int, action_dim: int, support_size: int = 21):
        """Outputs categorical value & policy logits."""
        super().__init__()
        flat = state_ch * 21 * 21
        self.head_val = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, support_size),
        )
        self.head_pol = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )
        constant_init(self)

    def forward(self, x):
        v = self.head_val(x)
        p = self.head_pol(x)
        return p, v

# ------------------------------ Debug main ------------------------------

def main():
    torch.manual_seed(0)  # Deterministic params & random tensors

    # ---- Fake environment specs ----
    obs_shape = (84, 84, 4)  # HWC
    action_dim = 9
    state_channels = 16  # matches Representation hidden_ch

    # ---- Build sub‑networks ----
    rep = Representation(obs_shape, hidden_ch=state_channels)
    dyn = Dynamics(state_channels, action_dim)
    pred = Prediction(state_channels, action_dim)

    # ---- Dummy input ----
    obs = torch.randn(1, obs_shape[2], obs_shape[0], obs_shape[1])  # NCHW
    print("Input observation:", obs.shape)

    # ---- Representation ----
    state = rep(obs)
    print("After Representation → state:", state.shape)
    print("state[0,0,0,:5] =", state[0, 0, 0, :5])

    # ---- Build dummy one‑hot action plane (choose id=3) ----
    action_id = 3
    h, w = state.shape[2:]
    action_plane = torch.full((1, 1, h, w), fill_value=(action_id + 1) / action_dim)

    # ---- Dynamics ----
    reward, next_state = dyn(state, action_plane)
    print("Reward tensor shape:", reward.shape, "value:", reward.squeeze())
    print("Next state shape:", next_state.shape)

    # ---- Prediction ----
    policy_logits, value_logits = pred(next_state)
    print("Policy logits shape:", policy_logits.shape)
    print("Value logits shape:", value_logits.shape)
    print("First 5 policy logits:", policy_logits[0, :5])
    print("First 5 value logits:", value_logits[0, :5])


if __name__ == "__main__":
    main()
