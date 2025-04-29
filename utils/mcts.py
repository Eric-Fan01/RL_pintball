import numpy as np
import torch

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.children = {}  # action -> Node
        self.state = None   # stored latent state

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

def select_child(node, c_puct=1.0):
    """Use PUCT formula to select child node."""
    best_score = -float('inf')
    best_action = None
    best_child = None

    for action, child in node.children.items():
        ucb_score = child.value() + c_puct * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
        if ucb_score > best_score:
            best_score = ucb_score
            best_action = action
            best_child = child

    return best_action, best_child

def expand_node(node, network, state, action_space_size):
    """Expand a node with all possible actions."""
    policy_logits, value = network.prediction(state)
    policy = torch.softmax(policy_logits, dim=1).detach().cpu().numpy()[0]  # shape (action_space_size,)

    for action in range(action_space_size):
        node.children[action] = Node(prior=policy[action])

def backup(path, value):
    """Backpropagate value along the visited path."""
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1

def mcts_search(network, obs, num_simulations=50, action_space_size=4, device="cuda"):
    """Perform MCTS starting from given obs."""
    # obs = torch.tensor(obs.transpose(2,0,1), dtype=torch.float32, device=device).unsqueeze(0) / 255.0
    obs = obs.to(device).float() / 255.0
    state = network.representation(obs)

    root = Node(prior=1.0)
    root.state = state

    expand_node(root, network, state, action_space_size)

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # Selection
        while node.children:
            action, node = select_child(node)
            search_path.append(node)

        parent = search_path[-2]
        action_taken = [k for k,v in parent.children.items() if v == node][0]

        # Dynamics
        action_plane = torch.ones((1,1,state.shape[2],state.shape[3]), device=device) * (action_taken / action_space_size)
        reward, next_state = network.dynamics(parent.state, action_plane)
        next_state = next_state.detach()

        node.state = next_state

        # Expand
        expand_node(node, network, next_state, action_space_size)

        # Evaluate value
        _, value_logits = network.prediction(next_state)
        value_probs = torch.softmax(value_logits, dim=1)
        support = torch.linspace(-10, 10, steps=value_logits.shape[1]).to(value_logits.device)
        value = (value_probs * support).sum(dim=1).item()


        # Backup
        backup(search_path, value)

    # Pick action based on most visit counts
    visit_counts = np.array([child.visit_count for action, child in root.children.items()])
    action_probs = visit_counts / visit_counts.sum()
    action = np.argmax(action_probs)

    return action
