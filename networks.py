import torch


class DeepRecurrentNeuralNetwork(torch.nn.Module):
    def __init__(self, cell_class: type[torch.nn.RNNCell | torch.nn.LSTMCell | torch.nn.GRUCell], input_size: int, hidden_sizes: list[int], output_size: int, device: torch.device):
        """
        hidden_sizes: list like [16, 32, 20] meaning:
            layer 0: 16 hidden units
            layer 1: 32 hidden units
            layer 2: 20 hidden units
        """
        super().__init__()
        self.num_layers = len(hidden_sizes)
        self.hidden_sizes = hidden_sizes
        self.device = device

        # Create RNNCell for each layer
        self.cells = torch.nn.ModuleList()

        for i, h in enumerate(hidden_sizes):
            inp = input_size if i == 0 else hidden_sizes[i - 1]
            self.cells.append(cell_class(inp, h, device=self.device))

        # Final linear output layer
        self.fc = torch.nn.Linear(hidden_sizes[-1], output_size, device=self.device)

        self.to(self.device)

    def _init_hidden(self, batch: int, device: torch.device = None):
        """Create zero-initialized hidden state for a given batch size."""
        if device is None:
            device = self.device
        h = []
        for hs, cell in zip(self.hidden_sizes, self.cells):
            if isinstance(cell, torch.nn.LSTMCell):
                # LSTM needs (h, c)
                h.append(
                    (
                        torch.zeros(batch, hs, device=device),
                        torch.zeros(batch, hs, device=device),
                    )
                )
            else:
                # RNNCell or GRUCell
                h.append(torch.zeros(batch, hs, device=device))
        return h

    @staticmethod
    def detach_hidden(h: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]):
        """
        Return a version of hidden state detached from the computation graph.
        Useful to keep persistent hidden state across forward calls without graph growth.
        :param h: list of hidden states for each layer
        :return: detached hidden state
        """
        detached = []
        for state in h:
            if isinstance(state, tuple):
                # LSTM: (h, c)
                detached.append((state[0].detach(), state[1].detach()))
            else:
                detached.append(state.detach())
        return detached

    def forward(self, x, h=None):
        """
        x: (batch, seq_len, input_size)
        h: list of hidden states for each layer (optional)
        """
        if x.dim() != 3:
            raise ValueError("x must be shape (batch, seq_len, input_size)")

        batch, seq_len, _ = x.size()
        device = x.device if x.device is not None else self.device

        # Initialize or move hidden states to device
        if h is None:
            h = self._init_hidden(batch, device=device)
        else:
            prepared = []
            for layer_state in h:
                if isinstance(layer_state, tuple):
                    prepared.append((layer_state[0].to(device), layer_state[1].to(device)))
                else:
                    prepared.append(layer_state.to(device))
            h = prepared

        inp = None  # placeholder for per-layer input

        # Process sequence
        for t in range(seq_len):
            inp = x[:, t]  # (batch, input_size) for first layer
            for layer_idx, cell in enumerate(self.cells):
                layer_state = h[layer_idx]
                if isinstance(cell, torch.nn.LSTMCell):
                    # layer_state is a tuple (h_prev, c_prev)
                    h_prev, c_prev = layer_state
                    h_next, c_next = cell(inp, (h_prev, c_prev))
                    h[layer_idx] = (h_next, c_next)
                    inp = h_next  # Output passed to next layer
                else:
                    # RNNCell or GRUCell
                    h_next = cell(inp, layer_state)
                    h[layer_idx] = h_next
                    inp = h_next

        # inp now holds the final-layer output for last timestep: shape (batch, hidden_size_last)
        out = self.fc(inp)  # (batch, output_size)
        return out, h


class DeepRNN(DeepRecurrentNeuralNetwork):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, device: torch.device):
        super().__init__(torch.nn.RNNCell, input_size, hidden_sizes, output_size, device)


class DeepLSTM(DeepRecurrentNeuralNetwork):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, device: torch.device):
        super().__init__(torch.nn.LSTMCell, input_size, hidden_sizes, output_size, device)


class DeepGRU(DeepRecurrentNeuralNetwork):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int, device: torch.device):
        super().__init__(torch.nn.GRUCell, input_size, hidden_sizes, output_size, device)

